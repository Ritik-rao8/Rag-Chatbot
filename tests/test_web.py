import socket
from email.message import Message
from io import BytesIO
from unittest.mock import Mock, patch

import pytest

from src.config import Settings
from src.source_ui import location
from src.vector_store import DocumentLibrary
from src.web_ingestion import WebError, extract_webpage, fetch_html, public_target
from tests.helpers import TestEmbeddings, make_pdf

HTML = b"<html><title>Returns policy</title><nav>Ignore menu</nav><main><h1>Refunds</h1><p>Returns are accepted within thirty days. Keep the original receipt.</p><script>ignore script</script></main></html>"


def test_html_extracts_readable_content_without_fake_pdf_pages():
    result = extract_webpage(HTML, "https://example.com/policy")
    assert result.filename == "Returns policy"
    assert "Ignore menu" not in result.pages[0].page_content
    assert "ignore script" not in result.pages[0].page_content
    assert result.total_pages == 0
    assert location(result.pages[0].metadata) == "Webpage"
    assert result.pages[0].metadata["url"] == "https://example.com/policy"
    with pytest.raises(WebError, match="Too little"):
        extract_webpage(b"<html><script>only JavaScript</script></html>", "https://example.com")


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "::1",
        "fc00::1",
        "0.0.0.0",
        "::ffff:127.0.0.1",
        "224.0.0.1",
        "ff02::1",
    ],
)
def test_private_destinations_are_blocked(ip):
    with (
        patch(
            "src.web_ingestion.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))],
        ),
        pytest.raises(WebError, match="not allowed"),
    ):
        public_target("http://internal.example/")


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com",
        "https://name:password@example.com",
        "http://example.com:8080",
        "https://example.com/\nheader",
    ],
)
def test_unsupported_urls_are_blocked(url):
    with pytest.raises(WebError):
        public_target(url)


def public_dns(*args, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]


class Response:
    def __init__(self, body=HTML, status=200, content_type="text/html", redirect=None):
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if redirect:
            self.headers["Location"] = redirect
        self.content = BytesIO(body)

    def getheader(self, key, default=None):
        return self.headers.get(key, default)

    def read1(self, size):
        return self.content.read(size)


def test_fetch_pins_public_ip_and_limits_bytes(monkeypatch):
    monkeypatch.setattr("src.web_ingestion.socket.getaddrinfo", public_dns)
    connection = Mock()
    connection.getresponse.return_value = Response()
    factory = Mock(return_value=connection)
    monkeypatch.setattr("src.web_ingestion.PinnedConnection", factory)
    content, url = fetch_html("https://example.com/policy#part")
    assert content == HTML and url == "https://example.com/policy"
    assert factory.call_args.args[2] == "93.184.216.34"
    connection.close.assert_called_once()
    connection.getresponse.return_value = Response()
    with pytest.raises(WebError, match="2 MB"):
        fetch_html("https://example.com", max_bytes=10)


def test_redirect_revalidates_destination_and_content_type(monkeypatch):
    def resolver(host, *args, **kwargs):
        return public_dns() if host == "example.com" else [(2, 1, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr("src.web_ingestion.socket.getaddrinfo", resolver)
    connection = Mock()
    connection.getresponse.return_value = Response(status=302, redirect="http://localhost/private")
    factory = Mock(return_value=connection)
    monkeypatch.setattr("src.web_ingestion.PinnedConnection", factory)
    with pytest.raises(WebError, match="not allowed"):
        fetch_html("https://example.com")
    assert factory.call_count == 1
    connection.getresponse.return_value = Response(content_type="application/pdf")
    with pytest.raises(WebError, match="not an HTML"):
        fetch_html("https://example.com")


def test_url_persistence_deduplication_and_deletion(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.web_ingestion.fetch_html", lambda _: (HTML, "https://example.com/policy")
    )
    library = DocumentLibrary(Settings(data_dir=tmp_path), TestEmbeddings())
    page, added = library.index_url("https://example.com/policy")
    assert added and page["source_type"] == "url"
    duplicate, added = library.index_url("https://example.com/policy")
    assert not added and page == duplicate
    pdf, _ = library.index_pdf(make_pdf(["Shipping takes five days."]), "delivery.pdf")
    result, _ = library.search("refund", document_id=page["document_id"])[0]
    assert result.metadata["source_type"] == "url"
    reopened = DocumentLibrary(Settings(data_dir=tmp_path))
    with patch(
        "src.vector_store.load_embeddings",
        side_effect=AssertionError("Deletion must not load a model"),
    ):
        reopened.delete_source(page["document_id"])
    assert not library._source_dir(page["document_id"]).exists()
    assert [source["document_id"] for source in library.list_sources()] == [pdf["document_id"]]
    assert library.store.get(where={"document_id": page["document_id"]})["ids"] == []
    assert library.search("shipping")[0][0].metadata["document_id"] == pdf["document_id"]
    _, added = library.index_url("https://example.com/policy")
    assert added


def test_pdf_deletion_removes_files_and_rejects_invalid_ids(tmp_path):
    library = DocumentLibrary(Settings(data_dir=tmp_path), TestEmbeddings())
    source, _ = library.index_pdf(make_pdf(["Refund policy"]), "policy.pdf")
    library.delete_source(source["document_id"])
    assert library.list_sources() == [] and library.search("refund") == []
    assert not library.pdf_path(source["document_id"]).exists()
    with pytest.raises(ValueError):
        library.delete_source("../outside")
