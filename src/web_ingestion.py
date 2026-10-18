"""Fetch a single public HTML page with validated, pinned destination addresses."""

import hashlib
import http.client
import ipaddress
import socket
import ssl
import time
from datetime import UTC, datetime
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import certifi
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.ingestion import ExtractedSource


class WebError(ValueError):
    """An actionable URL ingestion error."""


def public_target(url: str):
    if len(url) > 4096 or any(ord(char) < 33 for char in url):
        raise WebError("Enter a valid public HTTP or HTTPS URL without whitespace.")
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").encode("idna").decode("ascii")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        if (
            parts.scheme not in {"http", "https"}
            or not host
            or parts.username is not None
            or parts.password is not None
        ):
            raise WebError("Use a public HTTP or HTTPS URL without embedded credentials.")
        if port not in {80, 443}:
            raise WebError("Only standard web ports (80 and 443) are supported.")
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        ips = list(dict.fromkeys(item[4][0] for item in addresses))
        if not ips or any(
            not ipaddress.ip_address(ip).is_global or ipaddress.ip_address(ip).is_multicast
            for ip in ips
        ):
            raise WebError("Local, private, and reserved network addresses are not allowed.")
    except WebError:
        raise
    except (ValueError, UnicodeError, OSError):
        raise WebError(
            "Could not resolve this public URL. Check the address and your connection."
        ) from None
    authority = f"[{host}]" if ":" in host else host
    if port != (443 if parts.scheme == "https" else 80):
        authority += f":{port}"
    canonical = urlunsplit(
        (
            parts.scheme,
            authority,
            quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~"),
            quote(parts.query, safe="%=&/?@:;+,$!()*'-._~"),
            "",
        )
    )
    return canonical, host, port, ips[0]


class PinnedConnection(http.client.HTTPConnection):
    def __init__(self, host, port, ip, *, secure, timeout):
        super().__init__(host, port, timeout=timeout)
        self.ip = ip
        self.secure = secure

    def connect(self):
        # The numeric IP is the one validated above; DNS cannot redirect this connection.
        self.sock = socket.create_connection((self.ip, self.port), self.timeout)
        if self.secure:
            context = ssl.create_default_context(cafile=certifi.where())
            self.sock = context.wrap_socket(self.sock, server_hostname=self.host)


def fetch_html(url: str, *, max_bytes=2 * 1024 * 1024, timeout=20):
    deadline = time.monotonic() + timeout
    for _ in range(4):
        canonical, host, port, ip = public_target(url.strip())
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WebError("This webpage took too long to respond.")
        connection = PinnedConnection(
            host, port, ip, secure=canonical.startswith("https:"), timeout=remaining
        )
        try:
            parts = urlsplit(canonical)
            connection.request(
                "GET",
                parts.path + (f"?{parts.query}" if parts.query else ""),
                headers={
                    "User-Agent": "LocalDocumentExplorer/0.3",
                    "Accept": "text/html, application/xhtml+xml",
                    "Accept-Encoding": "identity",
                },
            )
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise WebError("The webpage returned a redirect without a destination.")
                url = urljoin(canonical, location)
                continue
            if response.status != 200:
                raise WebError(
                    f"The webpage returned HTTP {response.status}. Try a publicly accessible page."
                )
            if response.headers.get_content_type() not in {"text/html", "application/xhtml+xml"}:
                raise WebError(
                    "This URL is not an HTML webpage. Download PDFs and use the PDF uploader."
                )
            if response.getheader("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise WebError(
                    "The server returned compressed content despite requesting plain HTML."
                )
            data = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WebError("This webpage took too long to download.")
                if connection.sock:
                    connection.sock.settimeout(remaining)
                block = response.read1(min(65536, max_bytes + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
                if len(data) > max_bytes:
                    raise WebError("This webpage exceeds the 2 MB download limit.")
            return bytes(data), canonical
        except (OSError, http.client.HTTPException):
            raise WebError(
                "Could not fetch this webpage securely. Check the URL or try another page."
            ) from None
        finally:
            connection.close()
    raise WebError("This webpage has too many redirects (maximum three).")


def extract_webpage(html: bytes, url: str) -> ExtractedSource:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True)[:200] if soup.title else url
    for element in soup.select(
        "script, style, nav, footer, header, aside, form, noscript, svg, template, [hidden], [aria-hidden='true']"
    ):
        element.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    text = body.get_text("\n", strip=True).replace("\x00", "")
    if len(text.strip()) < 40:
        raise WebError(
            "Too little readable text was found. Login-only or JavaScript-rendered pages are not supported."
        )
    document_id = hashlib.sha256(f"{url}\n{title}\n{text}".encode()).hexdigest()
    fetched_at = datetime.now(UTC).isoformat()
    page = Document(
        page_content=text,
        metadata={
            "document_id": document_id,
            "source": title or url,
            "source_type": "url",
            "url": url,
            "fetched_at": fetched_at,
            "page_number": 1,
            "page_label": "",
            "total_pages": 0,
        },
    )
    return ExtractedSource(document_id, title or url, [page], 0, [], "url", url, fetched_at)


def load_webpage(url: str) -> ExtractedSource:
    html, final_url = fetch_html(url)
    return extract_webpage(html, final_url)
