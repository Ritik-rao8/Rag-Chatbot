import subprocess
import sys
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.chunking import chunk_pages
from src.config import Settings
from src.ingestion import PDFError, extract_pdf
from src.vector_store import DocumentLibrary
from tests.helpers import TestEmbeddings, make_pdf


def test_page_numbers_survive_blank_pages_and_filename_is_sanitized():
    result = extract_pdf(make_pdf(["Refund policy.", "", "Shipping details."]), "../../policy.pdf")
    assert result.filename == "policy.pdf"
    assert result.total_pages == 3
    assert result.empty_pages == [2]
    assert [page.metadata["page_number"] for page in result.pages] == [1, 2, 3]
    assert result.pages[2].page_content == "Shipping details."
    assert result.pages[2].metadata["page_label"] == "3"


@pytest.mark.parametrize(
    ("content", "message"),
    [(b"", "empty"), (b"not a pdf", "valid PDF"), (b"%PDF-broken", "Could not read")],
)
def test_invalid_pdfs_have_actionable_errors(content, message):
    with pytest.raises(PDFError, match=message):
        extract_pdf(content, "bad.pdf")


def test_empty_encrypted_and_oversized_pdfs():
    with pytest.raises(PDFError, match="No extractable text"):
        extract_pdf(make_pdf([""]), "scan.pdf")
    with pytest.raises(PDFError, match="Password-protected"):
        extract_pdf(make_pdf(["Secret"], encrypted=True), "locked.pdf")
    with pytest.raises(PDFError, match="exceeds"):
        extract_pdf(make_pdf(["Text"]), "large.pdf", max_bytes=10)
    with pytest.raises(PDFError, match="limit is 1"):
        extract_pdf(make_pdf(["One", "Two"]), "long.pdf", max_pages=1)


def test_chunks_keep_provenance_and_fit_token_budget():
    embeddings = TestEmbeddings()
    pages = [
        Document(page_content="refund " * 60, metadata={"document_id": "a", "page_number": 1}),
        Document(page_content="shipping " * 30, metadata={"document_id": "a", "page_number": 3}),
    ]
    chunks = chunk_pages(pages, embeddings.tokenizer, chunk_tokens=20, overlap_tokens=4)
    assert len(chunks) > 2
    assert all(0 < chunk.metadata["token_count"] <= 20 for chunk in chunks)
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    for chunk in chunks:
        assert ("refund" in chunk.page_content) == (chunk.metadata["page_number"] == 1)
        assert ("shipping" in chunk.page_content) == (chunk.metadata["page_number"] == 3)


def test_duplicate_filtering_and_persistence(tmp_path):
    settings = Settings(data_dir=tmp_path)
    library = DocumentLibrary(settings, TestEmbeddings())
    content = make_pdf(["Refund policy: returns within thirty days.", "Shipping takes five days."])
    manifest, added = library.index_pdf(content, "policy.pdf")
    assert added
    same, added = library.index_pdf(content, "renamed.pdf")
    assert not added and same == manifest
    other, _ = library.index_pdf(make_pdf(["Garden plants need sunlight."]), "garden.pdf")
    reopened = DocumentLibrary(settings, TestEmbeddings())
    result, distance = reopened.search("refund", k=1)[0]
    assert result.metadata["page_number"] == 1
    assert result.metadata["source"] == "policy.pdf"
    assert distance < 0.01
    assert reopened.pdf_path(manifest["document_id"]).read_bytes() == content
    assert len(reopened.read_pages(manifest["document_id"])) == 2
    assert len(reopened.read_chunks(manifest["document_id"])) == 2
    filtered = reopened.search("refund", document_id=other["document_id"])
    assert all(doc.metadata["document_id"] == other["document_id"] for doc, _ in filtered)
    # Read Chroma from a fresh process, independent of its in-process client cache.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import chromadb, sys; "
                "client=chromadb.PersistentClient(path=sys.argv[1]); "
                "assert client.get_collection(sys.argv[2]).count() == 3"
            ),
            str(tmp_path / "chroma"),
            library.collection_name,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_failed_index_is_not_committed_and_can_be_retried(tmp_path):
    library = DocumentLibrary(Settings(data_dir=tmp_path), TestEmbeddings())
    content = make_pdf(["Refund policy."])
    store = library.store
    actual_add = store.add_documents

    def add_then_fail(*args, **kwargs):
        actual_add(*args, **kwargs)
        raise RuntimeError("Simulated interruption")

    with (
        patch.object(store, "add_documents", side_effect=add_then_fail),
        pytest.raises(RuntimeError, match="Simulated"),
    ):
        library.index_pdf(content, "policy.pdf")
    assert library.list_sources() == []
    assert store.get()["ids"] == []
    _, added = library.index_pdf(content, "policy.pdf")
    assert added and len(library.list_sources()) == 1


def test_search_validation_and_empty_library(tmp_path):
    library = DocumentLibrary(Settings(data_dir=tmp_path), TestEmbeddings())
    assert library.search("refund") == []
    with pytest.raises(ValueError, match="Enter a question"):
        library.search("   ")
    with pytest.raises(ValueError, match="between 1 and 10"):
        library.search("refund", k=0)
    library.index_pdf(make_pdf(["Refund policy"]), "policy.pdf")
    with pytest.raises(ValueError, match="not in this library"):
        library.search("refund", document_id="missing")
    with pytest.raises(ValueError, match="too long"):
        library.search("refund " * 300)


def test_search_excludes_uncommitted_chunks(tmp_path):
    library = DocumentLibrary(Settings(data_dir=tmp_path), TestEmbeddings())
    manifest, _ = library.index_pdf(make_pdf(["Refund policy"]), "policy.pdf")
    library.store.add_documents(
        [Document(page_content="Garden plants", metadata={"document_id": "orphan"})],
        ids=["unfinished-ingestion"],
    )
    results = library.search("garden")
    assert results
    assert all(doc.metadata["document_id"] == manifest["document_id"] for doc, _ in results)
