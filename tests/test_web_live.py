"""Opt-in full webpage ingestion using public example.com and a temporary library."""

import os

import pytest

from src.config import Settings, get_settings
from src.embeddings import load_embeddings
from src.vector_store import DocumentLibrary


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_WEB_LIVE") != "1", reason="Set RUN_WEB_LIVE=1 for network access"
)
def test_public_webpage_to_real_search(tmp_path):
    defaults = get_settings()
    embeddings = load_embeddings(defaults.model_name, defaults.data_dir / "models")
    library = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    manifest, added = library.index_url("https://example.com")
    assert added and manifest["source_type"] == "url"
    assert manifest["filename"] == "Example Domain"
    result, _ = library.search("What is this domain used for?", k=1)[0]
    assert result.metadata["url"] == "https://example.com/"
    assert "documentation" in result.page_content.lower()
    reopened = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    assert len(reopened.list_sources()) == 1
    assert reopened.search("example domain", k=1)
