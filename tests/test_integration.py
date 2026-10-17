"""Opt-in verification with the real local model, including a process restart."""

import subprocess
import sys

import pytest

from src.config import Settings, get_settings
from src.embeddings import load_embeddings
from src.vector_store import DocumentLibrary
from tests.helpers import make_pdf


@pytest.mark.integration
def test_real_pdf_retrieval_after_restart(tmp_path):
    defaults = get_settings()
    embeddings = load_embeddings(defaults.model_name, defaults.data_dir / "models")
    settings = Settings(data_dir=tmp_path)
    library = DocumentLibrary(settings, embeddings)
    content = make_pdf(
        [
            "Returns and refunds. Customers may return a purchase within thirty days for a full refund.",
            "Delivery information. Standard shipping takes five business days. Express delivery takes two days.",
            "Plant care. Water the garden weekly and keep seedlings in direct sunlight.",
        ]
    )
    manifest, added = library.index_pdf(content, "sample-policies.pdf")
    assert added and manifest["chunk_count"] == 3
    for question, page in [
        ("How long do I have to return an item?", 1),
        ("When will standard delivery arrive?", 2),
        ("How often should I water my plants?", 3),
    ]:
        result, _ = library.search(question, k=1)[0]
        assert result.metadata["page_number"] == page
    code = """
import sys
from pathlib import Path
from src.config import Settings
from src.embeddings import load_embeddings
from src.vector_store import DocumentLibrary
settings = Settings(data_dir=Path(sys.argv[1]))
embeddings = load_embeddings(settings.model_name, Path(sys.argv[2]))
library = DocumentLibrary(settings, embeddings)
assert len(library.list_sources()) == 1
result, distance = library.search('How long do I have to return an item?', k=1)[0]
assert result.metadata['page_number'] == 1
assert 'thirty days' in result.page_content
print('Fresh-process retrieval passed:', result.metadata['source'], 'page', result.metadata['page_number'])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(defaults.data_dir / "models")],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    print(completed.stdout)
