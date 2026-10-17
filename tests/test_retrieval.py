from unittest.mock import Mock

from langchain_core.documents import Document

from src.config import Settings
from src.retrieval import KeywordIndex, fuse
from src.vector_store import DocumentLibrary
from tests.helpers import TestEmbeddings, make_pdf


def doc(text, key, source="one"):
    return Document(page_content=text, metadata={"chunk_id": key, "document_id": source})


def test_keyword_index_matches_rare_term_and_respects_scope():
    rare = doc("Backpropagation uses gradients.", "rare")
    index = KeywordIndex(
        [
            doc("Shipping takes days.", "shipping"),
            rare,
            doc("Backpropagation secret.", "private", "two"),
        ]
    )
    assert index.search("What is backpropagation?", {"one"}) == [rare]
    assert index.search("What is a unicorn?", {"one"}) == []


def test_fusion_retrieves_lexical_candidate_and_deduplicates_overlap():
    semantic = [doc(f"General topic {i}", str(i)) for i in range(12)]
    rare = doc("Exact technical term ABC123", "rare")
    duplicate = doc("Exact technical term ABC123", "duplicate")
    selected = fuse(semantic, [rare, duplicate], 4)
    assert rare in selected
    assert duplicate not in selected
    assert len(selected) == 4


def test_hybrid_candidates_include_keyword_only_vector_and_rebuild_after_delete(tmp_path):
    library = DocumentLibrary(Settings(data_dir=tmp_path), TestEmbeddings())
    target, _ = library.index_pdf(make_pdf(["QuasarXYZ technical definition."]), "rare.pdf")
    other, _ = library.index_pdf(make_pdf(["Ordinary general topic."]), "other.pdf")
    semantic_doc = library.read_chunks(other["document_id"])[0]
    library.store.similarity_search_with_score = Mock(return_value=[(semantic_doc, 0.2)])
    result = library.search("QuasarXYZ", k=2, hybrid=True)
    assert any(item.metadata["document_id"] == target["document_id"] for item, _ in result)
    assert all(isinstance(distance, float) for _, distance in result)
    library.delete_source(target["document_id"])
    result = library.search("QuasarXYZ", hybrid=True)
    assert all(item.metadata["document_id"] != target["document_id"] for item, _ in result)
