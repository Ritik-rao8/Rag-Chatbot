import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.config import ROOT, Settings
from src.sidebar_ui import matching_sources
from src.vector_store import DocumentLibrary
from tests.helpers import TestEmbeddings, make_pdf
from tests.test_web import HTML


@pytest.fixture
def library_app(tmp_path, monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    embeddings = TestEmbeddings()
    library = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    alpha, _ = library.index_pdf(make_pdf(["Refund policy."]), "Alpha.pdf")
    beta, _ = library.index_pdf(make_pdf(["Shipping policy."]), "Beta.pdf")
    monkeypatch.setattr(
        "src.web_ingestion.fetch_html", lambda _: (HTML, "https://example.com/policy")
    )
    library.index_url("https://example.com/policy")
    monkeypatch.setattr("src.vector_store.load_embeddings", lambda *_: embeddings)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    yield library, app, alpha, beta
    st.cache_resource.clear()


def test_filter_matches_name_or_domain_and_combines_type(library_app):
    library, _, _, _ = library_app
    sources = library.list_sources()
    assert len(matching_sources(sources, "alpha", "PDFs")) == 1
    assert matching_sources(sources, "alpha", "Webpages") == []
    assert len(matching_sources(sources, "EXAMPLE.COM", "Webpages")) == 1
    assert [s["filename"] for s in matching_sources(sources, "", "PDFs")] == [
        "Alpha.pdf",
        "Beta.pdf",
    ]


def test_source_actions_choose_the_exact_document_and_tab(library_app):
    _, app, alpha, beta = library_app
    app.button(key=f"inspect-source-{beta['document_id']}").click().run()
    assert not app.exception
    assert app.session_state["inspect_source"] == beta["document_id"]
    assert app.session_state["workspace_tabs"] == "Inspect documents"
    app.button(key=f"chat-source-{alpha['document_id']}").click().run()
    assert not app.exception
    assert app.session_state["answer_source"] == alpha["document_id"]
    assert app.session_state["workspace_tabs"] == "Chat"


def test_search_empty_state_and_reset_preserve_sources(library_app):
    library, app, _, _ = library_app
    app.text_input(key="source_filter").set_value("not-a-source").run()
    assert not app.exception
    assert any("No matching sources" in info.value for info in app.info)
    next(b for b in app.button if b.label == "Reset filters").click().run()
    assert not app.exception
    assert app.text_input(key="source_filter").value == ""
    assert len(library.list_sources()) == 3
    assert sum(b.label == "Chat with this source" for b in app.button) == 3


def test_remove_requires_confirmation_cancel_and_filters_do_not_delete(library_app):
    library, app, alpha, beta = library_app
    key = alpha["document_id"]
    app.button(key=f"remove-source-{key}").click().run()
    assert not app.exception and len(library.list_sources()) == 3
    app.button(key=f"cancel-remove-{key}").click().run()
    assert not app.exception and len(library.list_sources()) == 3
    assert not any(b.label == "Confirm removal" for b in app.button)
    app.button(key=f"remove-source-{key}").click().run()
    app.text_input(key="source_filter").set_value("Beta").run()
    assert not app.exception
    assert "pending_source_delete" not in app.session_state
    assert not any(b.label == "Confirm removal" for b in app.button)
    assert library.pdf_path(key).exists() and library.pdf_path(beta["document_id"]).exists()


def test_confirmation_removes_only_the_selected_source(library_app):
    library, app, alpha, beta = library_app
    key = beta["document_id"]
    app.button(key=f"remove-source-{key}").click().run()
    app.button(key=f"confirm-remove-{key}").click().run()
    assert not app.exception
    remaining = {s["document_id"] for s in library.list_sources()}
    assert key not in remaining and alpha["document_id"] in remaining
    assert not library.pdf_path(key).exists()
    assert app.metric[0].value == "2"
    assert any("Removed Beta.pdf" in notice.value for notice in app.success)
