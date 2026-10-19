from streamlit.testing.v1 import AppTest

from src.config import ROOT, Settings
from src.vector_store import DocumentLibrary
from tests.helpers import TestEmbeddings, make_pdf
from tests.test_chat import FakeModel, draft


def test_empty_app_starts_without_downloading_model(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "Document Explorer"
    assert any("Upload a text-based PDF" in info.value for info in app.info)
    assert not (tmp_path / "models").exists()


def test_inspection_and_search_ui(tmp_path, monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    embeddings = TestEmbeddings()
    library = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    library.index_pdf(
        make_pdf(["Refund policy: thirty days.", "Shipping takes five days."]), "policy.pdf"
    )
    monkeypatch.setattr("src.vector_store.load_embeddings", lambda *_: embeddings)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert any("Refund policy" in entry.value for entry in app.text)
    next(field for field in app.text_input if field.label == "Question or search phrase").set_value(
        "refund"
    )
    next(button for button in app.button if button.label == "Search").click()
    app.run()
    assert not app.exception
    assert app.subheader[0].value == "1. policy.pdf — PDF page 1"
    assert any("Cosine distance" in entry.value for entry in app.caption)
    st.cache_resource.clear()


def test_answer_ui_citations_persist_and_clear(tmp_path, monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    embeddings = TestEmbeddings()
    library = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    library.index_pdf(make_pdf(["Returns are accepted within thirty days."]), "policy.pdf")
    monkeypatch.setattr("src.vector_store.load_embeddings", lambda *_: embeddings)
    monkeypatch.setattr("src.chat.groq_configuration", lambda: ("unit-test-key", "test-model"))
    monkeypatch.setattr("src.chat_ui.groq_configuration", lambda: ("unit-test-key", "test-model"))
    model = FakeModel(draft(), {"supported": True, "complete": True})
    monkeypatch.setattr("src.chat.create_model", lambda *_: model)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.chat_input(key="chat_prompt").set_value("When can I return an item?")
    app.run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("[1](#answer-" in item.value for item in app.markdown)
    assert any("PDF page 1" in item.label for item in app.expander)
    app.run()
    assert len(model.calls) == 2  # reruns never send a duplicate API request
    next(button for button in app.button if button.label == "New chat").click()
    app.run()
    assert not app.chat_message
    assert len(library.list_sources()) == 1
    st.cache_resource.clear()


def test_webpage_citations_and_deletion_clear_chat(tmp_path, monkeypatch):
    import streamlit as st

    from tests.test_web import HTML

    st.cache_resource.clear()
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    embeddings = TestEmbeddings()
    library = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    monkeypatch.setattr(
        "src.web_ingestion.fetch_html", lambda _: (HTML, "https://example.com/policy")
    )
    library.index_url("https://example.com/policy")
    monkeypatch.setattr("src.vector_store.load_embeddings", lambda *_: embeddings)
    monkeypatch.setattr("src.chat.groq_configuration", lambda: ("unit-test-key", "test-model"))
    monkeypatch.setattr("src.chat_ui.groq_configuration", lambda: ("unit-test-key", "test-model"))
    monkeypatch.setattr(
        "src.chat.create_model",
        lambda *_: FakeModel(draft(), {"supported": True, "complete": True}),
    )
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.chat_input(key="chat_prompt").set_value("When can I return an item?")
    app.run()
    assert not app.exception
    assert any("· Webpage" in item.label for item in app.expander)
    assert not app.number_input
    next(button for button in app.button if button.label == "Remove source").click()
    app.run()
    assert len(library.list_sources()) == 1
    next(button for button in app.button if button.label == "Confirm removal").click().run()
    assert not app.exception and not app.chat_message
    assert library.list_sources() == []
    st.cache_resource.clear()


def test_chat_error_retry_replaces_failed_turn_and_does_not_resubmit_on_rerun(
    tmp_path, monkeypatch
):
    import streamlit as st

    from src.chat import AnswerError

    st.cache_resource.clear()
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    embeddings = TestEmbeddings()
    library = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    library.index_pdf(make_pdf(["Returns are accepted within thirty days."]), "policy.pdf")
    monkeypatch.setattr("src.vector_store.load_embeddings", lambda *_: embeddings)
    monkeypatch.setattr("src.chat_ui.groq_configuration", lambda: ("test-key", "test-model"))
    from unittest.mock import Mock

    from src.chat import AnswerResult
    from src.citations import AnswerDraft, Passage

    response = AnswerResult(
        "answered",
        draft=AnswerDraft.model_validate(draft()),
        passages=[
            Passage(1, library.read_chunks(library.list_sources()[0]["document_id"])[0], 0.1)
        ],
    )
    answer = Mock(side_effect=[AnswerError("Could not reach Groq."), response])
    monkeypatch.setattr("src.chat_ui.answer_question", answer)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.chat_input(key="chat_prompt").set_value("Refund?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any("Could not reach" in item.value for item in app.warning)
    next(button for button in app.button if button.label == "Retry answer").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert answer.call_count == 2
    app.run()
    assert answer.call_count == 2
    st.cache_resource.clear()


def test_export_includes_answer_sources_and_no_provider_configuration():
    from langchain_core.documents import Document

    from src.chat import AnswerResult
    from src.chat_ui import export_chat
    from src.citations import AnswerDraft, Passage

    result = AnswerResult(
        "answered",
        draft=AnswerDraft.model_validate(draft()),
        passages=[
            Passage(
                1,
                Document(
                    page_content="Returns are accepted within thirty days.",
                    metadata={"source": "policy.pdf", "page_number": 7},
                ),
                0.1,
            )
        ],
    )
    text = export_chat([{"question": "Refund?", "scope": "All documents", "result": result}])
    assert "Refund?" in text and "policy.pdf · PDF page 7" in text
    assert "thirty days" in text
    assert "API_KEY" not in text
