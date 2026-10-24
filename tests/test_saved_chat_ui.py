from unittest.mock import Mock

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.chat import AnswerResult
from src.citations import AnswerDraft, Passage
from src.config import ROOT, Settings
from src.conversations import ConversationStore
from src.vector_store import DocumentLibrary
from tests.helpers import TestEmbeddings, make_pdf
from tests.test_chat import draft


@pytest.fixture
def chat_app(tmp_path, monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    embeddings = TestEmbeddings()
    library = DocumentLibrary(Settings(data_dir=tmp_path), embeddings)
    source, _ = library.index_pdf(
        make_pdf(["Cover page", "Returns are accepted within thirty days.", "Third page"]),
        "policy.pdf",
    )
    monkeypatch.setattr("src.vector_store.load_embeddings", lambda *_: embeddings)
    monkeypatch.setattr("src.chat_ui.groq_configuration", lambda: ("test-key", "test-model"))
    result = AnswerResult(
        "answered",
        draft=AnswerDraft.model_validate(draft()),
        passages=[Passage(1, library.read_chunks(source["document_id"])[1], 0.1)],
    )
    answer = Mock(return_value=result)
    monkeypatch.setattr("src.chat_ui.answer_question", answer)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.chat_input(key="chat_prompt").set_value("Refund?").run()
    yield app, library, source, answer
    st.cache_resource.clear()


def test_save_open_fresh_session_and_autosave_new_reply(chat_app):
    app, library, _, answer = chat_app
    app.text_input(key="conversation_name").set_value("Policy notes")
    next(b for b in app.button if b.label == "Save conversation").click().run()
    assert not app.exception
    store = ConversationStore(library.settings.data_dir)
    identifier = store.list()[0]["id"]
    app.chat_input(key="chat_prompt").set_value("Receipt?").run()
    assert store.load(identifier)["turns"][-1]["question"] == "Receipt?"
    assert len(store.load(identifier)["turns"]) == 2
    fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not fresh.chat_message
    next(b for b in fresh.button if b.label == "Open conversation").click().run()
    assert not fresh.exception
    assert len(fresh.chat_message) == 4
    assert fresh.session_state["active_conversation_title"] == "Policy notes"
    assert answer.call_count == 2  # Saving and reopening never call Groq.


def test_citation_preview_opens_at_cited_page_and_navigates_without_generation(chat_app):
    app, _, _, answer = chat_app
    next(b for b in app.button if b.label == "Preview PDF page 2").click().run()
    assert not app.exception
    picker = next(x for x in app.number_input if str(x.key).startswith("preview-page-"))
    assert picker.value == 2
    assert any("PDF page 2 of 3" in image.captions for image in app.get("image"))
    next(b for b in app.button if b.label == "Next").click().run()
    assert not app.exception
    assert next(x for x in app.number_input if str(x.key).startswith("preview-page-")).value == 3
    assert answer.call_count == 1


def test_reopened_chat_with_deleted_source_is_readable_and_uses_fresh_context(chat_app):
    app, library, source, answer = chat_app
    app.text_input(key="conversation_name").set_value("Archived notes")
    next(b for b in app.button if b.label == "Save conversation").click().run()
    # Add a replacement before removing the old source; archived citations must remain snapshots.
    library.index_pdf(make_pdf(["Another policy."]), "replacement.pdf")
    library.delete_source(source["document_id"])
    app.run()
    next(b for b in app.button if b.label == "Open conversation").click().run()
    assert not app.exception
    assert any("Archived excerpt" in item.value for item in app.info)
    assert not any(b.label == "Preview PDF page 2" for b in app.button)
    app.chat_input(key="chat_prompt").set_value("What about that?").run()
    assert not app.exception
    assert answer.call_args.kwargs["history"] == []
