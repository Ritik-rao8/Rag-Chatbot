import json
import sqlite3
from uuid import uuid4

import pytest

from src.chat import AnswerResult
from src.citations import AnswerDraft, Passage
from src.config import Settings
from src.conversations import ConversationError, ConversationStore
from src.vector_store import DocumentLibrary
from tests.helpers import TestEmbeddings, make_pdf
from tests.test_chat import draft


@pytest.fixture
def saved_chat(tmp_path):
    library = DocumentLibrary(Settings(data_dir=tmp_path), TestEmbeddings())
    source, _ = library.index_pdf(
        make_pdf(["Cover page", "Returns are accepted within thirty days."]), "policy.pdf"
    )
    passage = library.read_chunks(source["document_id"])[1]
    result = AnswerResult(
        "answered",
        draft=AnswerDraft.model_validate(draft()),
        passages=[Passage(1, passage, 0.1)],
        model="test-model",
    )
    turn = {
        "id": uuid4().hex,
        "question": "Refund?",
        "scope": "policy.pdf",
        "scope_id": source["document_id"],
        "result": result,
        "library_signature": [source["document_id"]],
    }
    store = ConversationStore(tmp_path)
    options = {
        "scope_id": source["document_id"],
        "library_signature": [source["document_id"]],
        "profile": library.collection_name,
    }
    return library, source, store, turn, options


def test_save_and_reopen_preserves_answers_quotes_and_page(saved_chat):
    library, source, store, turn, options = saved_chat
    identifier = store.save("Return notes", [turn], **options)
    loaded = ConversationStore(library.settings.data_dir).load(identifier)
    assert loaded["title"] == "Return notes"
    result = loaded["turns"][0]["result"]
    assert (
        result.draft.statements[0].evidence[0].quote == "Returns are accepted within thirty days."
    )
    assert result.passages[0].document.metadata["page_number"] == 2
    assert loaded["turns"][0]["scope_id"] == source["document_id"]
    assert loaded["turns"][0]["library_signature"] == [source["document_id"]]
    assert store.list()[0]["turn_count"] == 1


def test_autosave_updates_same_conversation_and_names_are_unique(saved_chat):
    _, _, store, turn, options = saved_chat
    identifier = store.save("Notes", [turn], **options)
    second = {**turn, "id": uuid4().hex}
    assert store.save("Notes", [turn, second], conversation_id=identifier, **options) == identifier
    assert len(store.list()) == 1 and store.list()[0]["turn_count"] == 2
    with pytest.raises(ConversationError, match="already exists"):
        store.save("notes", [turn], **options)
    store.rename(identifier, "Updated notes")
    assert store.load(identifier)["title"] == "Updated notes"
    store.delete(identifier)
    assert store.list() == []


def test_source_removal_keeps_saved_excerpts_and_chat_removal_keeps_sources(saved_chat):
    library, source, store, turn, options = saved_chat
    identifier = store.save("Archive", [turn], **options)
    library.delete_source(source["document_id"])
    assert store.load(identifier)["turns"][0]["result"].status == "answered"
    assert not library.pdf_path(source["document_id"]).exists()
    source, _ = library.index_pdf(make_pdf(["Another source."]), "other.pdf")
    store.delete(identifier)
    assert library.pdf_path(source["document_id"]).exists()


def test_corrupt_or_unverified_saved_content_is_rejected(saved_chat):
    _, _, store, turn, options = saved_chat
    identifier = store.save("Notes", [turn], **options)
    with sqlite3.connect(store.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT snapshot FROM conversations WHERE id=?", (identifier,)
            ).fetchone()[0]
        )
        payload["turns"][0]["result"]["draft"]["statements"][0]["evidence"][0]["quote"] = (
            "Private-corruption-marker"
        )
        connection.execute(
            "UPDATE conversations SET snapshot=? WHERE id=?", (json.dumps(payload), identifier)
        )
    with pytest.raises(ConversationError, match="damaged") as error:
        store.load(identifier)
    assert "Private-corruption-marker" not in str(error.value)
    assert len(store.list()) == 1  # A corrupt entry cannot crash the conversation list.


def test_empty_names_and_unavailable_ids_are_safe(saved_chat):
    _, _, store, turn, options = saved_chat
    for name in ["", "x" * 81, "name\nother"]:
        with pytest.raises(ConversationError):
            store.save(name, [turn], **options)
    with pytest.raises(ConversationError):
        store.save("Empty", [], **options)
    with pytest.raises(ConversationError):
        store.load("../../outside")
    with pytest.raises(ConversationError):
        store.save("Missing", [turn], conversation_id=uuid4().hex, **options)
