"""Local, versioned conversation snapshots. No pickle or API credentials are stored."""

import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.chat import AnswerResult
from src.citations import AnswerDraft, Passage, validate_citations

MAX_TURNS = 50
SourceId = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ConversationError(ValueError):
    """Safe message for a local persistence failure."""


class StoredPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: int = Field(ge=1, strict=True)
    text: str = Field(max_length=20000)
    metadata: dict
    distance: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def check_metadata(self):
        if not re.fullmatch(r"[0-9a-f]{64}", str(self.metadata.get("document_id", ""))):
            raise ValueError("Invalid saved source ID")
        page = self.metadata.get("page_number")
        if type(page) is not int or page < 1:
            raise ValueError("Invalid saved page")
        if not isinstance(self.metadata.get("source"), str):
            raise ValueError("Missing saved source name")  # noqa: TRY004 - Pydantic wraps value errors
        return self

    def restore(self):
        return Passage(
            self.source_id, Document(page_content=self.text, metadata=self.metadata), self.distance
        )


class StoredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence", "validation_failed", "error"]
    draft: AnswerDraft | None = None
    passages: list[StoredPassage] = Field(default_factory=list, max_length=10)
    message: str = Field(default="", max_length=3000)
    model: str = Field(default="", max_length=200)
    resolved_question: str = Field(default="", max_length=1200)
    validation_issue: str = Field(default="", max_length=100)
    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    api_calls: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_answer(self):
        if self.status == "answered":
            if self.draft is None or not self.draft.answerable:
                raise ValueError("A saved answer needs supporting evidence")
            if len({p.source_id for p in self.passages}) != len(self.passages):
                raise ValueError("Duplicate saved citation IDs")
            validate_citations(self.draft, [p.restore() for p in self.passages])
        elif self.draft is not None:
            raise ValueError("Unvalidated draft in saved conversation")
        return self

    def restore(self):
        return AnswerResult(
            **self.model_dump(exclude={"passages", "draft"}),
            draft=self.draft,
            passages=[p.restore() for p in self.passages],
        )


class StoredTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    question: str = Field(min_length=1, max_length=1200)
    scope: str = Field(max_length=1000)
    scope_id: str = Field(pattern=r"^(all|[0-9a-f]{64})$")
    library_signature: list[SourceId]
    result: StoredAnswer

    def restore(self):
        return {**self.model_dump(exclude={"result"}), "result": self.result.restore()}


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    profile: str = Field(max_length=200)
    scope_id: str = Field(pattern=r"^(all|[0-9a-f]{64})$")
    library_signature: list[SourceId]
    turns: list[StoredTurn] = Field(min_length=1, max_length=MAX_TURNS)


def snapshot(turns, scope_id, library_signature, profile):
    rows = []
    for turn in turns:
        result = turn["result"]
        rows.append(
            {
                **{key: turn[key] for key in ("id", "question", "scope", "scope_id")},
                "library_signature": turn.get("library_signature", sorted(library_signature)),
                "result": {
                    **{
                        key: getattr(result, key)
                        for key in (
                            "status",
                            "message",
                            "model",
                            "resolved_question",
                            "validation_issue",
                            "elapsed_seconds",
                            "api_calls",
                        )
                    },
                    "draft": result.draft.model_dump() if result.draft else None,
                    "passages": [
                        {
                            "source_id": p.source_id,
                            "text": p.document.page_content,
                            "metadata": p.document.metadata,
                            "distance": p.distance,
                        }
                        for p in result.passages
                    ],
                },
            }
        )
    return Snapshot.model_validate(
        {
            "profile": profile,
            "scope_id": scope_id,
            "library_signature": sorted(library_signature),
            "turns": rows,
        }
    )


class ConversationStore:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.path = data_dir / "conversations.sqlite3"
        try:
            with self._connect() as connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, title_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    turn_count INTEGER NOT NULL, snapshot TEXT NOT NULL)""")
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as exc:
            raise ConversationError("Could not open local conversation storage.") from exc

    @contextmanager
    def _connect(self):
        try:
            connection = sqlite3.connect(self.path, timeout=10)
            connection.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise ConversationError("Could not open local conversation storage.") from exc
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _id(conversation_id):
        if not re.fullmatch(r"[0-9a-f]{32}", str(conversation_id)):
            raise ConversationError("This saved conversation is not available.")
        return conversation_id

    @staticmethod
    def _title(title):
        title = title.strip()
        if not 1 <= len(title) <= 80 or any(ord(c) < 32 for c in title):
            raise ConversationError("Use a conversation name between 1 and 80 characters.")
        return title

    def list(self):
        try:
            with self._connect() as connection:
                return [
                    dict(row)
                    for row in connection.execute(
                        "SELECT id,title,created_at,updated_at,turn_count FROM conversations ORDER BY updated_at DESC,id"
                    )
                ]
        except sqlite3.Error as exc:
            raise ConversationError("Could not read saved conversations.") from exc

    def save(self, title, turns, *, scope_id, library_signature, profile, conversation_id=None):
        title = self._title(title)
        if not turns:
            raise ConversationError("Send a message before saving this conversation.")
        try:
            payload = snapshot(turns, scope_id, library_signature, profile).model_dump_json()
        except (ValueError, TypeError, KeyError) as exc:
            raise ConversationError("This conversation could not be saved safely.") from exc
        if len(payload.encode()) > 8 * 1024 * 1024:
            raise ConversationError("This conversation is too large to save.")
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                if conversation_id:
                    conversation_id = self._id(conversation_id)
                    cursor = connection.execute(
                        "UPDATE conversations SET title=?,title_key=?,updated_at=?,turn_count=?,snapshot=? WHERE id=?",
                        (title, title.casefold(), now, len(turns), payload, conversation_id),
                    )
                    if cursor.rowcount != 1:
                        raise ConversationError(
                            "This conversation was removed. Save it with a new name."
                        )
                else:
                    conversation_id = uuid4().hex
                    connection.execute(
                        "INSERT INTO conversations VALUES (?,?,?,?,?,?,?)",
                        (conversation_id, title, title.casefold(), now, now, len(turns), payload),
                    )
            return conversation_id
        except sqlite3.IntegrityError as exc:
            raise ConversationError(
                "A conversation with that name already exists. Choose another name."
            ) from exc
        except sqlite3.Error as exc:
            raise ConversationError(
                "Could not save this conversation. Your current chat is still open."
            ) from exc

    def load(self, conversation_id):
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM conversations WHERE id=?", (self._id(conversation_id),)
                ).fetchone()
            if row is None:
                raise ConversationError("This saved conversation is no longer available.")
            if len(row["snapshot"].encode()) > 8 * 1024 * 1024:
                raise ValueError("Snapshot too large")
            payload = Snapshot.model_validate_json(row["snapshot"])
            return {
                **{key: row[key] for key in ("id", "title", "created_at", "updated_at")},
                "profile": payload.profile,
                "scope_id": payload.scope_id,
                "library_signature": payload.library_signature,
                "turns": [turn.restore() for turn in payload.turns],
            }
        except (sqlite3.Error, ValueError, TypeError) as exc:
            if isinstance(exc, ConversationError):
                raise
            raise ConversationError(
                "This saved conversation cannot be opened. Its data may be damaged."
            ) from exc

    def rename(self, conversation_id, title):
        title = self._title(title)
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "UPDATE conversations SET title=?,title_key=?,updated_at=? WHERE id=?",
                    (
                        title,
                        title.casefold(),
                        datetime.now(UTC).isoformat(),
                        self._id(conversation_id),
                    ),
                )
                if cursor.rowcount != 1:
                    raise ConversationError("This saved conversation is no longer available.")
        except sqlite3.IntegrityError as exc:
            raise ConversationError("A conversation with that name already exists.") from exc
        except sqlite3.Error as exc:
            raise ConversationError("Could not rename this conversation.") from exc

    def delete(self, conversation_id):
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM conversations WHERE id=?", (self._id(conversation_id),)
                )
        except sqlite3.Error as exc:
            raise ConversationError("Could not remove this saved conversation.") from exc
