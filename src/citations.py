"""Validate model-selected evidence against retrieved text, never invented metadata."""

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: StrictInt = Field(ge=1)
    quote: str = Field(min_length=1, max_length=3000)


class Statement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)
    evidence: list[Evidence] = Field(min_length=1, max_length=6)


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answerable: StrictBool = Field(
        description="True when passages support an answer, including a negative/no answer. False only when evidence is insufficient; then statements must be empty."
    )
    statements: list[Statement] = Field(max_length=8)


@dataclass
class Passage:
    source_id: int
    document: Document
    distance: float


class CitationValidationError(ValueError):
    """A fixed reason code; contains no model text or private source content."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def normalized(text: str) -> str:
    return " ".join(text.split())


def validate_citations(draft: AnswerDraft, passages: list[Passage]) -> None:
    """Validate provenance and literal quotes; this does not prove semantic support."""
    if not draft.answerable:
        if draft.statements:
            raise CitationValidationError("contradictory_abstention")
        return
    if not draft.statements:
        raise CitationValidationError("missing_statements")
    sources = {passage.source_id: passage.document for passage in passages}
    for statement in draft.statements:
        if not statement.text.strip():
            raise CitationValidationError("blank_statement")
        for evidence in statement.evidence:
            document = sources.get(evidence.source_id)
            if document is None:
                raise CitationValidationError("unknown_source")
            if evidence.quote.strip() and evidence.quote in document.page_content:
                continue
            # PDF extraction can join words that the model separates (or vice versa).
            # Accept only identical non-whitespace characters, and replace the model's
            # quote with the actual source span before displaying/verifying it.
            compact_quote = re.sub(r"\s", "", evidence.quote)
            positions = [i for i, char in enumerate(document.page_content) if not char.isspace()]
            compact_source = "".join(document.page_content[i] for i in positions)
            start = compact_source.find(compact_quote) if compact_quote else -1
            if start < 0:
                raise CitationValidationError("quote_mismatch")
            end = start + len(compact_quote) - 1
            evidence.quote = document.page_content[positions[start] : positions[end] + 1]
