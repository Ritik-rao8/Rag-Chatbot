import json
from unittest.mock import Mock

import httpx
import pytest
from groq import APIConnectionError, AuthenticationError, RateLimitError
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.chat import AnswerError, answer_question, conversation_context


class FakeModel:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output if isinstance(output, AIMessage) else AIMessage(content=json.dumps(output))


def draft(source_id=1, quote="Returns are accepted within thirty days."):
    return {
        "answerable": True,
        "statements": [
            {
                "text": "You can return an item within thirty days.",
                "evidence": [{"source_id": source_id, "quote": quote}],
            }
        ],
    }


@pytest.fixture
def library():
    return Mock(
        search=Mock(
            return_value=[
                (
                    Document(
                        page_content="Returns are accepted within thirty days. Keep your receipt.",
                        metadata={
                            "source": "policy.pdf",
                            "page_number": 7,
                            "page_label": "iv",
                            "chunk_id": "original",
                        },
                    ),
                    0.2,
                )
            ]
        )
    )


def test_grounded_answer_uses_retrieved_metadata_and_filters(library):
    model = FakeModel(draft(), {"supported": True, "complete": True})
    answer = answer_question(
        library, "When can I return an item?", document_id="chosen", model=model
    )
    assert answer.status == "answered"
    assert answer.passages[0].document.metadata["page_number"] == 7
    assert answer.passages[0].document.metadata["page_label"] == "iv"
    library.search.assert_called_once_with(
        "When can I return an item?", k=4, document_id="chosen", hybrid=True
    )
    assert len(model.calls) == 2
    assert "untrusted data" in model.calls[0][0].content
    assert json.loads(model.calls[0][1].content)["passages"][0]["source_id"] == 1


@pytest.mark.parametrize(
    "response",
    [
        draft(source_id=99),
        draft(quote="Returns accepted for ninety days."),
        draft(quote="   "),
        draft(source_id=True),
        {"answerable": True, "statements": []},
        {"answerable": False, "statements": draft()["statements"]},
        {"answerable": "true", "statements": []},
        {"answerable": True, "statements": [{"text": "No citation", "evidence": []}]},
    ],
)
def test_unverifiable_output_is_never_displayed_as_an_answer(library, response):
    model = FakeModel(response, response)
    result = answer_question(library, "Refund?", model=model)
    assert result.status == "validation_failed" and result.draft is None
    assert len(model.calls) in {1, 2}


def test_malformed_json_fails_closed(library):
    result = answer_question(
        library,
        "Refund?",
        model=FakeModel(AIMessage(content="not JSON"), AIMessage(content="not JSON")),
    )
    assert result.status == "validation_failed"


def test_whitespace_normalized_quotes_are_valid(library):
    result = answer_question(
        library,
        "Refund?",
        model=FakeModel(
            draft(quote="Returns are accepted\nwithin thirty days."),
            {"supported": True, "complete": True},
        ),
    )
    assert result.status == "answered"


def test_verifier_rejects_valid_quote_with_unsupported_claim(library):
    response = draft()
    response["statements"][0]["text"] = "Returns are allowed for ninety days."
    result = answer_question(
        library, "Refund?", model=FakeModel(response, {"supported": False, "complete": True})
    )
    assert result.status == "insufficient_evidence"
    assert result.draft is None


def test_abstention_and_empty_library_do_not_need_verification(library):
    model = FakeModel({"answerable": False, "statements": []})
    assert (
        answer_question(library, "What is the CEO's age?", model=model).status
        == "insufficient_evidence"
    )
    assert len(model.calls) == 1
    library.search.return_value = []
    unused = FakeModel()
    assert answer_question(library, "Refund?", model=unused).status == "insufficient_evidence"
    assert not unused.calls


def test_missing_key_is_actionable(library, monkeypatch):
    monkeypatch.setattr("src.chat.groq_configuration", lambda: ("", "test-model"))
    with pytest.raises(AnswerError, match="GROQ_API_KEY"):
        answer_question(library, "Refund?")


@pytest.mark.parametrize(
    "kind, expected", [(AuthenticationError, "rejected"), (RateLimitError, "rate limit")]
)
def test_provider_errors_do_not_expose_raw_responses(library, kind, expected):
    response = httpx.Response(
        401 if kind is AuthenticationError else 429,
        request=httpx.Request("POST", "https://api.groq.com"),
    )
    error = kind("SENSITIVE-TEST-MARKER", response=response, body={"prompt": "private text"})
    with pytest.raises(AnswerError, match=expected) as caught:
        answer_question(library, "Refund?", model=FakeModel(error))
    assert "SENSITIVE" not in str(caught.value)
    assert "private text" not in str(caught.value)


def test_connection_failure_and_truncation_are_errors_not_abstentions(library):
    error = APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
    with pytest.raises(AnswerError, match="reach Groq"):
        answer_question(library, "Refund?", model=FakeModel(error))
    with pytest.raises(AnswerError, match="cut short"):
        answer_question(
            library,
            "Refund?",
            model=FakeModel(AIMessage(content="{}", response_metadata={"finish_reason": "length"})),
        )


def test_followup_rewrites_then_retrieves_without_history_as_evidence(library):
    model = FakeModel(
        {"question": "What is the deadline for returning an item?"},
        draft(),
        {"supported": True, "complete": True},
    )
    history = [{"question": "Can I return an item?", "answer": "Yes, a receipt is required."}]
    result = answer_question(library, "How long do I have for that?", history=history, model=model)
    assert result.status == "answered"
    assert result.resolved_question == "What is the deadline for returning an item?"
    library.search.assert_called_once_with(
        result.resolved_question, k=4, document_id=None, hybrid=True
    )
    assert len(model.calls) == 3
    assert "history" in json.loads(model.calls[0][1].content)
    assert "history" not in json.loads(model.calls[1][1].content)


def test_invalid_rewrite_does_not_search(library):
    with pytest.raises(AnswerError, match="standalone"):
        answer_question(
            library,
            "What about that?",
            history=[{"question": "Refund?"}],
            model=FakeModel({"question": " "}),
        )
    library.search.assert_not_called()


def test_history_is_bounded_and_stops_at_scope_changes(library):
    result = answer_question(
        library, "Refund?", model=FakeModel(draft(), {"supported": True, "complete": True})
    )
    turns = [{"scope_id": "all", "question": f"Question {i}", "result": result} for i in range(6)]
    assert len(conversation_context(turns, "all")) == 3
    assert conversation_context(turns, "other") == []
    turns[-2]["scope_id"] = "other"
    assert len(conversation_context(turns, "all")) == 1


@pytest.mark.parametrize(
    "output, code, message",
    [
        (draft(source_id=99), "unknown_source", "source ID"),
        (draft(quote="Invented quote"), "quote_mismatch", "match"),
        (AIMessage(content="not JSON"), "answer_format", "structured format"),
    ],
)
def test_validation_failure_reports_safe_specific_reason(library, output, code, message):
    result = answer_question(library, "Refund?", model=FakeModel(output, output))
    assert result.status == "validation_failed"
    assert result.validation_issue == code
    assert message in result.message
    assert result.draft is None
    assert "Invented quote" not in result.message
    assert len(result.passages) == 1


def test_bad_verifier_response_is_not_reported_as_bad_citations(library):
    result = answer_question(
        library, "Refund?", model=FakeModel(draft(), {"supported": "yes"}, {"supported": "yes"})
    )
    assert result.status == "validation_failed"
    assert result.validation_issue == "verification_format"
    assert "grounding-check" in result.message
    assert result.draft is None


def test_quote_mismatch_gets_one_fresh_attempt_and_still_requires_verification(library):
    model = FakeModel(draft(quote="Not in source"), draft(), {"supported": True, "complete": True})
    result = answer_question(library, "Refund?", model=model)
    assert result.status == "answered"
    assert len(model.calls) == 3
    payload = json.loads(model.calls[1][1].content)
    assert "quote_mismatch" in payload["validation_feedback"]
    assert "Not in source" not in json.dumps(payload)
    assert json.loads(model.calls[2][1].content)["draft"]["answerable"] is True


def test_citation_retry_is_bounded(library):
    model = FakeModel(draft(quote="Wrong"), draft(quote="Still wrong"))
    result = answer_question(library, "Refund?", model=model)
    assert result.status == "validation_failed"
    assert result.validation_issue == "quote_mismatch"
    assert result.draft is None
    assert len(model.calls) == 2


def test_pdf_joined_words_restore_exact_source_quote_before_verification(library):
    library.search.return_value[0][
        0
    ].page_content = "The perceptron applies theactivation function."
    response = draft(quote="The perceptron applies the activation function.")
    response["statements"][0]["text"] = "A perceptron applies an activation function."
    model = FakeModel(response, {"supported": True, "complete": True})
    result = answer_question(library, "What is a perceptron?", model=model)
    assert result.status == "answered"
    actual = "The perceptron applies theactivation function."
    assert result.draft.statements[0].evidence[0].quote == actual
    verified = json.loads(model.calls[1][1].content)
    assert verified["draft"]["statements"][0]["evidence"][0]["quote"] == actual


def test_spacing_recovery_does_not_accept_changed_words_or_punctuation(library):
    for quote in [
        "Returns are not accepted within thirty days.",
        "Returns are accepted within thirty days!",
    ]:
        model = FakeModel(draft(quote=quote), draft(quote=quote))
        result = answer_question(library, "Refund?", model=model)
        assert result.status == "validation_failed"
        assert result.validation_issue == "quote_mismatch"


def test_standalone_topic_avoids_rewrite_but_keeps_verification(library):
    model = FakeModel(draft(), {"supported": True, "complete": True})
    result = answer_question(
        library,
        "What is a return policy?",
        model=model,
        history=[{"question": "Shipping?", "answer": "Five days."}],
    )
    assert result.status == "answered"
    assert result.api_calls == 2
    assert result.resolved_question == "What is a return policy?"
    assert result.elapsed_seconds >= 0
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "question",
    [
        "What is that?",
        "Explain its limitations",
        "What about that?",
        "And how long does it take?",
        "What is the deadline?",
    ],
)
def test_dependent_questions_still_require_rewrite(question):
    from src.chat import needs_rewrite

    assert needs_rewrite(question)


def test_real_client_uses_distinct_strict_schemas_without_relaxing_local_limits():
    from src.chat import GENERATION_PROMPT, VERIFICATION_PROMPT, GroqModel, _invoke
    from src.citations import AnswerDraft

    client = Mock()
    client.bind.return_value.invoke.return_value = AIMessage(content="{}")
    model = GroqModel(client, True)
    _invoke(model, GENERATION_PROMPT, {"question": "Refund?"})
    generated = client.bind.call_args.kwargs["response_format"]["json_schema"]
    assert generated["name"] == "AnswerDraft" and generated["strict"] is True
    assert generated["schema"]["additionalProperties"] is False
    assert "maxItems" not in generated["schema"]["properties"]["statements"]
    assert AnswerDraft.model_json_schema()["properties"]["statements"]["maxItems"] == 8
    _invoke(model, VERIFICATION_PROMPT, {})
    assert client.bind.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_strict_provider_rejection_falls_back_to_json_but_keeps_local_validation():
    from groq import BadRequestError

    from src.chat import GroqModel
    from src.citations import AnswerDraft

    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
    error = BadRequestError(
        "private provider response", response=response, body={"code": "json_validate_failed"}
    )
    client = Mock()
    client.bind.return_value.invoke.side_effect = [error, AIMessage(content="{}")]
    model = GroqModel(client, True)
    model.invoke([], AnswerDraft)
    assert client.bind.call_count == 2
    assert client.bind.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_incomplete_supported_answer_is_repaired_and_checked_again(library):
    model = FakeModel(
        draft(),
        {"supported": True, "complete": False},
        draft(),
        {"supported": True, "complete": True},
    )
    result = answer_question(library, "Refund?", model=model)
    assert result.status == "answered" and result.api_calls == 4
    payload = json.loads(model.calls[2][1].content)
    assert "incomplete" in payload["validation_feedback"]
    assert "previous_draft" in payload


def test_persistently_incomplete_answer_is_withheld(library):
    model = FakeModel(
        draft(),
        {"supported": True, "complete": False},
        draft(),
        {"supported": True, "complete": False},
    )
    result = answer_question(library, "Refund?", model=model)
    assert result.status == "insufficient_evidence"
    assert result.draft is None and result.api_calls == 4


def test_supported_negative_answer_repairs_contradictory_abstention(library):
    broken = {"answerable": False, "statements": draft()["statements"]}
    model = FakeModel(broken, draft(), {"supported": True, "complete": True})
    result = answer_question(library, "Refund?", model=model)
    assert result.status == "answered" and result.api_calls == 3
    assert "negative/no" in json.loads(model.calls[1][1].content)["validation_feedback"]
