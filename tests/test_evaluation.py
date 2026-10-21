import json
from types import SimpleNamespace

from langchain_core.documents import Document

from evaluation.run import HERE, score_case
from src.citations import AnswerDraft, Evidence, Passage, Statement


def test_benchmark_has_independent_sources_and_valid_followup_parents():
    cases = json.loads((HERE / "questions.json").read_text())
    sources = {p.stem for p in (HERE / "corpus").glob("*.html")}
    assert len(cases) == 10 and len(sources) > 4
    seen = set()
    for case in cases:
        assert case["id"] not in seen
        assert set(case["expected_sources"]) <= sources
        if "parent" in case:
            assert case["parent"] in seen
        seen.add(case["id"])
    assert sum(not c["expected_sources"] for c in cases) == 2


def test_retrieval_and_citation_scores_distinguish_wrong_source():
    case = {"expected_sources": ["returns", "shipping"], "term_groups": [["thirty", "30"]]}
    passages = [
        Passage(
            i,
            Document(page_content="Thirty days.", metadata={"url": f"https://example.test/{slug}"}),
            0.1,
        )
        for i, slug in enumerate(["returns", "care"], 1)
    ]
    result = SimpleNamespace(
        status="answered",
        message="",
        passages=passages,
        draft=AnswerDraft(
            answerable=True,
            statements=[
                Statement(
                    text="Thirty days.", evidence=[Evidence(source_id=2, quote="Thirty days.")]
                )
            ],
        ),
    )
    scores = score_case(case, result)
    assert scores["retrieval_recall"] == 0.5
    assert scores["expected_sources_cited"] is False
    assert scores["reference_terms_present"] is True
    assert scores["expected_behavior"] is True  # Behavior alone is not correctness.


def test_errors_and_abstentions_are_not_confused():
    case = {"expected_sources": [], "term_groups": []}
    result = SimpleNamespace(
        status="validation_failed", message="Withheld", passages=[], draft=None
    )
    assert score_case(case, result)["expected_behavior"] is False
    result.status = "insufficient_evidence"
    scores = score_case(case, result)
    assert scores["expected_behavior"] is True
    assert scores["retrieval_recall"] is None
    assert scores["expected_sources_cited"] is None
