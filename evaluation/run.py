"""Run with python -m evaluation.run; only synthetic fixtures are sent to Groq."""

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from src.chat import AnswerError, answer_question, groq_configuration
from src.config import Settings, get_settings
from src.embeddings import load_embeddings
from src.vector_store import DocumentLibrary
from src.web_ingestion import extract_webpage

HERE = Path(__file__).resolve().parent


def seed_library(directory: Path) -> DocumentLibrary:
    defaults = get_settings()
    embeddings = load_embeddings(defaults.model_name, defaults.data_dir / "models")
    library = DocumentLibrary(Settings(data_dir=directory), embeddings)
    for path in sorted((HERE / "corpus").glob("*.html")):
        extracted = extract_webpage(path.read_bytes(), f"https://example.test/{path.stem}")
        # Trusted local fixtures exercise extraction/indexing without fetching fictional URLs.
        with library._lock:
            library._index_source(extracted, extracted.pages[0].page_content.encode())
    return library


def score_case(case, result):
    retrieved = {Path(p.document.metadata["url"]).name for p in result.passages}
    expected = set(case["expected_sources"])
    text = " ".join(s.text for s in result.draft.statements) if result.draft else ""
    cited_ids = (
        {e.source_id for s in result.draft.statements for e in s.evidence}
        if result.draft
        else set()
    )
    cited = {
        Path(p.document.metadata["url"]).name for p in result.passages if p.source_id in cited_ids
    }
    return {
        "retrieval_recall": len(expected & retrieved) / len(expected) if expected else None,
        "expected_sources_cited": expected <= cited if expected else None,
        "reference_terms_present": all(
            any(term in text.lower() for term in group) for group in case["term_groups"]
        )
        if expected
        else None,
        "expected_behavior": result.status == ("answered" if expected else "insufficient_evidence"),
        "answer": text or result.message,
    }


def write_report(report, path):
    rows = report["cases"]
    relevant = [row for row in rows if row.get("retrieval_recall") is not None]
    recall = statistics.mean(row["retrieval_recall"] for row in relevant) if relevant else 0
    lines = [
        "# Evaluation results",
        "",
        f"Run: {report['timestamp']} · Model: `{report['model']}`",
        "",
        f"- Expected answer/abstention behavior: {sum(row.get('expected_behavior', False) for row in rows)}/{len(rows)}.",
        f"- Mean expected-source recall@4: {recall:.1%} across {len(relevant)} completed answerable cases.",
        f"- Errors: {sum('error' in row for row in rows)}.",
        "",
        "This is a small synthetic smoke benchmark, not an accuracy estimate. Term matching checks coverage only; inspect answers and quotes for semantic correctness. Retrieval and citation checks use independently specified source IDs. The runtime model-based verifier is not an independent evaluator.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']}: {row['question']}",
                "",
                f"Reference: {row['reference_answer']}",
                "",
                f"Status: {row.get('status', 'error')} · Time: {row['seconds']:.2f}s",
                "",
                row.get("answer", row.get("error", "")),
                "",
                f"Checks: behavior={row.get('expected_behavior')}, source recall={row.get('retrieval_recall')}, expected citations={row.get('expected_sources_cited')}, reference terms={row.get('reference_terms_present')}",
                "",
            ]
        )
        for citation in row.get("evidence", []):
            lines.extend([f"- {citation['source']}: {citation['quote']}"])
        lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--questions", type=Path, default=HERE / "questions.json")
    parser.add_argument(
        "--include-challenges",
        action="store_true",
        help="Append six condition/exclusion challenge cases",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0,
        help="Optional seconds between cases to respect account quotas",
    )
    parser.add_argument(
        "--seed-demo", type=Path, help="Seed synthetic sources only, then exit (no Groq calls)"
    )
    args = parser.parse_args()
    if args.seed_demo:
        if any(args.seed_demo.glob("*/sources/*/manifest.json")):
            raise SystemExit("Choose an empty demo directory to avoid mixing libraries.")
        library = seed_library(args.seed_demo.resolve())
        # Share the local model cache; documents and vectors remain isolated.
        demo_cache = args.seed_demo.resolve() / "models"
        if not demo_cache.exists():
            demo_cache.symlink_to(
                (get_settings().data_dir / "models").resolve(), target_is_directory=True
            )
        print(f"Seeded {len(library.list_sources())} synthetic sources in {args.seed_demo}")
        return
    if not groq_configuration()[0]:
        raise SystemExit("Configure GROQ_API_KEY in .env first.")
    cases = json.loads(args.questions.read_text())
    if args.include_challenges:
        cases += json.loads((HERE / "challenge-questions.json").read_text())
    fingerprint = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((HERE / "corpus").glob("*.html")))
        + args.questions.read_bytes()
        + ((HERE / "challenge-questions.json").read_bytes() if args.include_challenges else b"")
    ).hexdigest()
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": groq_configuration()[1],
        "embedding_model": get_settings().model_name,
        "chunk_tokens": 220,
        "overlap_tokens": 32,
        "k": 4,
        "retrieval": "semantic + BM25 reciprocal-rank fusion",
        "fixture_sha256": fingerprint,
        "python": platform.python_version(),
        "cases": [],
    }
    completed = {}
    with tempfile.TemporaryDirectory(prefix="rag-evaluation-") as temporary:
        library = seed_library(Path(temporary))
        report["source_count"] = len(library.list_sources())
        report["chunk_count"] = sum(s["chunk_count"] for s in library.list_sources())
        for case in cases:
            if report["cases"] and args.interval > 0:
                time.sleep(min(args.interval, 30))
            started = time.perf_counter()
            row = dict(case)
            try:
                history = []
                if case.get("parent"):
                    parent = completed.get(case["parent"])
                    if not parent or parent.get("status") != "answered":
                        raise ValueError(
                            "Parent question did not produce an answer; follow-up not evaluated."
                        )
                    history = [{"question": parent["question"], "answer": parent["answer"]}]
                result = answer_question(library, case["question"], history=history)
                row.update(score_case(case, result))
                row.update(
                    status=result.status,
                    resolved_question=result.resolved_question,
                    api_calls=result.api_calls,
                    validation_issue=result.validation_issue,
                )
                row["retrieved"] = [
                    {
                        "source": p.document.metadata["url"],
                        "distance": p.distance,
                        "text": p.document.page_content,
                    }
                    for p in result.passages
                ]
                sources = {p.source_id: p.document.metadata["url"] for p in result.passages}
                row["evidence"] = (
                    [
                        {"source": sources[e.source_id], "quote": e.quote}
                        for s in result.draft.statements
                        for e in s.evidence
                    ]
                    if result.draft
                    else []
                )
            except Exception as exc:  # noqa: BLE001 - redact provider details in saved reports
                row["error"] = (
                    str(exc)
                    if isinstance(exc, AnswerError)
                    else f"{type(exc).__name__}: evaluation failed; no answer scored."
                )
                row["expected_behavior"] = False
            row["seconds"] = round(time.perf_counter() - started, 3)
            report["cases"].append(row)
            completed[case["id"]] = row
            print(f"{case['id']}: {row.get('status', 'error')}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    write_report(report, args.output.with_suffix(".md"))
    print(f"Saved {args.output} and {args.output.with_suffix('.md')}")
    if any("error" in row for row in report["cases"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
