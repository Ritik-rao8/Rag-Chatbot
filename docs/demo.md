# Local portfolio demo

Use the synthetic store-policy corpus to demonstrate the app without exposing personal
uploads. The eight HTML files in `evaluation/corpus/` are authored fixtures; their
`example.test` URLs are fictional and must not be fetched or presented as real sources.
The fixture seeder feeds these files directly through extraction and indexing.

From the project root, after installing dependencies and caching the embedding model:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.run --seed-demo data/portfolio_demo
RAG_DATA_DIR=data/portfolio_demo HF_HUB_OFFLINE=1 .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

The seeder links the demo model cache to the existing local cache to avoid another
download. Keep that original cache available while using the demo.

The seeder refuses a directory with an existing source manifest. Seed only once; on
subsequent runs, start the server directly. This is a separate library from the normal
app on port 8501. Do not open two server processes on the same data directory. If the
model has not yet been cached, run without `HF_HUB_OFFLINE=1` for the initial download.

A three-minute walkthrough:

1. Show the eight sources and explain local embedding and persistent storage.
2. Ask: “How long do I have to return an unused item?” Show the thirty-day answer,
   its citation, and the exact supporting quote.
3. Follow up: “Do I need a receipt for that?” Expand Retrieval details to show the
   standalone question. Explain that history resolves the reference while documents
   supply the answer.
4. Click New chat. Ask: “If I return an unused item, what is the return deadline and
   will my original delivery charge be refunded?” Show evidence from both policies.
5. Click New chat. Ask: “What is the founder's date of birth?” Show abstention.
6. Show Search passages and Inspect documents to make extraction/retrieval visible.
   Explain that webpage snapshots have URLs, whereas uploaded PDFs have page citations.

For a PDF demonstration, upload a non-sensitive text PDF in this demo library, inspect
its page numbers, and ask a question supported by one page. The ten-case evaluation
covers HTML fixtures; PDF behavior is covered separately by the automated test suite.

## Repeatable evaluation

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.run
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -m 'not live' -q
```

The evaluation requires `GROQ_API_KEY` in `.env` or the environment. It builds a fresh
temporary index with real embeddings, runs ten questions through the real answer
pipeline, and saves `evaluation/results.json` and `evaluation/results.md`. It never
reads uploaded documents into the evaluation index. API requests use only synthetic
passages. Results include fixture hash, model, settings, retrieved passages, evidence,
and latency. A follow-up uses the actual answer to its designated parent question.

Read each answer against the reference and exact quotes. Expected behavior counts
answered versus abstained; it is not semantic correctness. Source recall checks known
sources among four retrieved passages. Expected citations check whether required sources
were cited. Reference-term matching is a coarse coverage signal. Exceptions fail the
command; quality mismatches remain visible in the report and need review. Ten simple
questions cannot establish production accuracy or robustness across languages,
long PDFs, tables, conflicting policies, or adversarial content.

## Quality and speed comparison

Run the baseline questions plus six harder condition/exclusion cases:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m evaluation.run --include-challenges --output evaluation/quality-results.json --interval 3
```

The interval spaces requests for account quotas and is excluded from each question's
latency. It does not speed up individual questions. See the quality report for measured
outcomes and limitations. Ordinary chat needs no interval setting.

## Verified UI capture

The local synthetic demo returned a thirty-day answer and exposed the matching source
quote. Captured in the app's narrow browser panel on 21 October 2026:

![Chat interface and source evidence](screenshots/improved-chat.jpg)
