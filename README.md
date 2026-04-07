# Long COVID Reddit RAG Navigator

A retrieval-augmented generation (RAG) pipeline over the r/LongCovid subreddit,
built as part of the AI PM Curriculum — Module 5 (RAG in Production).

## Purpose

Help Long COVID patients and researchers retrieve relevant patient experience,
symptom discussions, and treatment signals from ~312k community comments.

## Architecture (in progress)

```
Arctic Shift dump (NDJSON)
        │
        ▼
  validate_schema.py   — field coverage, date ranges, thread structure
        │
        ▼
  gate_analysis.py     — filter calibration; locked gates:
                           Gate 2: word count ≥ 25 (substance)
                           Score floor: comment_score ≥ 0
                           comment.score stored as ranking metadata
        │
        ▼
  chunk_data.py        — chunk + attach metadata (agreement_count, thanks_count, score)
        │
        ▼
  enrich_summaries.py  — LLM post summaries → post_summary on comment chunks (OpenRouter)
        │
        ▼
  embed + index        — vector store (TBD)
        │
        ▼
  query / retrieval    — RAG query interface (TBD)
```

## Indexable corpus

| Metric | Count | % |
|--------|------:|---|
| Total comments | 312,093 | 100% |
| Pass Gate 2 (≥ 25 words) | 161,552 | 51.8% |
| **Indexable (Gate 2 + score ≥ 0)** | **160,231** | **51.3%** |
| Excluded (score < 0) | 2,903 | 0.9% |

## Social signal metadata (attached at ingestion)

| Signal | Count | % of Gate 2 failures | Direction |
|--------|------:|:---:|---|
| `agreement_count` | 6,120 | 4.1% | 61% to comments, 39% to posts |
| `thanks_count` | 11,515 | 7.6% | 84% to comments, 16% to posts |

## Project layout

```
rag-longcovid-reddit-navigator/
├── src/
│   ├── validate_schema.py   — schema + date coverage validation
│   └── gate_analysis.py     — gate calibration analysis
├── docs/
│   ├── data-validation-plan.md
│   └── gate_analysis_findings.md
├── reports/                 — generated JSON outputs (gitignored, re-runnable)
├── data/                    — raw NDJSON corpus (gitignored, large files)
├── requirements.txt
└── README.md
```

## Running the scripts

Install dependencies (recommended: project virtualenv):

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Schema validation (generates reports/schema_report.json, reports/coverage_report.json)
python3 src/validate_schema.py

# Gate analysis (generates reports/gate_analysis_report_v3.json)
python3 src/gate_analysis.py

# Chunking (data/comment_chunks.jsonl, data/post_chunks.jsonl)
python3 src/chunk_data.py

# Post summaries for comment context — needs OPENROUTER_API_KEY in .env (see .env.example)
python3 src/enrich_summaries.py

# Phase 1b — generate candidate chunk manifest for eval corpus (needs data/golden_queries.json + chunk JSONL)
# Output: reports/eval_candidate_report.md  (reference only; used by build_eval_corpus.py)
python3 src/suggest_eval_chunks.py

# Phase 1b (cont.) — build eval corpus: auto-select positives + random distractors
# Output: data/eval_corpus.jsonl
python3 src/build_eval_corpus.py
```

## Data source

Arctic Shift dump of r/LongCovid — posts and comments exported as NDJSON.
Place files in `data/` before running:
- `data/r_LongCovid_posts.jsonl`
- `data/r_LongCovid_comments.jsonl`

## Design decisions

Key decisions documented in `docs/gate_analysis_findings.md`:
- Why Gate 2 threshold is 25 words (not 50)
- Why Gate 1 (post score ratio) was dropped
- Why `comment.score` is a ranking signal, not a binary gate
- Why `agreement_count` and `thanks_count` attach to both posts AND comments
