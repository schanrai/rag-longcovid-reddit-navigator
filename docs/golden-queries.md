# Golden queries (Phase 1a)

**File:** [`data/golden_queries.json`](../data/golden_queries.json)

## Schema (per query)

| Field | Purpose |
|--------|---------|
| `id` | Stable ID (`q01` …) — used by eval scripts and judgments |
| `query` | Natural language question as a user would ask it |
| `category` | `symptom` \| `treatment` \| `timeline` \| `prevalence` \| `emotional` \| `benefits` \| `meta` |
| `expected_terms` | Hints for search / labeling — terms often present in relevant chunks |
| `notes` | What “good retrieval” looks like for this query |
| `labeled_relevant` | **Leave empty until you finish review.** Later: list of `{ "chunk_id", "relevance" }` where `relevance` is 1–3 (Phase 1b) |

## Your review

1. Edit queries, categories, `expected_terms`, and `notes` in `golden_queries.json` to match the topics you care about most for the subreddit.
2. Add, remove, or merge queries as needed — keep `id` values unique and stable if eval data already references them (currently `q01`–`q20`).
3. Generate a candidate report (keyword + title hints on real chunks):

   ```bash
   python3 src/suggest_eval_chunks.py
   ```

   Outputs `reports/eval_candidate_report.md` (and `.manifest.json`). In each table, fill the **relevance (1–3)** column (leave blank to skip).

4. Merge labels into JSON:

   ```bash
   python3 src/ingest_eval_labels_from_report.py --report reports/eval_candidate_report.md
   ```

5. When labels are in place, continue with **eval corpus assembly** (`build_eval_corpus.py` — distractors + `eval_corpus.jsonl`).

Do not fill `labeled_relevant` until the query text is final enough to judge relevance against.
