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
## Your review

1. Edit queries, categories, `expected_terms`, and `notes` in `golden_queries.json` to match the topics you care about most for the subreddit.
2. Add, remove, or merge queries as needed — keep `id` values unique and stable if eval data already references them (currently `q01`–`q20`).
3. Generate a candidate report (keyword + title hints on real chunks):

   ```bash
   python3 src/suggest_eval_chunks.py
   ```

   Outputs `reports/eval_candidate_report.md` (and `.manifest.json`). Top candidates are auto-selected as positives for the eval corpus (Option B — LLM-as-judge, no manual labeling).

4. Build the eval corpus (auto-selected positives + random distractors at 75/25 neg/pos ratio):

   ```bash
   python3 src/build_eval_corpus.py
   ```

   Outputs `data/eval_corpus.jsonl` (~2,000 chunks) + `data/eval_corpus_positives.json`.
