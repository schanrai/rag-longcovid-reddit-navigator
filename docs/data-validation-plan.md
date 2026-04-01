# Data Validation Plan — Long COVID Reddit RAG

**Date:** 2026-03-27
**Session:** 2 hours (16:27–18:27 EDT)
**Input:** Arctic Shift download (r/LongCovid posts + comments, JSONL format)
**Location:** `/projects/rag-longcovid-reddit-navigator/`

---

## Objective

Before any build work begins, validate that the downloaded data is complete, well-structured, and understand the filtering impact of Gate 1 & Gate 2. This work informs chunking strategy, index sizing, and threshold calibration.

---

## Task 1: Schema Validation (~30 min)

**Goal:** Confirm the data has the fields we need, in the types we expect, with acceptable completeness.

**What we'll build:** A Python script (`validate_schema.py`) that:
- Stream-reads both JSONL files line by line (no full load into memory — files are large)
- Checks each record for required fields:
  - **Posts:** `id`, `title`, `selftext`, `score`, `created_utc`, `num_comments`, `author`, `permalink`
  - **Comments:** `id`, `body`, `score`, `created_utc`, `parent_id`, `link_id`, `author`
- Reports:
  - Total record count (posts and comments separately)
  - Field presence rate per field (% of records with that field non-null/non-empty)
  - **All unique field names** found across records (discover fields we didn't expect)
  - Sample of any malformed/unparseable lines
  - 3-5 sample records printed for manual eyeball check

**Thread structure investigation (critical — blocks chunking design):**
- Does `parent_id` exist on comments? What is its format?
- Does `link_id` exist on comments? What is its format?
- Do these fields use Reddit type prefixes (`t1_` for comment, `t3_` for post) or plain IDs?
- What prefix (if any) distinguishes "reply to original post" vs "reply to another comment"?
- Sample 10 comments: 5 that are direct replies to posts, 5 that are replies to other comments (identify by checking if `parent_id` matches a post `id` or a comment `id`). Print these for manual inspection.
- Report: % of comments that are direct post replies vs nested replies (informs whether thread depth is significant enough to handle in v1)

**Output:** Console summary + `schema_report.json` saved to project folder.

---

## Task 2: Date Coverage Analysis (~15 min)

**Goal:** Identify any gaps in temporal coverage (missing months, suspicious drops).

**What we'll build:** Extend `validate_schema.py` (or separate section) to:
- Extract `created_utc` from every record
- Bucket into year-month counts (posts and comments separately)
- Flag any months with zero records
- Flag any months where count drops >50% from the prior month (could indicate a gap or could be natural — we just want visibility)

**Output:** Monthly histogram printed to console + `coverage_report.json`.

---

## Task 3: Gate 1 & Gate 2 Filter Analysis (~30 min)

**Goal:** Understand how many comments survive each gate, and whether the 25% threshold for Gate 1 is reasonable — before we build the actual filtering pipeline.

**Important:** This is analysis on raw JSONL, not chunked data. We're measuring filter impact, not building the production filter.

**What we'll build:** A script (`gate_analysis.py`) that:

### Gate 2 (substance) — run first, it's simpler
- Count words in each comment `body`
- Report: distribution of comment word counts (min, max, median, p25, p75)
- Report: % of comments that pass ≥ 50 words
- Report: sample of comments right at the boundary (45-55 words) for manual review

### Gate 1 (signal) — requires linking comments to parent posts
- Build a lookup: post `id` → post `score`
- For each comment, find parent post score via `link_id`
- Calculate comment `score` / parent post `score` ratio
- Report: distribution of ratios
- Report: % of comments that pass ≥ 25% threshold
- Report: % of comments that pass both Gate 1 AND Gate 2
- Report: what the survival rate looks like at different thresholds (10%, 25%, 50%) — helps calibrate

**Output:** Console summary + `gate_analysis_report.json`.

---

## Task 4: Review & Decide (~30 min)

**Goal:** Look at the outputs together and answer:

1. **Schema:** Are there missing fields that force a scope change? Any surprises?
2. **Coverage:** Any months missing that we need to backfill via PRAW?
3. **Gates:** Is 25% the right Gate 1 threshold? Is 50 words the right Gate 2 threshold?
4. **Index sizing:** How many comments survive both gates? This is our approximate index size pre-chunking.

**Decisions to make:**
- Lock or adjust gate thresholds
- Note any schema quirks that affect chunking strategy
- Determine if PRAW backfill is needed for any date gaps

---

## Buffer (~15 min)

For debugging, unexpected data issues, or discussion.

---

## Dependencies

- Python 3.x (already available)
- No external packages needed — raw JSON parsing with stdlib (`json`, `collections`, `statistics`)
- `pip-audit` run if we add any dependency

---

## Not in scope today

- Chunking implementation
- Weaviate setup
- PRAW investigation
- Production filtering pipeline
- Ingestion script
