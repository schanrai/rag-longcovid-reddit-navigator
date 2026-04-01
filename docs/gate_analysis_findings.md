# Gate Analysis Findings — Long COVID Reddit RAG

**Date:** 2026-03-30
**Last updated:** 2026-03-31 (v4 — v2 report figures confirmed, thanks_count signal added, agreement_count architectural correction)
**Source data:** `gate_analysis_report_v2.json` (generated 2026-03-31T16:33:55 UTC, locked thresholds)
**Script:** `gate_analysis.py`
**Status:** ✅ All gate decisions locked — two new metadata signals added to scope (thanks_count, agreement_count on comments)

---

## 1. Corpus Overview

**Confirmed figures from `gate_analysis_report_v2.json` (generated 2026-03-31T16:33:55 UTC, locked thresholds):**

| Metric | Count | % |
|---|---|---|
| Total comments | 312,093 | 100% |
| Pass Gate 2 (≥ 25 words) | 161,552 | 51.8% |
| Excluded score < 0 | 2,903 | 0.9% |
| **Indexable corpus (Gate 2 + score ≥ 0)** | **160,231** | **51.3%** |
| Gate 2 failures (< 25 words) | 150,541 | 48.2% |
| Agreement detected in failures | 6,120 | 4.1% of failures |
| Thanks/acknowledgement detected in failures | 12,593 | 8.6% of failures |

**Score distribution (for ranking signal at retrieval):**

| Score | Count | % | Notes |
|---|---|---|---|
| score < 0 | 2,903 | 0.9% | Excluded at ingestion |
| score = 0 | 4,023 | 1.3% | Indexed |
| score = 1 | 155,460 | 49.8% | Zero community endorsement — indexed, ranks lower |
| score ≥ 2 | 149,707 | 48.0% | At least 1 real endorsement — indexed, ranks higher |

**Key discovery:** 49.8% of all comments have a score of exactly 1 — the default Reddit self-upvote. Score = 1 represents zero community endorsement beyond the author themselves. This reframed our entire approach to Gate 1 (see Section 6).

---

## 2. Gate 2 — Substance Filter (Word Count)

### Word count distribution (n = 312,093)

| Statistic | Value |
| --- | --- |
| Min | 0 words |
| p25 | 10 words |
| **Median** | **26 words** |
| p75 | 57 words |
| p90 | 110 words |
| p95 | 162 words |
| Max | 1,655 words |

**Key insight:** The median comment is 26 words — well below any reasonable substance threshold. The corpus is heavily skewed short.

### Gate 2 survival at candidate thresholds

| Threshold | Pass (count) | Pass % | Fail (count) | Notes |
| --- | --- | --- | --- | --- |
| **25 words** | **161,552** | **51.8%** | **150,541** | **◄ LOCKED** |
| 50 words | 91,311 | 29.3% | 220,782 | Original design target — rejected (see below) |
| 75 words | 56,087 | 18.0% | 256,006 |  |
| 100 words | 36,860 | 11.8% | 275,233 |  |

### Why we moved from 50 words to 25 words

Initial design target was 50 words. Boundary sample review at 50 words confirmed that comments at that threshold have genuine clinical substance. However, analysis of the 26-49 word bucket revealed **66,262 substantive comments being discarded:**

| Words | Body (excerpt) |
| --- | --- |
| 27 | *"What early symptoms did you guys have indicated you would have long covid?"* |
| 44 | *"Have had a headache for 20 days.... tested negative but thinking maybe it's an after covid symptom?"* |
| 41 | *"No, I've been hospitalised twice because of heart stuff and given the all clear every time..."* |
| 43 | *"Yeah sometimes I wonder if I'd have recovered by now if I didn't spend 87% of my day doom-scrolling..."* |

These are genuine questions, patient experience fragments, and treatment discussions — content with real retrieval value. Discarding 66k of these at the 50-word threshold was too aggressive. The threshold was lowered to 25 words to recover this bucket.

**Trade-off acknowledged:** Lowering to 25 words introduces more noise in the 6-24 word range. The agreement heuristic (Section 3) catches the lowest-signal content in this range. The remaining noise is managed by score-based ranking at retrieval time rather than permanent exclusion at ingestion.

### Gate 2 decision

| **Decision** | ✅ **Locked at 25 words** |
| **Rationale** | Boundary samples at 50 words contained genuine clinical substance. 26-49 word bucket analysis confirmed 66k substantive comments would be permanently discarded at 50-word threshold |
| **Pass count** | 161,552 (51.8% of corpus) |

---

## 3. Agreement Aggregation (Gate 2 Failures)

Comments that fail Gate 2 (< 25 words) are not simply discarded. They have two exit paths (see scope doc Section 4.3 for the architecture diagram):

1. **Agreement-type:** Express confirmation/prevalence signal → counted as `agreement_count` metadata on the parent post. The signal is *preserved* but *transformed* — not indexed as a standalone chunk.
2. **Other short content:** Noise, questions, referrals, single words → discarded entirely.

### Headline numbers (after heuristic v2 — widened 2026-03-30)

| Category | Count | % of Gate 2 failures |
| --- | --- | --- |
| Total failing Gate 2 (< 25 words) | ~150,541 | 100% |
| Agreement-type (keyword heuristic v2) | **7,391** | ~4.9% |
| Other short content | ~143,150 | ~95.1% |

*Note: Agreement counts were measured against the 50-word threshold corpus. Exact counts against the 25-word threshold will be updated when gate\_analysis.py is re-run with the locked threshold.*

### Word-count bucket breakdown (Gate 2 failures by length)

| Bucket | Total | Agreement | Agree % | Non-agreement | Non-agree % |
| --- | --- | --- | --- | --- | --- |
| 0-5 words | 46,875 | 4,570 | **9.7%** | 42,305 | 90.3% |
| 6-15 words | 62,340 | 879 | 1.4% | 61,461 | 98.6% |
| 16-24 words | 45,305 | 740 | 1.6% | 44,565 | 98.4% |

**Key insight:** Agreement signal is heavily concentrated in the 0-5 word bucket (9.7%). Beyond 5 words, detection drops to ~1.5%.

### Agreement detection status

| **Heuristic** | v2 (widened keywords + emoji patterns + short affirmation openers) |
| **Detection rate** | 3.3% of Gate 2 failures (up from 2.4% in v1) |
| **Decision** | ⏳ Leaning locked for v1 |
| **To be decided** | Per-post aggregation not yet run — cannot confirm if `agreement_count` metadata has retrieval value. 

---

## 4. Gate 1 — Original Formula and Why It Failed

### Original formula: `comment_score / post_score ≥ 25%`

**Score ratio distribution (n = 305,323 Gate 1 eligible comments):**

| Statistic | Value | Interpretation |
| --- | --- | --- |
| Min | -23.0 | Most downvoted comment relative to its post |
| p25 | 0.053 | 25% of comments score at ≤ 5.3% of their post's score |
| **Median** | **0.125** | Typical comment scores at ~12.5% of its post's score |
| p75 | 0.270 | 75% of comments score at ≤ 27% of their post's score |
| p95 | 1.0 | Top 5% score at least as high as their parent post |
| Max | 46.0 | Comment outscored its post by 46× |

The 25% threshold sits above the median — meaning over half of all eligible comments fail at that threshold. This prompted investigation of three edge cases:

### Edge case A: Viral post ("high-upvote, low-effort topic")
Post titled "Anyone else exhausted?" — 200 upvotes, 4-word body. Gate 1 at 25% requires `comment_score ≥ 50`. A thoughtful 150-word comment with 15 upvotes fails (15/200 = 7.5%). Post score measures community *topic resonance*, not a quality bar for comments.

### Edge case B: Active discussion, zero net votes
Post with score = 0 and 85 comments. `comment_score / 0` → division by zero. The 6,625 comments on zero-score posts bypassed Gate 1 entirely — an uncontrolled loophole.

### Edge case C: Small post, false positive
Post with score = 2, comment with score = 1. Ratio = 50% — passes at any threshold. But a single upvote on a 29k-member subreddit is not meaningful community validation.

### The core diagnosis

The ratio conflates two different things:
- **Absolute comment quality** — does this comment have community validation?
- **Relative post popularity** — how hot was the thread?

A ratio-based gate systematically punishes good comments under popular posts and rewards low-quality comments under obscure posts.

---

## 5. The Search for a Better Gate 1 Formula

### Why `num_comments` was explored as a normalizer

`comment_score / post_score` uses post score as the denominator. But post score measures topic resonance — unrelated to the quality bar a comment should meet. `comment_score / post_num_comments` instead asks: "Out of the people who showed up to discuss this thread, how many validated this comment?" More participation-weighted and intuitive.

However, `num_comments` as a pure ratio still fails Edge Case C (small denominators inflate ratios on tiny threads). This led to **Option E: floor + engagement scaling.**

### Option E: `comment_score ≥ max(absolute_min, ceil(post_num_comments × rate))`

Initially appeared to resolve all three edge cases:

| Option | Formula | Edge A | Edge B | Edge C |
| --- | --- | --- | --- | --- |
| Original | `score/post_score ≥ 25%` | FAIL | UNDEFINED | False positive |
| **E** | `score ≥ max(2, ceil(num_comments × rate))` | PASS ✓ | PASS ✓ | FAIL ✓ |

Option E was provisionally selected for v1.

### Why Option E was subsequently rejected

**The \****`rate`**\*\* parameter is unprincipled.** `rate` is meant to express "what percentage of thread participants should have upvoted this comment." But this assumes a stable conversion ratio between the commenting population and the voting population — which does not exist.

On every social platform, upvotes have significantly lower friction than comments. The population of people who upvoted a comment is primarily *lurkers who never commented at all*. `num_comments` systematically understates the actual voting audience, making the formula's `rate` impossible to calibrate in any principled way. Any value chosen is arbitrary.

This connects to the deeper point raised in the mentor review: **the value of a signal depends on context, and isolating it into a simple formula loses that context.** In econometric terms, `comment.score` is a variable with important interdependencies — its meaning shifts with thread size, community age, topic emotiveness, and time period. A single-variable formula treats it as context-independent, which it isn't. We don't have the data to control for those interdependencies, so any formula we build will be a distortion of the underlying signal.

### What other data was investigated

Following the mentor review, we investigated whether higher-quality engagement proxies were available:

| Field | Present | Non-zero/truthy | Verdict |
| --- | --- | --- | --- |
| `view_count` | 0% | 0% | Never exposed by Reddit's public API. Permanently null. |
| `clicked` | 70.8% (post-2023) | 0% — all `False` | Per-session flag for the scraping bot account. Not community data. |
| `visited` | 70.8% (post-2023) | 0% — all `False` | Same — per-session flag. Not community data. |
| `likes` | 0% | 0% | Deprecated by Reddit. |
| `ups` / `downs` | 100% / 70.8% | `ups` = same as `score`; `downs` all zero | Reddit deprecated raw upvote/downvote counts — `ups` now mirrors `score`. |
| `upvote_ratio` | 100% | Real values (0.55–1.0) | **Post-level only** — not available on comments. |
| `subreddit_subscribers` | 100% | Real values (463 → 29,000) | Cumulative monotonic counter — includes churned, inactive, and deceased members. Inflates active community size, especially for a health condition subreddit with natural attrition. Rejected as scaling variable. |

**Conclusion:** Reddit does not expose the data needed to build a principled engagement-scaled formula. `comment.score` is the only direct community signal available at the comment level.

---

## 6. The Self-Upvote Discovery and the Decision Boundary
A ratio-based gate systematically punishes good comments under popular posts and rewards mediocre comments under 
obscure posts.
### Score distribution at the low end (n = 312,093)

Running the corpus against absolute score floors revealed a critical pattern:

| Floor | Pass count | Pass % |
| --- | --- | --- |
| score ≥ 1 | 305,167 | 97.8% |
| score ≥ 2 | 149,707 | 48.0% |
| score ≥ 3 | 72,188 | 23.1% |
| score ≥ 4 | 41,152 | 13.2% |
| score ≥ 5 | 31,249 | 10.0% |

**The key finding:** 49.8% of all comments (155,460) have score = 1 exactly. On Reddit, every post and comment receives an automatic +1 upvote from the author at creation. Score = 1 therefore means **zero community endorsement** — nobody else voted. The jump from score ≥ 1 (97.8%) to score ≥ 2 (48.0%) is the boundary between "author posted it" and "at least one other person agreed."

We initially considered score ≥ 3 (at least 2 real endorsements, 23.1% pass rate). But applying that alongside Gate 2 at 25 words would have produced an index of roughly 12% of the corpus — aggressive given the structural uncertainty in any score-based threshold.

---

## 7. Three Paths Considered — and Why Path B Was Selected

With Gate 2 locked at 25 words (161,552 comments passing) and score ≥ 2 as the candidate floor:

| Path | Gate 1 approach | Index size | Index % |
| --- | --- | --- | --- |
| **A** | score ≥ 2 as hard binary gate | ~77,000 (est.) | ~25% |
| **B** | No binary gate — score stored as metadata, used as ranking signal at retrieval | 161,552 | 51.8% |
| **C** | score ≥ 2 as noise floor + score as ranking signal | ~77,000 (est.) | ~25% |

Paths A and C produce identical indexes. The real choice was between B and C.

### The decision boundary: what does Path C permanently exclude?

**69,936 comments (22.4% of corpus)** pass Gate 2 (≥ 25 words) but have score = 1. These are what C excludes and B includes. We sampled 20 randomly:

| Words | Score | Body (excerpt) |
| --- | --- | --- |
| 27 | 1 | *"Wow that's great, how is your breathing now? My only symptom is respiratory so I want to learn more about the breathing exercises that helped you."* |
| 36 | 1 | *"Look up protocols for these conditions/illnesses, which greatly overlap with long-Covid: ME/CFS, MCAS, POTS, hEDS, CCI, CSF Leak..."* |
| 64 | 1 | *"10mg H1 per day (split into two) is fine, pepsid 40mg at night is fine — if you don't take enough H1 it's hard to say if it'll work..."* |
| 102 | 1 | *"On right side of head: 24/7 constant headache, 24/7 ear popping/crackling... sinus/nose/cheek/eye pain, congestion, upper gums pain..."* |
| 116 | 1 | *"So much grief. The things that brought me joy in life, my job, singing, playing music, making costumes and props, have all been taken from me one by one..."* |
| 168 | 1 | *"It sounds like you are having adrenaline bursts, especially related to the sleep. This is often related to apnea, a common post-COVID condition..."* |

The sample contains POTS treatment protocols, detailed symptom descriptions, specific medication dosing, and patient experience narratives. Only 1-2 of the 20 samples were genuine noise. The 22.4% population that Path C would permanently exclude contains too much genuine clinical and experiential content to discard at the gate.

### Why Path B — the philosophy

The distinction is not just practical but conceptual:

A **binary gate** makes a permanent exclusion decision at ingestion time. It asserts the threshold is principled enough that the excluded content should never be retrieved — not now, not for any query. Given that we've demonstrated no principled basis for any score threshold (it's context-dependent, the voting audience is unknown, the metric is structurally flawed), making permanent exclusions on that signal is the wrong tradeoff.

A **ranking signal** makes a soft, reversible, query-relative judgement. Score = 1 doesn't permanently exclude a comment — it ranks lower than score = 15. But if a score-1 comment is the best semantic match for a specific query, it surfaces. Quality differentiation happens at retrieval, where it's query-aware, not at ingestion, where it's context-blind.

**score ≥ 0 as a hard floor** is the one score-based exclusion we can make with genuine confidence: negatively-scored comments (score < 0) have been *actively rejected* by the community — more people pushed back than endorsed them. This is categorically different from score = 0 (contested, equal votes) or score = 1 (no signal either way). Everything at score ≥ 0 enters the index and competes on semantic relevance. Quality differentiation between score = 1 and score = 15 happens at retrieval via ranking, not at ingestion.

**The resulting ingestion rule:**

| Score | Count | Decision | Rationale |
| --- | --- | --- | --- |
| score < 0 | 2,903 (0.9%) | ❌ Excluded | Community actively rejected — unambiguous negative signal |
| score = 0 | 4,023 (1.3%) | ✅ Indexed | Contested (equal up/down votes) — not rejected |
| score = 1 | 155,460 (49.8%) | ✅ Indexed | No community signal, but 22.4% of corpus contains genuine clinical substance |
| score ≥ 2 | 149,707 (48.0%) | ✅ Indexed, ranks higher | At least one real endorsement — boosted at retrieval |

### Gate 1 decision

| **Decision** | ✅ **Gate 1 as a binary ingestion gate — DROPPED** |
| **Replacement** | `comment.score` stored as metadata on every indexed chunk, used as ranking boost at retrieval time |
| **Only exclusion** | Comments with score < 0 (2,903 comments, 0.9%) — community actively downvoted these |
| **Rationale** | 22.4% of corpus (69,936 comments) passes Gate 2 with score = 1 and contains genuine clinical substance. Permanently excluding it based on a metric with no principled threshold is the wrong tradeoff. Score < 0 is the sole exception: it's the only score value where community intent is unambiguous. |

---

## 8. Social Signal Metadata — New Scope (2026-03-31)

Two metadata signals are now in scope for v1, following mentor review and data analysis. Both are derived from Gate 2 failures and attached as computed fields at ingestion time — not from the Arctic Shift dump directly.

### 8.1 Architectural correction: where agreement_count lives

**Previous design:** `agreement_count` attached to parent *post* only.

**Correction:** Agreement signals can be replies to either a post (t3_) or a comment (t1_). A "same here" reply to a comment sharing a recovery protocol is validating that *comment*, not the post. Attaching it only to posts loses the specificity of the signal.

**Revised design:** `agreement_count` computed on both posts AND comments. Every indexed document (post or comment chunk) that receives agreement replies carries an `agreement_count` field.

### 8.2 `thanks_count` — new signal (v1)

**Signal type:** Utility signal. A reply expressing gratitude or acknowledgement directed at a specific parent document indicates the parent content was actionable or helpful — distinct from agreement (prevalence) signals.

**Why v1 (not deferred):** Volume analysis confirms the signal is significant enough to build for v1:

| Metric | Value |
|---|---|
| Gate 2 failures (< 25 words) | 150,541 |
| Thanks/acknowledgement detected | **12,593 (8.6%)** |
| → Reply to a specific comment (t1_) | 10,734 (85.2%) |
| → Reply to a post (t3_) | 1,859 (14.8%) |

12,593 is more than double the agreement signal (6,120). 85.2% are directed at specific comments — confirming this lives on comments, not just posts.

**Sample thanks comments:**

| Words | Parent type | Body |
|---|---|---|
| 16 | comment | *"Thank you for posting this. I didn't know the WHO had these streamed question panels."* |
| 7 | comment | *"Thank you! This is suuuuper helpful! <3"* |
| 23 | post | *"Thank you for your advice! If you have any other suggestions or updates on how you're feeling I would really appreciate knowing!"* |
| 16 | comment | *"Thank you! I am learning that documentation seems to be the key. Wishing you the best!"* |
| 29 | comment | *"Thanks, it'll be really helpful to have even approximate benchmarks! Have you noticed any change in your allergic response?"* |

**Heuristic approach:** Keyword-based (same architecture as agreement heuristic). Phrases: "thank you", "thanks for", "really appreciate", "this helped", "so helpful", "grateful", etc. Short exact matches: "ty", "tysm", "thank you", "thanks". Short opener check: comments starting with "thank"/"thanks"/"grateful"/"appreciate".

**Distinction from agreement:**
- Agreement → "same here", "me too" → prevalence signal → how common is this experience?
- Thanks → "this helped", "thank you" → utility signal → was this content actionable?

**Metadata field:** `thanks_count` on both posts AND comments (symmetric with `agreement_count`).

**Status:** ✅ In scope for v1. Build alongside `agreement_count` in `gate_analysis.py` update.

---

## 9. Decision Register (Final)

| # | Decision | Locked value | Status | Rationale |
| --- | --- | --- | --- | --- |
| D1 | **Gate 2 threshold** | 25 words | ✅ Locked | 50-word threshold discarded 66k substantive comments in 26-49 word range. Boundary samples confirmed retrieval value in that bucket. |
| D2 | **Gate 1 formula** | Dropped as binary gate | ✅ Locked | Original ratio formula structurally flawed. All alternatives either unprincipled (`rate`), or use unavailable data (`view_count`). Score used as ranking signal instead. |
| D3 | **Zero-score post policy** | N/A — resolved by dropping Gate 1 | ✅ Resolved | No binary gate = no division-by-zero edge case. |
| D4 | **Agreement heuristic** | v2 (widened) — 4.1% detection at 25-word threshold | ✅ Locked for v1 | Detection rate confirmed against locked threshold. Diminishing returns on further expansion. |
| D5 | **`agreement_count` metadata** | On posts AND comments | ✅ In scope v1 | Architectural correction: agreement signals reply to comments (85%+) not just posts. Must attach to both. |
| D6 | **Gate 2 at 25 words** | Adopted | ✅ Locked — see D1 | |
| D7 | **`thanks_count` metadata** | On posts AND comments | ✅ In scope v1 | 12,593 signals (8.6% of Gate 2 failures), 85% directed at comments. Volume justifies v1 build. Utility signal — distinct from agreement's prevalence signal. |

---

## 9. Challenges and Risks (Resolved)

| # | Challenge | Resolution |
| --- | --- | --- |
| C1 | Gate 1 formula structurally flawed | ✅ Dropped as binary gate — score used as ranking signal |
| C2 | Per-post/comment agreement distribution unknown | ✅ Resolved — architectural correction made. `agreement_count` now attaches to both posts AND comments. `thanks_count` added as parallel utility signal. Both in scope for v1. |
| C3 | Gate 2 at 50 words discards substantive mid-range comments | ✅ Resolved — threshold lowered to 25 words |
| C4 | 6,625 comments on zero-score posts uncontrolled | ✅ Resolved — no binary gate removes the edge case |
| C5 | Time pressure | ✅ All decisions now locked — ready to proceed to chunking |

---

## 10. Next Steps

### Do next (open new chat — implementation)
1. **Update `gate_analysis.py`** with two new metadata signals:
   - Add `thanks_count` heuristic (keyword-based, same architecture as agreement heuristic)
   - Fix `agreement_count` to attach to both posts AND comments — not posts only
   - Re-run and generate `gate_analysis_report_v3.json` confirming per-document signal counts
2. **Proceed to `chunk_data.py`** — gates are locked:
   - Gate 2: word count ≥ 25
   - Gate 1: dropped — `comment.score` stored as metadata, score < 0 excluded
   - Chunk metadata must include: `comment.score`, `agreement_count`, `thanks_count`
3. **Update `long-covid-rag-scope-v2.md`** — reflect Path B architecture, score-as-ranking-signal, and both new metadata fields in Section 4.3 and Section 5.4.

### Can defer to v2
4. Time-based analysis of score distributions (do thresholds behave differently across 2020 vs 2025?).
5. Academic research on Reddit community dynamics / social hierarchy signals for future Gate 1 design.

---

*Machine-readable results: `gate_analysis_report_v2.json` (locked thresholds), `gate_analysis_report.json` (v1 at 50-word threshold)*
*Analysis script: `gate_analysis.py`*
*Schema findings: `schema_report.json`, `coverage_report.json`*
