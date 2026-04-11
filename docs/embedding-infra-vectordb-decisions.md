---
name: Embedding, Infrastructure & Vector DB Decisions
overview: Embedding model locked — Voyage-4-Large (Phase 1d LLM-judge eval, 2026-04-10). Infrastructure path — Path A (Voyage API + Weaviate Cloud). Chunking + eval corpus + embeddings complete; next — full-corpus ingest (Phase 2).
todos:
  - id: embedding-eval
    content: "DONE (2026-04-10): Phase 1d eval — cosine top-10 + LLM judge (Gemini 2.5 Flash Lite via OpenRouter); reports/embedding_eval_report.json; Voyage-4-Large selected; scope Section 5.3 + this doc updated"
    status: completed
  - id: lock-post-chunks
    content: Lock post chunk sizing (informed by 32k-class context, not 512) and exclusion rules
    status: done
  - id: spec-post-schema
    content: Spec post chunk schema in scope doc Section 6
    status: done
  - id: update-chunk-data
    content: Update chunk_data.py to process posts alongside comments
    status: done
isProject: false
---

# Embedding, Infrastructure & Vector DB Decisions — Long COVID Reddit RAG

## What is locked now (2026)

**We are not married to a 512-token context window.** Earlier scope text assumed a small bi-encoder (e.g. `bge-base-en-v1.5`, 512 tokens). That constraint is **retired**. Chunk sizing for comments and posts is driven by **retrieval quality** (granularity, overlap, post length distribution), not by a hard encoder ceiling. Any finalist embedding model must support a **long context** (on the order of **8k+ tokens**; leading candidates are **32k**).

**Chunking and ingestion are independent of the final embedding vendor.** Atom storage (`text`, `post_title`, `post_summary`) means re-embedding with a different model is a **new embedding pass only** — no re-chunking.

**Locked (2026-04-10):** **Voyage-4-Large** (API) — see **Corpus embedding eval results** below. Self-hosted finalists (Qwen3-0.6B, Jasper-600M, Qwen3-4B) remain documented as benchmarks; production indexing uses Voyage.

---

## Leading candidate: Voyage-4-Large (API)


| Spec                   | Value                                                                                               |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| Context window         | 32,000 tokens                                                                                       |
| Dimensions             | 1024 default (adjustable: 256, 512, 2048)                                                           |
| Architecture           | MoE — lower serving cost than comparable dense models                                               |
| Cost                   | ~$0.12/M input tokens (order-of-magnitude ~$2–4 one-time for full corpus at current estimates)      |
| MTEB retrieval         | Strong among API-hosted options (e.g. ~66.8 retrieval NDCG class)                                   |
| Free tier              | Large free token allowance on Voyage accounts (verify current docs)                                 |
| Shared embedding space | v4 family (large / mid / lite / nano) — compatible vectors for asymmetric index vs query strategies |
| Availability           | Voyage API; also listed on cloud marketplaces (MongoDB Atlas, GCP, AWS, Azure)                      |


**Why it remains the default hypothesis:** Zero fixed infra for MVP, fast query-time embedding, no GPU to operate. Aligns with **pay-per-use until there is traffic**.

---

## ~~Alternative under consideration: gte-Qwen2-1.5B-instruct (self-host)~~ DROPPED (2026-04-02)

**Dropped from eval shortlist.** Cross-domain leaderboard analysis (see Section below) shows Qwen3-Embedding-0.6B outperforms gte-Qwen2-1.5B-instruct on both Medical Retrieval (90.52 vs 85.38) and Social/Blog Retrieval (70.97 vs 69.72) — with 1/3 the parameters (0.44B vs 1.3B) and 1/9 the dimensionality (1,024 vs 8,960). There is no scenario where gte-Qwen2 wins the eval.

---

## How we got here (decision trail)

1. **Chunk size vs encoder:** Post chunking discussion exposed that chunk limits were tied to an unchosen 512-token placeholder. Sequencing corrected: **long-context models** allowed; chunk parameters follow **data + retrieval**, not arbitrary token caps.
2. **General vs medical MTEB:** API-first options looked strong on broad retrieval; **Medical** filter on MTEB favored **self-hosted 7B** models; **Voyage-3** ranked lower on medical. Upgraded to **Voyage-4** after user caught **missing "latest version" check** (now in [CLAUDE.md](../CLAUDE.md) — verify current release before recommending vendors/models).
3. **Domain models (PubMedBERT, BioRedditBERT, Reddit MPNet):** Most are **512-token** and/or **not retrieval-tuned** at modern scale. Reinforced: **do not regress to 512-token ceiling** for domain purity.
4. **API vs self-host:** Clarified — **not forced** to API. Vector DB and embedder can live on same cloud; **API** was chosen for **ops simplicity and $0 fixed cost**, not because colocation is impossible.
5. **MVP economics:** Risk of **paying for idle GPU** vs risk of **weak retrieval** killing the MVP. Resolution: **defer final model** until **corpus-level eval**; keep **API leader** as default; **do not** commit to 7B self-host for v1 budget.
6. **User decision:** **Final model choice open** until **data-driven test** (e.g. embed a sample with two candidates, run golden queries, compare). **Locked principle:** **no commitment to 512-token encoders.**

---

## Risks and mitigations (unchanged in spirit)


| Risk                                   | Mitigation                                                                                                              |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Benchmark ≠ our register               | Evaluate on **actual** Reddit chunks + queries                                                                          |
| Weak embedding hurts perceived quality | Compare candidates before full-index lock-in; hybrid **BM25 + semantic** reduces reliance on embeddings for exact terms |
| Vendor lock-in / cost                  | Atom storage; re-embed without re-chunk; watch query volume vs API bill                                                 |


---

## Chunk sizing — locked (2026-04-01)

✅ Both comments and posts use **300 words / 30-word overlap** with the same sliding-window chunker.

- **Comments:** 300/30. As-built: 164,812 chunks from 159,863 indexed comments. 97.5% single-chunk. Max 7 chunks. → `data/comment_chunks.jsonl`
- **Posts:** 300/30. As-built: 19,443 chunks from 16,766 indexed posts. 89.3% single-chunk. Max 21 chunks. → `data/post_chunks.jsonl`
- **Embed string:** title + optional summary + body chunk — still a small fraction of 32k context.

---

## MTEB Cross-Domain Leaderboard Analysis (2026-04-02)

**Source data:** MTEB Hugging Face leaderboard, filtered by Medical domain and Social/Blog domain. CSVs exported by user on 2026-04-02. Stored at project root as `huggingface_medical_only_topembedding models.csv` and `huggingface_social_blog_only_topembedding models.csv`.

**Why two domains:** Our corpus is informal Reddit health text — it sits at the intersection of medical terminology and social/conversational register. A model must perform well on both to succeed on our data. Neither domain benchmark alone is sufficient.

### Cross-domain retrieval comparison (models appearing in both leaderboards)

| Model | Social/Blog Retrieval | Medical Retrieval | Active Params | Dims | Max Tokens |
|---|---|---|---|---|---|
| Jasper-Token-Compression-600M | **78.28** | **89.92** | 0.45B | 2048 | 32k |
| Qwen3-Embedding-4B | **75.64** | **92.92** | 3.6B | 2560 | 32k |
| cde-small-v2 | **74.96** | 80.55 | ~0.3B | 768 | 512 |
| embeddinggemma-300m | **71.54** | 80.35 | 0.1B | 768 | 2048 |
| Qwen3-Embedding-0.6B | **70.97** | **90.52** | 0.44B | 1024 | 32k |
| gte-Qwen2-1.5B-instruct | **69.72** | 85.38 | 1.3B | 8960 | 32k |
| Yuan-embedding-2.0-en | **69.17** | 98.33 | 0.44B | 1024 | 2048 |
| bge-base-en-v1.5 | **63.75** | 78.03 | 0.086B | 768 | 512 |

### Key findings

1. **Voyage family scores poorly on Social/Blog Retrieval.** Models on the leaderboard (voyage-3, 3.5, 3-large, 3-lite) range from 50.83 to 66.78 on Social/Blog Retrieval. Voyage-4-Large is not yet on the leaderboard — its Social/Blog performance is unknown and must be tested directly.

2. **Yuan-embedding-2.0-en's 98.33 Medical score is suspect.** It drops to 69.17 on Social/Blog — a 29-point gap, far larger than any other model. Likely overfitting to the Medical benchmark. Also limited to 2,048 max tokens, which constrains chunk composition. **Excluded from shortlist.**

3. **Most models score significantly lower on Social/Blog than Medical.** This confirms domain matters — Medical benchmarks alone are insufficient for model selection on informal Reddit text.

4. **7B models do not top retrieval rankings.** The best retrieval scores across both domains come from models in the 0.4B–4B range. 7B models appear in the Social/Blog top (LGAI-Embedding-Preview at 86.95, bge-en-icl at 83.08) but offer no clear retrieval advantage over Qwen3-4B while being significantly heavier on CPU.

5. **cde-small-v2 and embeddinggemma-300m** have decent cross-domain scores but are limited to 512 and 2,048 max tokens respectively — incompatible with our 32k-class architecture.

### Eval shortlist (locked 2026-04-02)

| # | Candidate | Medical Retrieval | Social/Blog Retrieval | Params | Dims | Max Tokens | Path |
|---|---|---|---|---|---|---|---|
| 1 | **Qwen3-Embedding-0.6B** | 90.52 | 70.97 | 0.44B | 1024 | 32k | Self-host (sentence-transformers) + Qdrant or Weaviate |
| 2 | **Jasper-Token-Compression-600M** | 89.92 | 78.28 | 0.45B | 2048 | 32k | Self-host (sentence-transformers) + Qdrant or Weaviate |
| 3 | **Qwen3-Embedding-4B** | 92.92 | 75.64 | 3.6B | 2560 | 32k | Self-host (sentence-transformers) + Qdrant or Weaviate |
| 4 | **Voyage-4-Large** | Unknown | Unknown | Unknown (API) | 1024 | 32k | API + Weaviate Cloud (native integration) |

**Why these four:**
- **Qwen3-0.6B:** Best ratio of retrieval quality to model size. Strong Medical (90.52), decent Social/Blog (70.97), tiny (0.44B), manageable dims (1024), full 32k context. Compatible with sentence-transformers.
- **Jasper-600M:** Strongest cross-domain performer — highest Social/Blog Retrieval (78.28) of any sub-1B model while maintaining strong Medical (89.92). 2048 dims is higher than Qwen3-0.6B but manageable.
- **Qwen3-4B:** Highest Medical Retrieval in the shortlist (92.92) and strong Social/Blog (75.64). Included to test whether the extra 3B parameters buy meaningful retrieval quality over the 0.6B variant. Heavier on CPU — latency is the key risk.
- **Voyage-4-Large:** Represents the API path. Zero infra, native Weaviate integration, native reranker. Not on leaderboards yet — must benchmark directly. Included because it's the simplest pipeline if retrieval quality is competitive. The Voyage family's weak Social/Blog trend (50–67 range for older models) is a concern.

**Dropped:**
- **gte-Qwen2-1.5B-instruct:** Outperformed by Qwen3-0.6B on both domains with 1/3 params and 1/9 dims.
- **Yuan-embedding-2.0-en:** Suspicious 98.33 Medical score (29-point drop to Social/Blog), only 2048 max tokens.
- **7B-class models:** No retrieval advantage over 0.4B–4B candidates; significantly heavier on CPU.

### Infrastructure paths (determined by eval winner)

| If winner is... | Vector DB | Embedding | Infra |
|---|---|---|---|
| Voyage-4-Large | Weaviate Cloud (managed) | Voyage API (native integration) | Fully managed — simplest pipeline |
| Any self-hosted model | Qdrant Cloud (managed) or Weaviate | sentence-transformers in container | Middle-tier cloud (Railway/Render/Fly.io) |

Qdrant is preferred for the self-hosted path due to husband's hands-on experience with the Qdrant + sentence-transformers stack. Weaviate remains viable for self-hosted models via Hugging Face Hub API integration, but pulls toward managed cloud for simplicity only if using Voyage.

---

## Vector DB and hosting decisions (2026-04-02)

### Weaviate vs Qdrant — what each offers

**Weaviate:**
- Native hybrid search (BM25 + vector + RRF fusion) out of the box — matches our retrieval architecture (scope doc Section 5.2)
- Built-in embedding via model provider integrations — configure a model at the collection level, Weaviate handles embedding at ingest and query time automatically
- **Voyage AI integration:** Enabled by default on Weaviate Cloud. Supports Voyage embeddings + Voyage reranker natively. Zero separate embedding step needed.
- **Hugging Face integration:** Two modes: (1) HF Hub API — Weaviate calls HF Inference API, works on Weaviate Cloud; (2) Locally-hosted transformer container — Docker container with the model, but NOT available on Weaviate Cloud (requires self-hosted Weaviate)
- **Implication:** Using a sentence-transformers model (Qwen3, Jasper) with Weaviate Cloud requires the HF Hub API path. Running the model directly in a container requires self-hosting Weaviate too — the worst-of-both-worlds path.

**Qdrant:**
- Vector-storage-only — does not embed for you. You bring your own vectors.
- No native BM25 — would need to handle keyword search separately if going pure Qdrant
- **Qdrant Cloud** exists as a managed service (comparable to Weaviate Cloud)
- FastEmbed integration for local models
- **Key advantage for us:** Husband (CTO) has hands-on production experience with Qdrant + sentence-transformers stack. Live debugging resource if we get stuck.

### sentence-transformers compatibility

- **Voyage-4-Large:** API-only. NOT available as a sentence-transformers model. Embedding happens via Voyage API (directly or through Weaviate's native integration).
- **Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Jasper-Token-Compression-600M:** All compatible with the sentence-transformers Python library. Can be loaded locally and used for embedding.

### Hosting terminology (corrected 2026-04-02)

We use three tiers of hosting, not two:

| Tier | Who operates it | Examples | Cost profile |
|---|---|---|---|
| **Fully managed** | Vendor runs everything — you call APIs | Weaviate Cloud, Qdrant Cloud, Voyage API | Pay-per-use, zero ops |
| **Middle-tier cloud (infrastructure in the cloud)** | You provision and manage containers, but on simple PaaS | Railway, Render, Fly.io, DigitalOcean App Platform | Low fixed cost, Docker-based, no Kubernetes |
| **Full cloud** | You manage VPCs, instances, scaling | AWS EC2, GCP GCE, Azure VMs | High flexibility, high ops overhead |

Previous versions of this doc treated hosting as binary (managed SaaS vs "self-hosted on expensive AWS/GCP"). The middle tier was missing from the analysis, which led to prematurely ruling out models that require a container but not a GPU.

### The real decision (A vs C)

The vector DB + hosting choice follows from the embedding eval winner:

- **Path A (Voyage wins):** Voyage API + Weaviate Cloud. Fully managed, simplest pipeline, no containers to run.
- **Path C (self-hosted model wins):** sentence-transformers model in a container on middle-tier cloud + Qdrant Cloud for vector storage. Husband's known stack.
- **Path B (self-hosted model + self-hosted Weaviate) — rejected.** If we're self-hosting the embedding model anyway, there's no reason to also self-host Weaviate when Qdrant Cloud exists as managed vector storage and we have Qdrant expertise available.

BM25 handling on the Qdrant path is an open question — Qdrant doesn't do keyword search natively. Options include a separate BM25 index (Elasticsearch/OpenSearch) or exploring Qdrant's sparse vector support. To be resolved if Path C wins the eval.

---

## Correction: 7B models are not ruled out (2026-04-02)

The earlier rationale for excluding 7B-class models was:

> *"Not pursuing for MVP: 7B-class medical leaders — they require GPU infra ($150–400/mo always-on class, or hourly GPU with cold-start pain). Incompatible with a ~$10/mo MVP budget."*

**This assumption was wrong.** 7B models run on CPU — confirmed by domain experience (CTO running 7B Llama on CPU in middle-tier cloud) and community evidence (users running 14B quantized models on laptop CPUs at 5-15 tokens/s). Embedding is a single forward pass, not autoregressive generation, so it's less demanding than the text generation benchmarks cited.

**Infrastructure correction:** "Self-hosted" does not mean local laptop or expensive AWS/GCP. Middle-tier cloud providers (Railway, Render, Fly.io, DigitalOcean) can run these models in containers at reasonable cost. Qdrant Cloud also exists as a managed vector DB option.

**The candidate field was reopened** — but the subsequent leaderboard analysis (see "MTEB Cross-Domain Leaderboard Analysis" above) showed that 7B models do not actually top retrieval rankings. The best retrieval scores come from the 0.4B–4B range. The correction still stands on principle (7B is viable on CPU), but the data did not support including 7B models in the eval shortlist.

---

## Eval dimensions (2026-04-02)

The embedding model eval was intended to measure these dimensions. Ordered by priority:

**⚠️ Note (updated 2026-04-10):** Only **dimension 1** (retrieval quality) was directly measured in Phase 1c–1d. Dimensions 2–8 were assessed through reasoning, observed behaviour, or known specs — not controlled benchmarks. The decision record below reflects this honestly.

| # | Dimension | Assessment method | Non-negotiable? |
|---|---|---|---|
| 1 | **Retrieval quality** | ✅ **Measured** — LLM-as-judge (NDCG@10, MRR@10) on actual eval corpus + 20 golden queries | Yes — deciding factor |
| 2 | **Query latency** | ⚪ **Not measured** — Voyage API latency was not benchmarked per query. Self-hosted models were not timed at inference. Assumed acceptable for MVP. | Yes — hard ceiling (< 15s) |
| 3 | **Cost** | ⚪ **Reasoned** — Voyage API pricing is public ($0.12/M tokens). Self-hosted hosting costs estimated but not validated on actual infra. | Yes — must fit MVP budget |
| 4 | **Pipeline complexity** | ⚪ **Reasoned** — Voyage + Weaviate native integration is architecturally simpler than a container + Qdrant. Not empirically measured. | No — tiebreaker |
| 5 | **Vector dimensionality** | ✅ **Known from spec** — Voyage 1024, Qwen3-0.6B 1024, Jasper 2048, Qwen3-4B 2560. Storage/latency implications reasoned, not measured. | No — has cost/latency implications |
| 6 | **Domain fit** | ✅ **Measured via #1** — retrieval eval on actual Reddit corpus settles this; MTEB ranks were the pre-eval signal only | Measured by #1 |
| 7 | **Ingestion throughput** | ⚪ **Observed informally** — Voyage took ~2 min for 2k eval corpus via API; self-hosted models took 30–90 min on CPU. Not benchmarked at 184k scale. | No — one-time job, but affects iteration speed |
| 8 | **Support ecosystem** | ⚪ **Reasoned** — Voyage: Weaviate native integration + reranker. Self-hosted: CTO's hands-on Qdrant/sentence-transformers experience. | No — tiebreaker |

**Decision basis:** Voyage-4-Large won decisively on dimension 1 (the deciding factor) and is favoured on dimensions 4, 5, 7, and 8 through reasoning. Dimensions 2 and 3 remain to be validated during Phase 2 (full corpus ingest) and Phase 3 (retrieval layer) — those phases will surface any latency or cost surprises before the system is user-facing.

---

## Corpus embedding eval results (locked 2026-04-10)

**Artifacts:** `data/eval_corpus.jsonl` (~2k chunks), `data/golden_queries.json` (20 queries), per-model vectors under `data/embeddings/<model_id>/`, report `reports/embedding_eval_report.json` (schema `embedding_eval_report_v1`).

**Method (Phase 1c–1d):**
1. Embed the eval corpus + golden query strings with each shortlist model (`src/embed_eval.py`).
2. For each model × query: **cosine similarity top-10** chunk IDs.
3. **LLM-as-judge** (OpenRouter: `google/gemini-2.5-flash-lite`): each retrieved chunk scored **1** (not relevant), **2** (partial), **3** (highly relevant) using the golden query's `notes` / `expected_terms` as judge rubric, with chunk text composed via `compose_embedding_input`.
4. Metrics: **NDCG@10** (graded list quality), **MRR@10** with first hit ≥ 2 as "relevant" (`src/eval_judge.py`).

### Headline metrics (mean across 20 queries)

| Model | Mean NDCG@10 | Mean MRR@10 |
| --- | ---: | ---: |
| **Voyage-4-Large** | **0.9903** | **1.0000** |
| Jasper-Token-Compression-600M | 0.9852 | 1.0000 |
| Qwen3-Embedding-4B | 0.9768 | 1.0000 |
| Qwen3-Embedding-0.6B | 0.9697 | 0.9667 |

### Judge score distribution (200 judgments per model = 20 queries × 10)

| Model | Score 1 (irrelevant) | Score 2 (partial) | Score 3 (highly relevant) |
| --- | ---: | ---: | ---: |
| **Voyage-4-Large** | **3 (1.5%)** | **22 (11.0%)** | **175 (87.5%)** |
| Jasper-Token-Compression-600M | 6 (3.0%) | 33 (16.5%) | 161 (80.5%) |
| Qwen3-Embedding-4B | 9 (4.5%) | 47 (23.5%) | 144 (72.0%) |
| Qwen3-Embedding-0.6B | 16 (8.0%) | 41 (20.5%) | 143 (71.5%) |

Voyage leads on NDCG and surfaces the fewest irrelevant chunks in top-10 — 5× fewer score-1 results than Qwen3-0.6B.

### NDCG@10 by category

| Model | symptom | treatment | timeline | prevalence | emotional | benefits | meta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Voyage-4-Large** | **0.9993** | **0.9945** | **0.9920** | **1.0000** | 0.9534 | **1.0000** | **0.9927** |
| Jasper-600M | 0.9919 | 0.9933 | 0.9688 | 0.9896 | 0.9654 | 1.0000 | 0.9718 |
| Qwen3-4B | 0.9915 | 0.9767 | 0.9814 | 0.9990 | 0.9372 | 1.0000 | 0.9314 |
| Qwen3-0.6B | 0.9642 | 0.9845 | 0.9703 | 0.9977 | 0.9394 | 1.0000 | 0.9319 |

**Observation:** All models are weakest on **emotional** and **meta** categories (vs symptom/treatment/benefits). Voyage is the only model at 1.0 on prevalence and ≥ 0.99 on meta. Jasper slightly edges Voyage on emotional (0.9654 vs 0.9534) and on one specific query (q19 — "LC awareness…").

### Notable weak query

**q19** — "LC awareness. Doctor and family don't believe me =(" — tests **abbreviated "LC"** without spelling out Long COVID. Voyage's single lowest query score (0.9188). Root cause is likely the embedding seeing only the literal query string — query rewriting (expand "LC → Long COVID") and hybrid BM25 are the production mitigations, not a reason to choose a different model.

### Decision record

| Field | Value |
| --- | --- |
| **Locked model** | `voyage-4-large` (Voyage API), 1024 dimensions |
| **Winner rationale** | Best mean NDCG, tied best MRR (with Jasper + Qwen3-4B), fewest irrelevant top-10 results, smallest dims in shortlist, fastest full-corpus embed path, native Weaviate integration |
| **Trade-offs accepted** | Per-query API cost; vendor dependency; mitigated by atom storage (re-embed without re-chunk if vendor changes) |
| **Infrastructure path** | **Path A** — Voyage API + Weaviate Cloud (see "The real decision" section) |

---

## Next steps

1. ~~**Product:** Lock post chunk parameters and exclusions; extend scope doc Section 6 for post chunks.~~ ✅ Done (2026-04-01)
2. ~~**Code:** Extend `chunk_data.py` for post chunking into `data/` output.~~ ✅ Done (2026-04-01)
3. ~~**Confirm eval candidates:** Pull MTEB Medical + Social/Blog leaderboards, cross-reference, lock shortlist.~~ ✅ Done (2026-04-02). Shortlist: Qwen3-Embedding-0.6B, Jasper-Token-Compression-600M, Qwen3-Embedding-4B, Voyage-4-Large. gte-Qwen2-1.5B-instruct dropped.
4. ~~**Build golden query set:**~~ ✅ Done — `data/golden_queries.json` (20 queries, 7 categories).
5. ~~**Run embedding eval:**~~ ✅ Done (2026-04-10) — `reports/embedding_eval_report.json`; model locked to **Voyage-4-Large**; `long-covid-rag-scope-v3.md` Section 5.3 + this doc updated.
6. **Phase 2 — Full corpus ingest:** Provision Weaviate Cloud, embed ~184k chunks via Voyage API, build `src/index_weaviate.py`, ingest with full metadata payload.
7. **Cleanup (Phase 1e):** Delete `src/ingest_eval_labels_from_report.py`; remove manual relevance column from `suggest_eval_chunks.py` report (superseded by Option B eval).
