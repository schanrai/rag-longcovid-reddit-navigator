---
name: Embedding, Infrastructure & Vector DB Decisions
overview: Eval shortlist locked (Qwen3-0.6B, Jasper-600M, Qwen3-4B, Voyage-4-Large). Vector DB and hosting path determined by eval winner — Weaviate Cloud (Voyage) or Qdrant Cloud + middle-tier container (self-hosted model). Chunking pipeline complete. Next blocker: golden query set.
todos:
  - id: embedding-eval
    content: Run data-driven embedding comparison (Voyage-4-Large, Qwen3-Embedding-0.6B, Qwen3-Embedding-4B, Jasper-Token-Compression-600M) on corpus sample + golden queries; then lock model in scope doc Section 5.3
    status: pending
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

**What is not locked yet:** The specific model ID (API vs self-hosted) until a **data-driven comparison** on our actual corpus (informal Reddit health text), planned soon after this plan.

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

The embedding model eval must measure these dimensions. Ordered by priority:

| # | Dimension | What we're measuring | Non-negotiable? |
|---|---|---|---|
| 1 | **Retrieval quality** | Best results on actual Reddit corpus + golden queries. The deciding factor. | Yes |
| 2 | **Query latency** | Must be under 15 seconds per query. | Yes — hard ceiling |
| 3 | **Cost** | API per-token (Voyage) vs middle-tier cloud hosting (self-hosted). Must fit MVP budget. | Yes |
| 4 | **Pipeline complexity** | Native Weaviate integration (Voyage) vs separate embedding step + vector DB. Number of moving parts. | No — but a tiebreaker |
| 5 | **Vector dimensionality** | 1024 (Voyage) vs 8960 (gte-Qwen2) vs TBD (7B models). Affects vector DB storage cost and search speed. | No — but has cost/latency implications |
| 6 | **Domain fit** | MTEB Medical rank is a starting signal, but corpus is informal Reddit, not clinical text. Eval on our data settles this. | Measured by #1 |
| 7 | **Ingestion throughput** | Time to embed full 184k-chunk corpus. API rate limits may throttle Voyage; local CPU is slow but unconstrained. | No — one-time batch job, but affects iteration speed |
| 8 | **Support ecosystem** | Voyage: docs + Weaviate native integration + reranker. Self-hosted: husband's hands-on Qdrant/sentence-transformers experience. | No — but a tiebreaker |

---

## Next steps

1. ~~**Product:** Lock post chunk parameters and exclusions; extend scope doc Section 6 for post chunks.~~ ✅ Done (2026-04-01)
2. ~~**Code:** Extend `chunk_data.py` for post chunking into `data/` output.~~ ✅ Done (2026-04-01)
3. ~~**Confirm eval candidates:** Pull MTEB Medical + Social/Blog leaderboards, cross-reference, lock shortlist.~~ ✅ Done (2026-04-02). Shortlist: Qwen3-Embedding-0.6B, Jasper-Token-Compression-600M, Qwen3-Embedding-4B, Voyage-4-Large. gte-Qwen2-1.5B-instruct dropped.
4. **Build golden query set:** 15-20 representative queries (symptom, treatment, timeline, prevalence). Blocks the entire eval.
5. **Run embedding eval:** Compare 4 shortlisted candidates on eval dimensions (retrieval quality, latency, cost, dims, pipeline complexity). Then update [long-covid-rag-scope-v2.md](../../artifacts/module_5_context_packing/long-covid-rag-scope-v2.md) **Section 5.3** with chosen model + rationale.
