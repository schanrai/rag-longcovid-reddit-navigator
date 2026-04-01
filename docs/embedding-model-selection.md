---
name: Embedding Model Selection
overview: Locked architectural decision — we are not constrained to 512-token encoders. Final embedding model (Voyage-4-Large vs corpus-validated alternative) deferred until a data-driven comparison on our Reddit corpus. Chunking pipeline proceeds independently.
todos:
  - id: embedding-eval
    content: Run data-driven embedding comparison (e.g. Voyage-4-Large vs gte-Qwen2-1.5B-instruct) on corpus sample + golden queries; then lock model in scope doc Section 5.3
    status: pending
  - id: lock-post-chunks
    content: Lock post chunk sizing (informed by 32k-class context, not 512) and exclusion rules
    status: pending
  - id: spec-post-schema
    content: Spec post chunk schema in scope doc Section 6
    status: pending
  - id: update-chunk-data
    content: Update chunk_data.py to process posts alongside comments
    status: pending
isProject: false
---

# Embedding Model Selection for Long COVID Reddit RAG

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

## Alternative under consideration: gte-Qwen2-1.5B-instruct (self-host)

From the **MTEB Medical** domain leaderboard (user-provided): ranks **#5** with a high medical-domain mean; **1.3B active params**; **32,768 max tokens**; **~8,960 embedding dimensions** (much larger than Voyage defaults — implications for vector DB storage and search latency).

**Not pursuing for MVP:** **7B-class** medical leaders — they require **GPU** infra ($150–400/mo always-on class, or hourly GPU with cold-start pain). Incompatible with a **~$10/mo** MVP budget and “validate usage first” goal.

**Middle ground:** 1.5B Qwen is theoretically runnable on **small GPU** or **CPU** (slower). Realistic issues: **serverless GPU cold starts**, **CPU query latency**, **8960-dim vectors** vs 1k-class API models.

**Open question:** Medical leaderboard uses **formal** medical text; our corpus is **informal Reddit**. The **better** model for *our* data may not be the higher medical-MTEB rank — hence the planned **head-to-head on our chunks and queries**.

---

## How we got here (decision trail)

1. **Chunk size vs encoder:** Post chunking discussion exposed that chunk limits were tied to an unchosen 512-token placeholder. Sequencing corrected: **long-context models** allowed; chunk parameters follow **data + retrieval**, not arbitrary token caps.
2. **General vs medical MTEB:** API-first options looked strong on broad retrieval; **Medical** filter on MTEB favored **self-hosted 7B** models; **Voyage-3** ranked lower on medical. Upgraded to **Voyage-4** after user caught **missing “latest version” check** (now in [CLAUDE.md](../CLAUDE.md) — verify current release before recommending vendors/models).
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

## What this unlocks for chunk sizing

For any **32k-class** (or even **8k-class**) finalist:

- **Comments:** 200 words / 20 overlap remains consistent with comment length distribution (most single-chunk).
- **Posts:** **300 words** (or similar) is viable for **most** selftext without encoder truncation; tune after posts land in `chunk_data.py`.
- **Embed string:** title + optional summary + body chunk — still a small fraction of 32k context.

---

## Next steps

1. **Embedding:** Run planned **comparison** (timing TBD: e.g. tomorrow / early next week) — then update scope doc **Section 5.3** with **chosen model + rationale + “why not 512-only”**.
2. **Product:** Lock **post** chunk parameters and exclusions; extend [long-covid-rag-scope-v2.md](../artifacts/module_5_context_packing/long-covid-rag-scope-v2.md) Section 6 for post chunks.
3. **Code:** Extend [chunk_data.py](../projects/rag-longcovid-reddit-navigator/src/chunk_data.py) for **post** chunking into `data/` output (alongside comment chunks as designed).
