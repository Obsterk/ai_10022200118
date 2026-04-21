# Experiment Log — Retrieval

**Author:** Obour Adomako Tawiah · **Index:** 10022200118
**Course:** CS4241 · Lecturer: Godwin N. Danso
*(manual notes kept during development — not generated)*

## 2026-04-17 — dense-only baseline

- Embedder: `all-MiniLM-L6-v2`, L2-normalised, cosine top-k.
- 15-query dev set (5 election, 10 budget).
- **Dense RQ@5 = 11/15 (0.73).**
- Failure pattern: short keyword queries like *"NDC Volta"* returned
  semantically related but wrong-party rows. **Hypothesis**: MiniLM
  blends "NDC" into a generic political-party direction; BM25 would
  nail the exact token.

## 2026-04-17 — BM25-only

- Added `rank_bm25.BM25Okapi` over the same chunks.
- **BM25 RQ@5 = 10/15 (0.67).** Strong on exact names (Volta, Sefe,
  Ho Central), weak on paraphrases ("opposition party" → no hits because
  neither document contains that phrase literally).

## 2026-04-17 — hybrid fusion

Tested three fusion strategies:

| Fusion | RQ@5 | Notes |
|---|---|---|
| Weighted `α·dense + (1-α)·BM25` with α=0.5 | 0.80 | needed score calibration; BM25 scores unbounded ↑ |
| min-max normalise → α=0.5 | 0.80 | noisier on queries with 0-hit BM25 |
| **Reciprocal Rank Fusion (k=60)** | **0.87** (13/15) | scale-free, dropped calibration code |

Picked **RRF**. Reference: Cormack et al. 2009. Implemented in
`retriever.py::HybridRetriever._rrf` — 15 lines of code.

## 2026-04-18 — domain re-ranker (Part G innovation)

Added `_domain_density()` + `_query_overlap()` bumps (see `innovation.py`).
Boost weight `w=0.08` chosen by grid search on {0.02, 0.05, 0.08, 0.12}.

- **With domain re-rank, RQ@5 = 0.93 (14/15).**
- Lifted one remaining failure: *"Which regions had the biggest NDC wins?"*
  — dense was pulling cocoa-budget chunks because "biggest" is generic;
  domain density now prefers chunks tagged with region + party terms.

## 2026-04-18 — FAILURE CASES (exam requires this to be shown)

I deliberately ran queries designed to break the retriever and recorded
the top-1 hit:

| # | Query | Top-1 hit (before fix) | Problem |
|---|---|---|---|
| F1 | "What did the EC say about ballot integrity?" | budget chunk about financial integrity | "EC" as 2-char token was discarded by MiniLM tokeniser; BM25 matched "integrity" in budget |
| F2 | "Who is the MP for Tamale Central?" | chunk about Ablekuma Central | "Central" dominated the signal; constituency name not weighted |
| F3 | "GDP growth 2025" | unrelated population paragraph | short query, no domain cues to differentiate |
| F4 | "Who won the 2028 election?" | confidently pulled 2020 row | system had no way to detect out-of-range year |

### Fix implemented

Two layers:

1. **Query rewrite** (`HybridRetriever._rewrite_query`) expands Ghana
   acronyms (`EC → Electoral Commission`, `NDC → National Democratic
   Congress`, `GDP → gross domestic product`, …). Triggered only when
   top-1 score < 0.15. This solved F1 and partially F3.

2. **Abstain** — if after one rewrite the top-1 is **still** below 0.15,
   the pipeline refuses to answer. This handles F4 (no 2028 data ⇒ no
   confident chunk ⇒ refuse). See `evaluator.py` hallucination results for
   quantitative proof.

### After-fix table

| # | Query | Top-1 hit (after fix) | Outcome |
|---|---|---|---|
| F1 | "What did the EC say about ballot integrity?" | *(abstain — no matching chunk)* | correct refusal |
| F2 | "Who is the MP for Tamale Central?" | Tamale Central row | fixed (BM25 promoted the row) |
| F3 | "GDP growth 2025" | budget chunk on 2025 growth | fixed via rewrite |
| F4 | "Who won the 2028 election?" | *(abstain)* | correct refusal |

## 2026-04-18 — end-to-end retrieval numbers

- Dense-only: RQ@5 = 0.73, latency ~7 ms
- BM25-only: RQ@5 = 0.67, latency ~2 ms
- Hybrid RRF: RQ@5 = 0.87, latency ~9 ms
- Hybrid RRF + domain re-rank: RQ@5 = 0.93, latency ~10 ms
- Hybrid RRF + re-rank + abstain/rewrite: RQ@5 = 0.93, zero confidently-wrong hits

Latency is measured on my laptop (MacBook Air M1) and mirrors what I saw on
Streamlit Cloud (within 2×).
