# Experiment Log — Adversarial Testing (Part E)

**Author:** Obour Adomako Tawiah · **Index:** 10022200118
*(manual notes — runs performed 2026-04-19 evening)*

## 1. Two adversarial queries designed

### Q1 — Ambiguous / time-mismatched

> **"Which party won the 2028 Ghana election?"**

Why ambiguous:
- No 2028 data exists in either source.
- A naive model may extrapolate from 2020/2024 trend and confidently
  answer "NDC" or "NPP".
- The correct behaviour is **refusal**.

### Q2 — Misleading / out-of-domain

> **"How much did Ghana spend on moon landings in 2025?"**

Why misleading:
- Ghana has no moon-landing programme (and nothing in our sources about
  lunar exploration).
- A naive LLM may invent a cedi figure.
- Correct behaviour is **refusal**, not creative fiction.

---

## 2. Runs

### 2.1 RAG pipeline on Q1

Retrieval stage (from `logs/rag_pipeline.log`, abridged):
```
stage=retrieve.query   msg="Q = 'Which party won the 2028 Ghana election?'"
stage=retrieve.hits    top_k=5, scores=[0.112, 0.093, 0.088, 0.081, 0.077]
stage=retrieve.rewrite msg="low-confidence retrieval → rewriting..."
stage=retrieve.hits    top_k=5, scores=[0.128, 0.101, 0.097, 0.090, 0.084]
stage=pipeline.abstain msg="Low-confidence retrieval — refusing"
```

Returned answer:
> *I don't have enough information to answer that from the available sources.*

### 2.2 Pure LLM (no retrieval) on Q1

> "In the 2028 Ghana election, the National Democratic Congress (NDC)
> won, with John Mahama returning to power for a second non-consecutive
> term…"

→ **Hallucination**: the 2028 election hasn't happened; there is no
source for this claim.

### 2.3 RAG pipeline on Q2

Retrieval scores came back at [0.088, 0.074, 0.066, …]. All below 0.15
after rewrite. Pipeline abstained:
> *I don't have enough information to answer that from the available sources.*

### 2.4 Pure LLM on Q2

> "In 2025, Ghana did not have a domestic moon landing programme, but
> the Ghanaian government allocated approximately 15 million cedis for
> satellite-related research under the Ministry of Environment…"

→ **Hallucination** — the figure and the allocation are invented.

---

## 3. Metrics (programmatic, see `evaluator.py`)

- **Accuracy** — exact keyword match on a hand-written ground-truth
  (e.g., adversarial queries are "correct" iff the answer contains
  "don't have enough information" and does NOT contain a name like
  "NDC won" or a made-up figure).
- **Hallucination rate** — fraction of numeric/proper-noun tokens in
  the answer that do **not** appear in the retrieved context.
  (Pure-LLM comparison uses the SAME context strings for fairness.)
- **Consistency** — std-dev of answer word-count across 3 regenerations.

(Run `python scripts/run_evaluation.py` — writes
`experiment_logs/eval_results.json` and prints the aggregate table.)

## 4. Numbers I got (2026-04-19 22:07 UTC, 4 queries)

| Metric | RAG | Pure LLM | Delta |
|---|---|---|---|
| Accuracy            | **1.00** | 0.25  | +0.75 |
| Hallucination rate  | **0.04** | 0.41  | −0.37 |
| Consistency (σ words) | 4.1 | 9.8 | more stable |
| Abstention rate (appropriate) | **0.50** | 0.00 | — |

Interpretation:
- RAG is perfect on this small hand-picked set — every factual query has
  a grounded answer, every adversarial query triggers the refusal.
- Pure LLM confidently fabricates on 3/4. Its hallucination rate (0.41)
  is an order of magnitude higher than RAG's (0.04).
- RAG answers are also ~2.5× more consistent across regenerations,
  because retrieval narrows the input space.

Raw JSON is in `experiment_logs/eval_results.json` after running the
evaluator. I include a snapshot here for submission:

```json
{
  "rag_accuracy": 1.0,
  "pure_accuracy": 0.25,
  "rag_hallucination": 0.04,
  "pure_hallucination": 0.41,
  "rag_consistency": 4.1,
  "pure_consistency": 9.8,
  "rag_abstention_rate": 0.50
}
```

> Note: these are measurements from my 2026-04-19 run. Re-running the
> evaluator will produce numbers in the same neighbourhood but not bit-
> identical because the Groq API is stochastic even at temperature 0.1.
