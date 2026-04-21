# Experiment Log — RAG vs Pure LLM Comparison (Part E)

**Author:** Obour Adomako Tawiah · **Index:** 10022200118
*(manual notes · run 2026-04-19)*

## Setup

- Both systems use Groq `llama-3.3-70b-versatile` at temperature 0.1.
- **RAG system** = full pipeline (hybrid retrieval + domain rerank + CoT prompt).
- **Pure LLM** = same model, no retrieval; just the question in the user turn.
- 6-query dev set (factual + adversarial), 3 regenerations each.

## Per-query table

| # | Query type | Query (abridged) | RAG answer | Pure LLM answer | RAG correct? | Pure correct? |
|---|---|---|---|---|---|---|
| 1 | factual | "Who won Ho Central in 2020?" | "Richard Kwami Sefe (NDC) with 44 321 votes [#1]" | "NDC won Ho Central; I don't know the exact vote count." | ✅ | ❌ (vague) |
| 2 | factual | "What parties contested in 2020?" | "NDC and NPP are mentioned in the results [#1][#2]" | "NDC, NPP, and several smaller parties including the CPP and GUM." | ✅ | ⚠ (mentions parties not in our source) |
| 3 | factual | "What does the 2025 budget say about inflation?" | grounded paragraph citing [#3] | fabricates "projected at 13.5%" with no source | ✅ | ❌ (unverifiable) |
| 4 | factual | "Which region had the largest NPP margin?" | picks Greater Accra / Ashanti row with exact numbers [#1] | hedges — "likely Ashanti" | ✅ | ⚠ |
| 5 | adversarial (ambiguous) | "Who won the 2028 election?" | **refuses** | confidently invents an NDC win in 2028 | ✅ | ❌ (hallucinates the future) |
| 6 | adversarial (misleading) | "Ghana's moon-landing budget 2025?" | **refuses** | fabricates ~15 million cedis | ✅ | ❌ |

## Aggregate

| Metric | RAG | Pure LLM |
|---|---:|---:|
| Accuracy (strict) | 6/6 = **1.00** | 0.5/6 ≈ 0.08 |
| Accuracy (lenient, partial OK) | 6/6 = **1.00** | 2/6 ≈ 0.33 |
| Hallucination rate (token-level) | **0.04** | 0.38 |
| Refusal on OOD queries | 2/2 = **1.00** | 0/2 = 0.00 |
| Consistency (σ answer words, avg) | 4.1 | 9.8 |
| Avg latency (ms) | 1 420 (retrieval 12 + LLM ~1 400) | 1 050 |

## What this means for Academic City

1. **Public-facing correctness**: if this bot is deployed as an info desk
   for students asking about Ghana elections or the 2025 Budget, the pure
   LLM is actively dangerous — it would confidently misstate budget
   figures. The RAG grounding completely removes that risk on OOD queries.

2. **Acceptable latency overhead**: +370 ms for retrieval + context assembly
   is trivial next to the quality gap.

3. **Consistency matters**: if a student asks the same question twice,
   our RAG system gives them the *same* answer 95 % of the time (within a
   ±5-word rewording). The pure LLM varies much more widely, which
   makes debugging and trust impossible.

## Caveat / threats to validity I actually worried about

- The test set is small (6 queries, 3 regenerations). I extended it to 20
  queries on 2026-04-19 23:10 and the direction of results didn't change
  (RAG ≥ 0.95 accuracy, pure ≤ 0.25). Raw run in
  `experiment_logs/eval_results.json`.
- Both systems share the same base model, so this measures *grounding*
  gain, not model-quality gain. That's exactly the effect the exam asks
  me to measure.
- Token-level hallucination scoring is proxy (false positives when the
  model paraphrases a number like "44 321 votes" as "44,321"). I normalised
  digit separators before scoring (see `evaluator._extract_claim_tokens`).
