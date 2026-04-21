# Experiment Log — Prompt Engineering

**Author:** Obour Adomako Tawiah · **Index:** 10022200118
*(manual notes, kept on paper first then typed up)*

## Test query used throughout:
> "How did the NDC perform in the Volta region?"

I kept retrieval constant (top_k=5, hybrid+domain) and varied only the
prompt. Temperature fixed at 0.1 for reproducibility.

---

## V1 — "naive"

**System:** *"You are a helpful assistant. Use the context to answer the
question."*

**Output:**
> The NDC did very well in the Volta region, which has historically been
> a stronghold of the party. Richard Kwami Sefe won Ho Central with a
> landslide, and candidates in Anlo and Keta also secured victories…

Problems observed:
- "Keta" appears nowhere in the retrieved context — **hallucinated**.
- No citation tags.
- Confident tone despite no grounding instruction.
- Tends to pad with generic political commentary.

## V2 — "guarded"

**System:** explicit "answer only from context", refuses if unknown, demands `[#n]` citations.

**Output:**
> In the Volta region, Richard Kwami Sefe (NDC) won Ho Central with 44 321
> votes, defeating Divine Richard Bosson (NPP, 8 876) [#1]. NDC's Gbolu
> Richard Sellah also won Anlo with 27 654 votes [#2].

Improvements:
- All numbers are in the context.
- "Keta" hallucination is gone.
- Citations appear.

Remaining issue:
- When I asked the adversarial query *"Who won the 2028 election?"* the
  model still occasionally produced a hedged guess instead of the exact
  refusal sentence.

## V3 — "guarded + CoT"  ← final choice

**System:** same as V2 but adds:
- a private `<scratchpad>` block forcing step-by-step reasoning about
  which chunks are relevant
- a numbered 6-rule list including an **exact** refusal string
- a 150-word cap on the visible answer

**Output:**
> Richard Kwami Sefe (NDC) won Ho Central with 44 321 votes (over
> Divine Richard Bosson, NPP, 8 876) [#1]. Gbolu Richard Sellah (NDC)
> won Anlo with 27 654 votes [#2]. Both victories came by more than a
> 3:1 margin, consistent with NDC's historical dominance in the Volta
> region [#1][#2].

**Scratchpad was stripped** before display by `_strip_scratchpad()`.

For the adversarial query *"Who won the 2028 election?"* V3 consistently
produced exactly:
> I don't have enough information to answer that from the available sources.

...which is programmatically checkable (see `evaluator.py::GROUND_TRUTH`).

---

## Evidence of improvement (10-query evaluation set)

I hand-scored 10 queries for three metrics, 2026-04-18:

| Prompt | Grounded | Cited correctly | Exact refusal on OOD |
|---|---|---|---|
| V1 naive | 5 / 10 | 0 / 10 | 0 / 3 |
| V2 guarded | 8 / 10 | 7 / 10 | 2 / 3 |
| V3 guarded+CoT | **10 / 10** | **10 / 10** | **3 / 3** |

CoT adds ~200 ms to generation time — acceptable.

## Notes on context window management

Implemented in `prompt_builder.py::pack_context`:

- Hits are sorted by score (desc).
- Greedy add until we hit `max_context_tokens=3500`.
- Each chunk gets a `[#n]` header so the LLM can cite it.
- If nothing fits (shouldn't happen after abstain), fall back to `"<<no
  relevant context>>"` — the LLM then triggers the refusal rule.

Tried `max_context_tokens=6000`. Accuracy didn't improve but latency did.
Kept 3500.
