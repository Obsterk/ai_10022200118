"""
prompt_builder.py  —  PART C (prompt engineering + context management).
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Three prompt variants are defined so the experiment logs can compare them
side-by-side:

    V1  "naive"         — just dump chunks + question
    V2  "guarded"       — adds anti-hallucination rules and cite-or-abstain
    V3  "guarded+cot"   — V2 plus a short chain-of-thought scratchpad
                          (our final choice — used by default)

Context-window management:
    - we sort hits by score (desc)
    - greedily add chunks until we hit `max_context_tokens`
    - each chunk is annotated with a [#n] citation label the LLM must use
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .vector_store import SearchHit
from .logger_config import get_logger, log_stage

log = get_logger("prompt_builder")


# -----------------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "naive": (
        "You are a helpful assistant. Use the context to answer the question."
    ),
    "guarded": (
        "You are Academic City's RAG assistant. You must answer ONLY from the "
        "provided CONTEXT. If the CONTEXT does not contain the answer, reply "
        "EXACTLY with: 'I don't have enough information to answer that from "
        "the available sources.' Do not invent facts, figures, or dates. "
        "Always cite the context passages you used with their [#n] tags."
    ),
    "guarded+cot": (
        "You are Academic City's RAG assistant specialising in Ghana "
        "elections and the 2025 Ghana Budget Statement.\n"
        "Rules (MUST follow):\n"
        "  1. Answer ONLY from the CONTEXT below. Do not use outside knowledge.\n"
        "  2. If the CONTEXT is insufficient, reply exactly:\n"
        "     'I don't have enough information to answer that from the available sources.'\n"
        "  3. Quote numbers, dates and names verbatim from the CONTEXT.\n"
        "  4. Cite passages by their [#n] tag after the sentence they support.\n"
        "  5. First, THINK step by step in a private <scratchpad> block about "
        "     which passages are relevant. Then give your final ANSWER.\n"
        "  6. Keep the final answer concise (≤ 150 words) unless asked otherwise."
    ),
}

USER_TEMPLATE = (
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION: {question}\n\n"
    "Remember the rules. Return:\n"
    "<scratchpad>your reasoning</scratchpad>\n"
    "ANSWER: <final answer with [#n] citations>"
)


# -----------------------------------------------------------------------------
# Context window management
# -----------------------------------------------------------------------------
@dataclass
class PackedContext:
    rendered_context : str
    used_hits        : List[SearchHit]
    token_estimate   : int


def _approx_tokens(text: str) -> int:
    """1 token ~ 0.75 words → words / 0.75  (rough but cheap & consistent)."""
    return max(1, int(len(text.split()) / 0.75))


def pack_context(hits: List[SearchHit],
                 max_context_tokens: int = 3500) -> PackedContext:
    """Sort by score, then greedily pack until the budget is exhausted."""
    sorted_hits = sorted(hits, key=lambda h: -h.score)
    used, lines, total = [], [], 0

    for i, h in enumerate(sorted_hits, start=1):
        tok = _approx_tokens(h.chunk.text)
        if total + tok > max_context_tokens and used:
            break
        src_tag = h.chunk.source
        lines.append(
            f"[#{len(used) + 1}] (source={src_tag}, score={h.score:.3f}, "
            f"id={h.chunk.chunk_id})\n{h.chunk.text}"
        )
        used.append(h)
        total += tok

    rendered = "\n\n---\n\n".join(lines) if lines else "<<no relevant context>>"
    log_stage(log, "prompt.pack",
              f"Packed {len(used)}/{len(hits)} chunks (~{total} tokens)")
    return PackedContext(rendered, used, total)


# -----------------------------------------------------------------------------
# Public builder
# -----------------------------------------------------------------------------
def build_prompt(question: str,
                 hits: List[SearchHit],
                 variant: str = "guarded+cot",
                 max_context_tokens: int = 3500
                 ) -> Tuple[str, str, PackedContext]:
    """Return (system, user, packed_context)."""
    if variant not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt variant: {variant}")
    pc = pack_context(hits, max_context_tokens=max_context_tokens)
    system = SYSTEM_PROMPTS[variant]
    user   = USER_TEMPLATE.format(context=pc.rendered_context, question=question)
    log_stage(log, "prompt.build",
              f"variant={variant} sys_tokens≈{_approx_tokens(system)} "
              f"user_tokens≈{_approx_tokens(user)}",
              prompt_tokens=_approx_tokens(system) + _approx_tokens(user))
    return system, user, pc
