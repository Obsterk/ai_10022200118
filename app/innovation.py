"""
innovation.py  —  PART G (innovation component).
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Two complementary features are bundled here:

    1. DOMAIN-SPECIFIC SCORING FUNCTION
       A re-ranker that boosts chunks whose text contains terms from
       a hand-curated Ghana-politics + Ghana-economy lexicon.  This gives
       the retriever prior knowledge about what matters in our domain.

    2. FEEDBACK LOOP FOR IMPROVING RETRIEVAL
       Positive / negative user votes (from the Streamlit UI) are persisted
       to data/processed/feedback.jsonl and used to:
          * permanently *down-weight* chunks that were marked irrelevant
          * permanently *up-weight* chunks that were marked helpful

Both features are pure-Python, cheap, and inspectable — no ML pipelines or
frameworks are used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

from .config import settings
from .data_prep import Chunk
from .vector_store import SearchHit
from .logger_config import get_logger, log_stage

log = get_logger("innovation")

# =============================================================================
# 1. Domain-specific re-ranker
# =============================================================================
_DOMAIN_TERMS = {t.lower() for t in settings.domain_boost_keywords}
_FEEDBACK_FILE = Path(settings.log_file).parent.parent / "data" / "processed" / "feedback.jsonl"


def _domain_density(text: str) -> float:
    """Fraction of tokens that are in our domain lexicon (bounded at 1)."""
    toks = text.lower().split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t.strip(",.;:()%") in _DOMAIN_TERMS)
    return min(hits / max(len(toks), 1) * 10, 1.0)  # scaled so a few hits matter


def _query_overlap(query: str, chunk_text: str) -> float:
    """Jaccard overlap of rare query terms vs chunk (helps for numbers/years)."""
    q = set(query.lower().split())
    c = set(chunk_text.lower().split())
    if not q:
        return 0.0
    return len(q & c) / max(len(q), 1)


def domain_rerank(query: str, hits: List[SearchHit]) -> List[SearchHit]:
    """Apply domain boost + live feedback weights, then re-sort."""
    if not hits:
        return hits
    feedback = _load_feedback_weights()
    w = settings.domain_boost_weight

    rescored: List[SearchHit] = []
    for h in hits:
        boost  = w * _domain_density(h.chunk.text)
        ovl    = 0.5 * w * _query_overlap(query, h.chunk.text)
        fb     = feedback.get(h.chunk.chunk_id, 0.0)
        new_score = h.score + boost + ovl + fb
        rescored.append(SearchHit(chunk=h.chunk, score=new_score, rank=h.rank))

    rescored.sort(key=lambda x: -x.score)
    for r, hit in enumerate(rescored):
        hit.rank = r
    log_stage(log, "innovation.rerank",
              f"Applied domain rerank (+feedback) to {len(rescored)} hits")
    return rescored


# =============================================================================
# 2. Feedback loop — read / write
# =============================================================================
def _load_feedback_weights() -> dict:
    """Aggregate all historical votes into a chunk_id → weight map."""
    weights: dict = {}
    if not _FEEDBACK_FILE.exists():
        return weights
    with _FEEDBACK_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            for cid in ev.get("chunk_ids", []):
                # Each upvote contributes +0.02, each downvote -0.02; capped at ±0.1
                delta = 0.02 if ev.get("helpful") else -0.02
                weights[cid] = max(-0.1, min(0.1, weights.get(cid, 0.0) + delta))
    return weights


def record_feedback(query: str,
                    chunk_ids: List[str],
                    helpful  : bool,
                    note: str = "") -> None:
    """Persist one feedback event so future retrievals benefit from it."""
    _FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "query"    : query,
        "chunk_ids": list(chunk_ids),
        "helpful"  : bool(helpful),
        "note"     : note,
    }
    with _FEEDBACK_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    log_stage(log, "innovation.feedback",
              f"Recorded feedback (helpful={helpful}) for {len(chunk_ids)} chunks")


def feedback_summary() -> dict:
    """For the UI: show how many total feedback events have been logged."""
    if not _FEEDBACK_FILE.exists():
        return {"total": 0, "positive": 0, "negative": 0}
    positive = negative = 0
    with _FEEDBACK_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("helpful"):
                positive += 1
            else:
                negative += 1
    return {"total": positive + negative,
            "positive": positive, "negative": negative}
