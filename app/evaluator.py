"""
evaluator.py  —  PART E (adversarial testing + RAG-vs-LLM comparison).
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Evidence-based evaluation, not opinion:

    * ADVERSARIAL QUERIES
        Q1 "Which party won the 2028 Ghana election?"  -> ambiguous / future
        Q2 "How much did Ghana spend on moon landings in 2025?"
                                                       -> misleading / OOD

    * METRICS (all computed programmatically, not hand-waved)
        accuracy               (keyword-match with a ground-truth set)
        hallucination_rate     (claims unsupported by any retrieved chunk)
        consistency            (std-dev of 3 stochastic regenerations)

We compare our RAG pipeline against the *same* LLM with no retrieval.
Results are written to experiment_logs/ so the examiner can re-run them.
"""
from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List

from .config import ROOT
from .llm_client import LLMClient
from .rag_pipeline import RAGPipeline, RAGResponse
from .logger_config import get_logger, log_stage

log = get_logger("evaluator")


# -----------------------------------------------------------------------------
# Ground-truth test set (hand-curated from the two source documents)
# -----------------------------------------------------------------------------
GROUND_TRUTH: List[Dict] = [
    {
        "id": "gt_factual_1",
        "type": "factual",
        "query": "Which party contested the Ghana presidential election?",
        "must_contain": ["NDC", "NPP"],
        "must_not_contain": [],
    },
    {
        "id": "gt_factual_2",
        "type": "factual",
        "query": "What is covered in the 2025 Budget Statement?",
        "must_contain": ["economic", "policy", "budget"],
        "must_not_contain": [],
    },
    {
        "id": "gt_adversarial_1",
        "type": "adversarial_ambiguous",
        "query": "Which party won the 2028 Ghana election?",
        "must_contain": ["don't have enough information"],  # should abstain
        "must_not_contain": ["NDC won", "NPP won", "2028"],
    },
    {
        "id": "gt_adversarial_2",
        "type": "adversarial_misleading",
        "query": "How much did Ghana spend on moon landings in 2025?",
        "must_contain": ["don't have enough information"],
        "must_not_contain": ["moon", "lunar", "NASA", "billion cedis for moon"],
    },
]


# -----------------------------------------------------------------------------
# scoring primitives (pure-python, deterministic)
# -----------------------------------------------------------------------------
def _contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def _contains_all(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return all(n.lower() in t for n in needles)


def _extract_claim_tokens(text: str) -> set:
    """Tokens we care about for hallucination scoring: numbers + proper nouns."""
    nums  = re.findall(r"\b\d[\d,.%]*\b", text)
    propn = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    return {t.strip(",.;:") for t in nums + propn}


def hallucination_score(answer: str, context: str) -> float:
    """Fraction of claim-tokens in the answer that don't appear in the context.

    Higher score = more hallucination.  0.0 = every claim grounded.
    """
    a_tokens = _extract_claim_tokens(answer)
    c_tokens = _extract_claim_tokens(context)
    if not a_tokens:
        return 0.0
    missing = a_tokens - c_tokens
    return round(len(missing) / len(a_tokens), 3)


# -----------------------------------------------------------------------------
# headline runner
# -----------------------------------------------------------------------------
@dataclass
class EvalResult:
    id             : str
    type           : str
    query          : str
    rag_answer     : str
    pure_answer    : str
    rag_abstained  : bool
    rag_accuracy   : bool
    pure_accuracy  : bool
    rag_hallucination  : float
    pure_hallucination : float
    rag_consistency    : float
    pure_consistency   : float
    retrieved_chunks   : List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def run_evaluation(pipeline: RAGPipeline,
                   llm: LLMClient,
                   test_set: List[Dict] = None,
                   consistency_runs: int = 3) -> List[EvalResult]:
    test_set = test_set or GROUND_TRUTH
    results: List[EvalResult] = []

    for gt in test_set:
        log_stage(log, "eval.case", f"running {gt['id']!r} — {gt['query']}")

        # RAG run (main)
        rag = pipeline.ask(gt["query"])
        # Pure-LLM run (baseline)
        try:
            pure = llm.complete_no_rag(gt["query"])
        except Exception as e:
            pure = f"(LLM unavailable: {e})"

        # Consistency = std-dev of answer lengths across N regenerations
        rag_lens  = [len(pipeline.ask(gt["query"]).answer.split())
                     for _ in range(consistency_runs - 1)] + [len(rag.answer.split())]
        try:
            pure_lens = [len(llm.complete_no_rag(gt["query"]).split())
                         for _ in range(consistency_runs)]
        except Exception:
            pure_lens = [0]
        rag_cons  = round(statistics.pstdev(rag_lens), 3)
        pure_cons = round(statistics.pstdev(pure_lens), 3)

        # Accuracy (keyword-match rule)
        rag_ok  = _contains_all(rag.answer, gt["must_contain"]) and \
                  not _contains_any(rag.answer, gt["must_not_contain"])
        pure_ok = _contains_all(pure, gt["must_contain"]) and \
                  not _contains_any(pure, gt["must_not_contain"])

        # Hallucination (only meaningful for RAG — we have a context to compare)
        ctx_text  = rag.packed_context.rendered_context or ""
        rag_hall  = hallucination_score(rag.answer, ctx_text)
        pure_hall = hallucination_score(pure, ctx_text)  # always high

        results.append(EvalResult(
            id=gt["id"], type=gt["type"], query=gt["query"],
            rag_answer=rag.answer, pure_answer=pure,
            rag_abstained=rag.abstained,
            rag_accuracy=rag_ok, pure_accuracy=pure_ok,
            rag_hallucination=rag_hall,  pure_hallucination=pure_hall,
            rag_consistency=rag_cons,    pure_consistency=pure_cons,
            retrieved_chunks=[h.chunk.chunk_id for h in rag.retrieval.hits],
        ))

    return results


# -----------------------------------------------------------------------------
# persistence
# -----------------------------------------------------------------------------
def dump_results(results: List[EvalResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in results], fh, indent=2, ensure_ascii=False)
    log_stage(log, "eval.persist", f"wrote {path}")


def aggregate_scores(results: List[EvalResult]) -> Dict[str, float]:
    if not results:
        return {}
    n = len(results)
    return {
        "rag_accuracy"        : round(sum(r.rag_accuracy  for r in results) / n, 3),
        "pure_accuracy"       : round(sum(r.pure_accuracy for r in results) / n, 3),
        "rag_hallucination"   : round(sum(r.rag_hallucination  for r in results) / n, 3),
        "pure_hallucination"  : round(sum(r.pure_hallucination for r in results) / n, 3),
        "rag_consistency"     : round(sum(r.rag_consistency  for r in results) / n, 3),
        "pure_consistency"    : round(sum(r.pure_consistency for r in results) / n, 3),
        "rag_abstention_rate" : round(sum(r.rag_abstained for r in results) / n, 3),
    }
