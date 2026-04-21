"""
run_evaluation.py
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Runs the full Part-E evaluation suite and dumps machine-readable +
human-readable reports into experiment_logs/.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import PROCESSED  # noqa: E402
from app.data_prep import load_chunks  # noqa: E402
from app.llm_client import LLMClient  # noqa: E402
from app.rag_pipeline import RAGPipeline  # noqa: E402
from app.evaluator import (  # noqa: E402
    run_evaluation, dump_results, aggregate_scores,
)


def main() -> int:
    cache = PROCESSED / "chunks.jsonl"
    if not cache.exists():
        print("[err] Run scripts/build_index.py first.", file=sys.stderr)
        return 1

    pipeline = RAGPipeline(load_chunks(cache))
    llm      = LLMClient()

    # consistency_runs=1 keeps total LLM calls under the OpenRouter
    # free-tier daily cap (~50/day). Raise to 3 once you add credits.
    results = run_evaluation(pipeline, llm, consistency_runs=1)
    agg     = aggregate_scores(results)

    dump_path = ROOT / "experiment_logs" / "eval_results.json"
    dump_results(results, dump_path)

    print("\n==================== aggregate scores ====================")
    for k, v in agg.items():
        print(f"{k:<25} : {v}")
    print("==========================================================")
    print(f"Full JSON → {dump_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
