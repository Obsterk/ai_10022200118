"""
build_index.py
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Builds the chunk corpus and the embedding-backed vector store once,
so the Streamlit UI boots in a few seconds after the first run.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import RAW_DIR, PROCESSED, INDEX_DIR, SAMPLE_DIR  # noqa: E402
from app.data_prep import build_all_chunks, save_chunks  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
from app.vector_store import VectorStore  # noqa: E402


def main() -> int:
    csv = RAW_DIR / "Ghana_Election_Result.csv"
    pdf = RAW_DIR / "2025-Budget-Statement.pdf"
    if not csv.exists():
        print(f"[warn] {csv} missing — falling back to sample CSV")
        csv = SAMPLE_DIR / "Ghana_Election_Result_sample.csv"
    if not pdf.exists():
        print(f"[warn] {pdf} missing — budget PDF will be skipped")
        pdf = None

    chunks = build_all_chunks(election_csv=csv, budget_pdf=pdf)
    save_chunks(chunks, PROCESSED / "chunks.jsonl")
    print(f"[ok  ] saved {len(chunks)} chunks → {PROCESSED / 'chunks.jsonl'}")

    embedder = Embedder()
    vecs = embedder.encode([c.text for c in chunks])
    vs = VectorStore(dim=vecs.shape[1])
    vs.add(chunks, vecs)
    vs.save(INDEX_DIR / "vector_store.pkl")
    print(f"[ok  ] saved vector store ({len(vs)} chunks, dim={vs.dim}) "
          f"→ {INDEX_DIR / 'vector_store.pkl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
