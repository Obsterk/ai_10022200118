"""
test_smoke.py
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Minimal smoke tests — run offline (no LLM call, no internet):
    * chunking produces sane output
    * vector store round-trip works
    * hybrid retriever returns relevant hits on the sample CSV

Run with:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from app.config import SAMPLE_DIR
from app.data_prep import load_and_clean_election_csv, row_chunker, sentence_chunker
from app.embeddings import Embedder
from app.vector_store import VectorStore
from app.retriever import HybridRetriever, BM25Index


def test_csv_chunking():
    df = load_and_clean_election_csv(SAMPLE_DIR / "Ghana_Election_Result_sample.csv")
    chunks = row_chunker(df)
    assert chunks, "row_chunker produced nothing"
    assert all(c.text for c in chunks)
    assert all(c.source == "election_csv" for c in chunks)
    assert len(chunks) >= 20


def test_sentence_chunking_basic():
    text = ("Ghana's 2025 budget emphasises fiscal discipline. " * 50)
    chunks = sentence_chunker(text, target_size=80, overlap=10,
                              source="budget_pdf")
    assert chunks
    assert chunks[0].source == "budget_pdf"
    assert all("Ghana" in c.text for c in chunks[:1])


def test_vector_store_roundtrip():
    df = load_and_clean_election_csv(SAMPLE_DIR / "Ghana_Election_Result_sample.csv")
    chunks = row_chunker(df)

    embedder = Embedder()
    vecs = embedder.encode([c.text for c in chunks])

    vs = VectorStore(dim=vecs.shape[1])
    vs.add(chunks, vecs)
    assert len(vs) == len(chunks)

    q = embedder.encode("Who won in the Volta region?")
    hits = vs.search(q.reshape(-1), top_k=3)
    assert hits, "no hits returned"
    assert hits[0].score > 0.0


def test_hybrid_retriever():
    df = load_and_clean_election_csv(SAMPLE_DIR / "Ghana_Election_Result_sample.csv")
    chunks = row_chunker(df)

    embedder = Embedder()
    vecs = embedder.encode([c.text for c in chunks])
    vs = VectorStore(dim=vecs.shape[1])
    vs.add(chunks, vecs)

    bm25 = BM25Index(chunks)
    retriever = HybridRetriever(vs, bm25, embedder)
    res = retriever.retrieve("NDC candidates in Volta", top_k=3)
    assert res.hits, "no hits returned"
    # All top results should at least mention Volta OR NDC
    joined = " ".join(h.chunk.text for h in res.hits).lower()
    assert "volta" in joined or "ndc" in joined
