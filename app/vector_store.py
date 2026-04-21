"""
vector_store.py  —  PART B (custom vector storage).
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

A hand-rolled, in-memory vector database.  We deliberately avoid FAISS and
Chroma so every step is transparent and auditable.

Design:
    * vectors are stored as a single row-major (N × D) float32 NumPy matrix
    * all vectors are L2-normalised at insertion time
    * similarity search = matrix × vector dot product (= cosine similarity)
    * chunk metadata lives in a parallel list aligned by index

This fits ~100k chunks easily in RAM (~150 MB at dim=384) which is more
than enough for the Academic City corpus.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .data_prep import Chunk
from .logger_config import get_logger, log_stage

log = get_logger("vector_store")


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank : int


class VectorStore:
    """Custom in-memory vector store with cosine similarity search."""

    def __init__(self, dim: int):
        self.dim      : int           = dim
        self.matrix   : np.ndarray    = np.zeros((0, dim), dtype=np.float32)
        self.chunks   : List[Chunk]   = []
        self._id2pos  : dict          = {}

    # ------------------------------------------------------------------ build
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim, (
            f"expected (N, {self.dim}); got {vectors.shape}"
        )
        assert len(chunks) == vectors.shape[0]
        vectors = _l2_normalize(vectors.astype(np.float32))
        start   = len(self.chunks)
        self.matrix = np.vstack([self.matrix, vectors])
        for i, c in enumerate(chunks):
            self.chunks.append(c)
            self._id2pos[c.chunk_id] = start + i
        log_stage(log, "vector_store.add",
                  f"Added {len(chunks)} chunks; total={len(self.chunks)}")

    # ----------------------------------------------------------------- query
    def search(self, query_vec: np.ndarray, top_k: int = 5,
               min_score: float = 0.0,
               source_filter: Optional[str] = None) -> List[SearchHit]:
        if len(self.chunks) == 0:
            return []
        q = _l2_normalize(query_vec.reshape(1, -1)).astype(np.float32)
        sims = (self.matrix @ q.T).ravel()

        if source_filter:
            mask = np.array([c.source == source_filter for c in self.chunks])
            sims = np.where(mask, sims, -np.inf)

        k = min(top_k, int(np.sum(sims > -np.inf)))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]        # sort the top-k
        hits: List[SearchHit] = []
        for rank, i in enumerate(idx):
            s = float(sims[i])
            if s < min_score:
                continue
            hits.append(SearchHit(chunk=self.chunks[int(i)], score=s, rank=rank))
        return hits

    # ---------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({
                "dim"   : self.dim,
                "matrix": self.matrix,
                "chunks": [c.to_dict() for c in self.chunks],
            }, fh)
        log_stage(log, "vector_store.save",
                  f"Saved {len(self.chunks)} chunks → {path}")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        with path.open("rb") as fh:
            b = pickle.load(fh)
        vs = cls(dim=b["dim"])
        vs.matrix = b["matrix"].astype(np.float32)
        vs.chunks = [Chunk(**d) for d in b["chunks"]]
        vs._id2pos = {c.chunk_id: i for i, c in enumerate(vs.chunks)}
        log_stage(log, "vector_store.load",
                  f"Loaded {len(vs.chunks)} chunks from {path}")
        return vs

    def __len__(self) -> int:
        return len(self.chunks)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)
