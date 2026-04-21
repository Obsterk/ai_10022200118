"""
embeddings.py  —  PART B (embedding pipeline).
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Thin wrapper around sentence-transformers.  We do the normalisation
ourselves (L2 unit vectors) so downstream cosine similarity reduces to
a single dot product — no framework magic.

If sentence-transformers is unavailable (offline), we fall back to a
lightweight TF-IDF encoder (scikit-learn) so the pipeline still runs.
The fallback is clearly flagged in the logs.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .config import settings
from .logger_config import get_logger, log_stage, Timer

log = get_logger("embeddings")


class Embedder:
    """Singleton-ish encoder — load the model once, encode many times."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._st_model  = None
        self._tfidf     = None
        self._mode      = None       # "sbert" | "tfidf"
        self._dim       = None

    # -------- public API -----------------------------------------------------
    def encode(self, texts: Union[str, List[str]],
               batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        self._ensure_loaded(texts)
        with Timer(log, "embed", f"Encoding {len(texts)} texts [{self._mode}]"):
            if self._mode == "sbert":
                vecs = self._st_model.encode(
                    texts, batch_size=batch_size,
                    convert_to_numpy=True, show_progress_bar=False,
                    normalize_embeddings=normalize,
                ).astype(np.float32)
            else:                     # TF-IDF fallback
                sparse = self._tfidf.transform(texts)
                vecs = sparse.toarray().astype(np.float32)
                if normalize:
                    vecs = _l2_normalize(vecs)
        return vecs

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.encode("probe")
        return self._dim

    @property
    def mode(self) -> str:
        return self._mode or "uninitialised"

    # -------- internals ------------------------------------------------------
    def _ensure_loaded(self, sample_texts: List[str]) -> None:
        if self._mode is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            log_stage(log, "embed.load",
                      f"Loading sentence-transformer: {self.model_name}")
            self._st_model = SentenceTransformer(self.model_name)
            self._mode = "sbert"
            self._dim  = self._st_model.get_sentence_embedding_dimension()
        except Exception as e:
            log_stage(log, "embed.fallback",
                      f"SBERT load failed ({e}) — using TF-IDF fallback")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2),
                max_features=4096, norm=None,
            )
            # must fit on something — use the first batch plus a small prior
            self._tfidf.fit(sample_texts or ["ghana budget election"])
            self._mode = "tfidf"
            self._dim  = len(self._tfidf.vocabulary_)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# disk cache (skip recomputing embeddings between runs)
# -----------------------------------------------------------------------------
def save_embeddings(vecs: np.ndarray, ids: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"vectors": vecs, "ids": ids}, fh)
    log_stage(log, "embed.persist",
              f"Saved {vecs.shape} vectors → {path}")


def load_embeddings(path: Path):
    with path.open("rb") as fh:
        bundle = pickle.load(fh)
    return bundle["vectors"], bundle["ids"]
