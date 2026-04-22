"""
retriever.py  —  PART B (hybrid search, re-ranking, failure-case fix).
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

The retriever combines three techniques to beat single-method baselines:

    1. Dense retrieval  — cosine over MiniLM embeddings
    2. Sparse retrieval — BM25 over a bag-of-words index
    3. Fusion           — Reciprocal Rank Fusion (RRF)  [Cormack et al., 2009]

On top of that a domain-specific re-ranker (see innovation.py) can re-order
the fused results.  This module also contains the *FAILURE-CASE FIX*
(abstain + query rewrite) required by Part B.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import settings
from .data_prep import Chunk
from .embeddings import Embedder
from .vector_store import SearchHit, VectorStore
from .logger_config import get_logger, log_stage

log = get_logger("retriever")


# -----------------------------------------------------------------------------
# BM25 index
# -----------------------------------------------------------------------------
class BM25Index:
    """Thin adapter over rank_bm25 — a keyword-search utility, not a
    RAG framework.  We keep the corpus/vocabulary management ourselves."""

    def __init__(self, chunks: List[Chunk]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.tokens = [_tokenize(c.text) for c in chunks]
        self.bm25   = BM25Okapi(self.tokens)
        log_stage(log, "bm25.build",
                  f"BM25 index built over {len(chunks)} chunks")

    def search(self, query: str, top_k: int) -> List[SearchHit]:
        scores = self.bm25.get_scores(_tokenize(query))
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [SearchHit(chunk=self.chunks[int(i)],
                          score=float(scores[int(i)]), rank=r)
                for r, i in enumerate(idx)]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# -----------------------------------------------------------------------------
# Retriever
# -----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    query           : str
    rewritten_query : Optional[str]
    hits            : List[SearchHit]
    method          : str
    abstained       : bool = False


class HybridRetriever:
    """Dense + Sparse + RRF + optional domain re-ranker."""

    def __init__(self, vector_store: VectorStore,
                 bm25_index: BM25Index,
                 embedder: Embedder,
                 alpha: float = settings.hybrid_alpha,
                 rrf_k: int = settings.rrf_k):
        self.vs        = vector_store
        self.bm25      = bm25_index
        self.embedder  = embedder
        self.alpha     = alpha
        self.rrf_k     = rrf_k

    # ------------------------------------------------------------------ main
    def retrieve(self, query: str,
                 top_k: int = settings.top_k,
                 pool : int = settings.rerank_top_n,
                 domain_rerank: bool = True) -> RetrievalResult:
        log_stage(log, "retrieve.query", f"Q = {query!r}", query=query)

        # (1) dense
        q_vec = self.embedder.encode(query)
        dense_hits = self.vs.search(q_vec, top_k=pool,
                                    min_score=0.0)          # keep all for fusion
        # (2) sparse
        sparse_hits = self.bm25.search(query, top_k=pool)

        # (3) fuse via RRF
        fused = self._rrf(dense_hits, sparse_hits, top_k=pool)

        # (4) domain re-rank (Part-G innovation — imported lazily)
        if domain_rerank:
            from .innovation import domain_rerank as _dr
            fused = _dr(query, fused)

        # (4b) source-balancing guard — MiniLM on our corpus is biased
        # toward the 615 near-duplicate election rows (they share a prefix
        # and schema, so they cluster tightly in embedding space and
        # query vectors tend to land closer to that cluster than to the
        # longer, more varied budget PDF paragraphs).
        # If the top-k comes out mono-source, promote the single best
        # chunk from the other source into 2nd place so the LLM at
        # least sees cross-source evidence when the query is ambiguous.
        fused = self._enforce_source_balance(fused, top_k=top_k)

        fused = fused[:top_k]

        # (5) FAILURE-CASE FIX — abstain if nothing is close enough
        rewritten = None
        abstained = False
        if not fused or fused[0].score < settings.min_similarity:
            rewritten = self._rewrite_query(query)
            if rewritten and rewritten != query:
                log_stage(log, "retrieve.rewrite",
                          f"low-confidence retrieval → rewriting to {rewritten!r}")
                q_vec2 = self.embedder.encode(rewritten)
                d2 = self.vs.search(q_vec2, top_k=pool)
                s2 = self.bm25.search(rewritten, top_k=pool)
                fused = self._rrf(d2, s2, top_k=top_k)
                if domain_rerank:
                    from .innovation import domain_rerank as _dr
                    fused = _dr(rewritten, fused)[:top_k]
            if not fused or fused[0].score < settings.min_similarity:
                abstained = True

        log_stage(log, "retrieve.hits",
                  f"returning {len(fused)} hits, abstained={abstained}",
                  top_k=len(fused),
                  chunk_ids=[h.chunk.chunk_id for h in fused],
                  scores=[round(h.score, 4) for h in fused])
        return RetrievalResult(
            query=query, rewritten_query=rewritten,
            hits=fused,
            method="hybrid-rrf" + ("+domain" if domain_rerank else ""),
            abstained=abstained,
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _rrf(dense: List[SearchHit], sparse: List[SearchHit],
             top_k: int, k_const: int = settings.rrf_k,
             sparse_weight: float = 1.5) -> List[SearchHit]:
        """Weighted Reciprocal-Rank-Fusion.

        BM25 gets a higher weight (1.5×) than dense. Rationale: dense
        retrieval on this corpus suffers from a centroid-pull pathology —
        the 615 election rows share a prefix and schema, so they form a
        tight cluster that query vectors drift toward, even for queries
        about budget content. BM25 doesn't have this problem: it correctly
        scores budget chunks highly for queries like "inflation" or "GDP".
        Weighting sparse > dense lets the stage that's actually right
        win more often.
        """
        table = {}
        by_id = {}
        for h in dense:
            cid = h.chunk.chunk_id
            table[cid] = table.get(cid, 0.0) + 1.0 / (k_const + h.rank + 1)
            by_id[cid] = h
        for h in sparse:
            cid = h.chunk.chunk_id
            table[cid] = table.get(cid, 0.0) + sparse_weight / (k_const + h.rank + 1)
            by_id.setdefault(cid, h)

        merged = sorted(table.items(), key=lambda x: -x[1])[:top_k]
        return [SearchHit(chunk=by_id[cid].chunk, score=score, rank=r)
                for r, (cid, score) in enumerate(merged)]

    @staticmethod
    def _enforce_source_balance(hits: List[SearchHit],
                                 top_k: int) -> List[SearchHit]:
        """Guarantee cross-source representation in top-k when available.

        If the naive top-k comes out mono-source (all election_csv or all
        budget_pdf) but the *second* source has at least one hit in the
        wider pool, swap the lowest-ranked mono-source hit for the best
        other-source hit. This ensures the LLM sees cross-source evidence
        even when one stage (usually dense) is biased by corpus imbalance.
        """
        if not hits or top_k <= 1:
            return hits
        top = hits[:top_k]
        top_sources = {h.chunk.source for h in top}
        if len(top_sources) > 1:
            return hits  # already balanced — nothing to do
        # top is mono-source — find the best hit from the other source
        mono = next(iter(top_sources))
        other_candidates = [h for h in hits[top_k:] if h.chunk.source != mono]
        if not other_candidates:
            return hits  # no other-source hits at all — can't balance
        # Swap the weakest top hit for the strongest other-source hit
        best_other = other_candidates[0]
        rebalanced = top[:-1] + [best_other] + hits[top_k:]
        # re-rank positions for display
        for r, h in enumerate(rebalanced):
            h.rank = r
        return rebalanced

    # ------------------------ Query rewrite (failure-case fix) --------------
    @staticmethod
    def _rewrite_query(q: str) -> Optional[str]:
        """Very small rule-based rewriter — expands common Ghana abbreviations
        and boosts recall when the original query under-retrieves.

        We deliberately keep this simple (no LLM) because the innovation
        component (Part G) adds LLM-based rewriting when enabled.
        """
        mapping = {
            r"\bNDC\b" : "National Democratic Congress (NDC)",
            r"\bNPP\b" : "New Patriotic Party (NPP)",
            r"\bCPP\b" : "Convention Peoples Party (CPP)",
            r"\bEC\b"  : "Electoral Commission",
            r"\bMOFEP\b": "Ministry of Finance and Economic Planning (MOFEP)",
            r"\bGDP\b" : "gross domestic product (GDP)",
            r"\bVAT\b" : "Value Added Tax (VAT)",
            r"\bGH\b"  : "Ghana",
            r"\bcedi\b": "Ghanaian cedi currency",
        }
        rewritten = q
        for pat, repl in mapping.items():
            rewritten = re.sub(pat, repl, rewritten, flags=re.I)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        return rewritten if rewritten.lower() != q.lower() else None
