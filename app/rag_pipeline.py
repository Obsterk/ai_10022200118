"""
rag_pipeline.py  —  PART D (the end-to-end orchestrator).
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

    User Query → Retrieval → Context Selection → Prompt → LLM → Response

Every stage logs:
    * the inputs it received
    * the decisions it made
    * the latency it took

The return value is a rich `RAGResponse` so the Streamlit UI can display
everything the examiner requires:
    - retrieved documents
    - similarity scores
    - the final prompt sent to the LLM
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import settings
from .data_prep import Chunk, build_all_chunks, load_chunks, save_chunks
from .embeddings import Embedder
from .vector_store import VectorStore, SearchHit
from .retriever import HybridRetriever, BM25Index, RetrievalResult
from .prompt_builder import build_prompt, PackedContext
from .llm_client import LLMClient
from .logger_config import get_logger, log_stage, Timer

log = get_logger("pipeline")


# -----------------------------------------------------------------------------
# rich response container
# -----------------------------------------------------------------------------
@dataclass
class RAGResponse:
    query            : str
    rewritten_query  : Optional[str]
    answer           : str
    retrieval        : RetrievalResult
    packed_context   : PackedContext
    system_prompt    : str
    user_prompt      : str
    prompt_variant   : str
    abstained        : bool
    used_chunk_ids   : List[str] = field(default_factory=list)

    def display_dict(self) -> dict:
        return {
            "query"          : self.query,
            "rewritten_query": self.rewritten_query,
            "answer"         : self.answer,
            "abstained"      : self.abstained,
            "method"         : self.retrieval.method,
            "hits": [
                {
                    "rank" : h.rank,
                    "score": round(h.score, 4),
                    "id"   : h.chunk.chunk_id,
                    "source": h.chunk.source,
                    "preview": h.chunk.text[:240] + ("…" if len(h.chunk.text) > 240 else ""),
                } for h in self.retrieval.hits
            ],
            "used_chunk_ids": self.used_chunk_ids,
            "prompt_variant": self.prompt_variant,
            "final_prompt": self.system_prompt + "\n---\n" + self.user_prompt,
        }


# -----------------------------------------------------------------------------
# RAG pipeline
# -----------------------------------------------------------------------------
class RAGPipeline:
    """Single-entry facade over every Part-A–G component."""

    def __init__(self,
                 chunks       : List[Chunk],
                 embedder     : Optional[Embedder] = None,
                 llm          : Optional[LLMClient] = None,
                 prompt_variant: str = "guarded+cot"):
        self.chunks  = chunks
        self.embedder = embedder or Embedder()
        self.llm      = llm or LLMClient()
        self.prompt_variant = prompt_variant

        log_stage(log, "pipeline.init",
                  f"Embedding {len(chunks)} chunks for vector store")
        vecs = self.embedder.encode([c.text for c in chunks])
        self.vs = VectorStore(dim=vecs.shape[1])
        self.vs.add(chunks, vecs)

        self.bm25 = BM25Index(chunks)
        self.retriever = HybridRetriever(self.vs, self.bm25, self.embedder)
        log_stage(log, "pipeline.ready",
                  f"Index built (dim={self.vs.dim}, N={len(self.vs)})")

    # ----------------------------------------------------------------- query
    def ask(self, query: str,
            top_k: int = settings.top_k,
            prompt_variant: Optional[str] = None) -> RAGResponse:
        variant = prompt_variant or self.prompt_variant

        # 1. Retrieval
        with Timer(log, "pipeline.retrieval", "Hybrid retrieval"):
            retrieval = self.retriever.retrieve(query, top_k=top_k)

        # 2. Early abstain
        if retrieval.abstained:
            msg = ("I don't have enough information to answer that from "
                   "the available sources.")
            log_stage(log, "pipeline.abstain",
                      "Low-confidence retrieval — refusing to answer")
            return RAGResponse(
                query=query,
                rewritten_query=retrieval.rewritten_query,
                answer=msg,
                retrieval=retrieval,
                packed_context=PackedContext("", [], 0),
                system_prompt="(not called — abstained)",
                user_prompt="(not called — abstained)",
                prompt_variant=variant,
                abstained=True,
                used_chunk_ids=[],
            )

        # 3. Prompt build
        with Timer(log, "pipeline.prompt", "Prompt construction"):
            system, user, packed = build_prompt(
                query, retrieval.hits, variant=variant,
            )

        # 4. LLM call
        with Timer(log, "pipeline.llm", "LLM generation"):
            answer = self.llm.chat(system, user)

        # Tidy up: if the model uses the scratchpad format, keep just the answer
        clean_answer = _strip_scratchpad(answer)

        log_stage(log, "pipeline.done",
                  f"Answer generated ({len(clean_answer)} chars)")
        return RAGResponse(
            query=query,
            rewritten_query=retrieval.rewritten_query,
            answer=clean_answer,
            retrieval=retrieval,
            packed_context=packed,
            system_prompt=system,
            user_prompt=user,
            prompt_variant=variant,
            abstained=False,
            used_chunk_ids=[h.chunk.chunk_id for h in packed.used_hits],
        )


# -----------------------------------------------------------------------------
# helper
# -----------------------------------------------------------------------------
def _strip_scratchpad(text: str) -> str:
    import re
    text = re.sub(r"<scratchpad>.*?</scratchpad>", "",
                  text, flags=re.S | re.I).strip()
    text = re.sub(r"^ANSWER:\s*", "", text, flags=re.I).strip()
    return text


# -----------------------------------------------------------------------------
# one-shot builder helpers (used by build_index.py and Streamlit)
# -----------------------------------------------------------------------------
def build_from_sources(election_csv: Path, budget_pdf: Path,
                       persist_to: Optional[Path] = None) -> RAGPipeline:
    chunks = build_all_chunks(election_csv=election_csv, budget_pdf=budget_pdf)
    if persist_to:
        save_chunks(chunks, persist_to)
    return RAGPipeline(chunks)


def build_from_cache(chunks_jsonl: Path) -> RAGPipeline:
    return RAGPipeline(load_chunks(chunks_jsonl))
