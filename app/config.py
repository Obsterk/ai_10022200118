"""
config.py  —  centralised runtime configuration.
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

All tunable knobs live here so nothing is hard-coded deep inside the pipeline.
Values can be overridden by environment variables (or the .env file).
"""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # dotenv is optional in production


# -----------------------------------------------------------------------------
# paths
# -----------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED   = DATA_DIR / "processed"
SAMPLE_DIR  = DATA_DIR / "sample"
INDEX_DIR   = DATA_DIR / "index"
LOG_DIR     = ROOT / "logs"

for p in (RAW_DIR, PROCESSED, SAMPLE_DIR, INDEX_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# runtime settings
# -----------------------------------------------------------------------------
@dataclass
class Settings:
    # LLM --------------------------------------------------------------------
    # Gemini is the active provider (free, no phone verification).
    # The legacy `groq_*` fields stay for any file that still imports them.
    llm_api_key   : str = os.getenv("LLM_API_KEY", os.getenv("OPENROUTER_API_KEY", os.getenv("GEMINI_API_KEY", "")))
    llm_model     : str = os.getenv("LLM_MODEL",   "openrouter/free")
    llm_timeout   : int = 60
    llm_temperature: float = 0.1           # low — we want grounded answers

    # Back-compat aliases (old Groq field names) — map to the same values.
    groq_api_key  : str = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("GROQ_API_KEY", "")))
    groq_model    : str = os.getenv("LLM_MODEL",   os.getenv("GROQ_MODEL", "gemini-2.0-flash"))
    groq_timeout  : int = 60

    # Embedding --------------------------------------------------------------
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dim  : int = 384              # MiniLM produces 384-d vectors

    # Chunking ---------------------------------------------------------------
    # 500 tokens ≈ 375 English words — fits neatly within LLM context
    # 80-token overlap preserves semantic continuity across chunk boundaries
    chunk_size   : int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 80))

    # Retrieval --------------------------------------------------------------
    top_k          : int   = int(os.getenv("TOP_K", 5))
    rerank_top_n   : int   = 10                 # fetch 10, keep top-k after rerank
    min_similarity : float = 0.05               # below this → abstain (calibrated for RRF-fused scores)
    hybrid_alpha   : float = 0.6                # weight of dense vs sparse
    rrf_k          : int   = 60                 # Reciprocal-Rank-Fusion const

    # Domain-specific re-ranker (Part G) -------------------------------------
    domain_boost_keywords: List[str] = field(default_factory=lambda: [
        "ghana", "volta", "ashanti", "greater accra", "ndc", "npp", "parliament",
        "constituency", "votes", "polling", "budget", "gdp", "cedi", "mofep",
        "inflation", "tax", "revenue", "expenditure", "deficit", "policy",
    ])
    domain_boost_weight: float = 0.03    # calibrated against RRF score magnitudes (~0.03) so the prior boost nudges ranking without dominating retrieval evidence

    # Logging ---------------------------------------------------------------
    log_level : str = os.getenv("LOG_LEVEL", "INFO")
    log_file  : Path = LOG_DIR / "rag_pipeline.log"


settings = Settings()
