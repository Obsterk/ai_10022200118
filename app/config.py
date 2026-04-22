"""config.py — centralised runtime configuration."""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
PROCESSED   = DATA_DIR / "processed"
SAMPLE_DIR  = DATA_DIR / "sample"
INDEX_DIR   = DATA_DIR / "index"
LOG_DIR     = ROOT / "logs"

for p in (RAW_DIR, PROCESSED, SAMPLE_DIR, INDEX_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    llm_api_key   : str = os.getenv("LLM_API_KEY", os.getenv("OPENROUTER_API_KEY", os.getenv("GEMINI_API_KEY", "")))
    llm_model     : str = os.getenv("LLM_MODEL",   "openrouter/free")
    llm_timeout   : int = 60
    llm_temperature: float = 0.1
    groq_api_key  : str = os.getenv("LLM_API_KEY", os.getenv("GEMINI_API_KEY", os.getenv("GROQ_API_KEY", "")))
    groq_model    : str = os.getenv("LLM_MODEL",   os.getenv("GROQ_MODEL", "gemini-2.0-flash"))
    groq_timeout  : int = 60

    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim  : int = 384

    chunk_size   : int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 80))

    top_k          : int   = int(os.getenv("TOP_K", 5))
    rerank_top_n   : int   = 10
    min_similarity : float = 0.05
    hybrid_alpha   : float = 0.6
    rrf_k          : int   = 60

    domain_boost_keywords: List[str] = field(default_factory=lambda: [
        "ghana", "volta", "ashanti", "greater accra", "ndc", "npp", "parliament",
        "constituency", "votes", "polling", "budget", "gdp", "cedi", "mofep",
        "inflation", "tax", "revenue", "expenditure", "deficit", "policy",
    ])
    domain_boost_weight: float = 0.03

    log_level : str = os.getenv("LOG_LEVEL", "INFO")
    log_file  : Path = LOG_DIR / "rag_pipeline.log"


settings = Settings()
