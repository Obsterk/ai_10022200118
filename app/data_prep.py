"""
data_prep.py  —  PART A: Data cleaning & chunking.
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

This module is written from scratch — NO LangChain / LlamaIndex splitters.

Three chunking strategies are implemented because the two source documents
have very different shapes:

    1. row_chunker()       → one structured sentence per CSV row
                             (Ghana_Election_Result.csv)
    2. fixed_chunker()     → baseline word-window chunker
    3. sentence_chunker()  → semantic-aware chunker using NLTK sentence
                             tokenisation + greedy packing to target size
                             (used for the PDF budget statement)

Chunk size = 500 tokens (~375 words), overlap = 80 tokens.
See docs/chunking_analysis.md for the comparative study justifying these.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .config import settings
from .logger_config import get_logger, log_stage

log = get_logger("data_prep")


# -----------------------------------------------------------------------------
# Chunk container
# -----------------------------------------------------------------------------
@dataclass
class Chunk:
    """Unit of retrieval — text plus rich metadata for downstream filtering."""
    chunk_id : str
    text     : str
    source   : str                 # "election_csv" | "budget_pdf"
    metadata : dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# Cleaning helpers
# -----------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_BULLETS = re.compile(r"^[\-\u2022\u25AA\u25CF\u25A0\*]+\s*", re.M)
_PAGE_NUM = re.compile(r"^\s*\d{1,4}\s*$", re.M)
_HEADER_NOISE = re.compile(
    r"(LG\s*-\s*Public|© *Ministry of Finance.*?Statement)", re.I
)


def clean_text(text: str) -> str:
    """Normalise unicode, strip PDF artefacts, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _HEADER_NOISE.sub(" ", text)
    text = _BULLETS.sub("", text)
    text = _PAGE_NUM.sub(" ", text)
    # kill hyphenation line-breaks (common in PDFs: "eco-\nnomic" → "economic")
    text = re.sub(r"-\n", "", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = _WS.sub(" ", text)
    return text.strip()


# -----------------------------------------------------------------------------
# CSV cleaning + row chunking  (Ghana_Election_Result.csv)
# -----------------------------------------------------------------------------
def load_and_clean_election_csv(path: Path) -> pd.DataFrame:
    """Read + clean the election CSV."""
    log_stage(log, "data_prep.csv.load", f"Reading {path}")
    df = pd.read_csv(path)

    # Normalise column headers so the row chunker never has to guess
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop rows that are 100% empty (EC CSVs sometimes have footer totals)
    df = df.dropna(how="all").reset_index(drop=True)

    # String columns — strip + title-case where appropriate
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": ""})

    # Numeric coercion where the column name screams numeric
    for col in df.columns:
        if any(k in col for k in ("vote", "percent", "year", "count", "share")):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the primary key (candidate or constituency) is empty
    primary_candidates = [c for c in ("candidate", "constituency", "party")
                          if c in df.columns]
    if primary_candidates:
        df = df.dropna(subset=primary_candidates, how="all")

    log_stage(log, "data_prep.csv.loaded",
              f"{len(df)} clean rows, columns={list(df.columns)}")
    return df


def row_chunker(df: pd.DataFrame) -> List[Chunk]:
    """Turn each row into ONE natural-language chunk.

    Election rows are structured — turning them into prose lets the
    sentence-transformer embed them semantically, and lets BM25 match
    keywords like 'Volta' or 'NDC' without any custom tokeniser.
    """
    chunks: List[Chunk] = []
    for idx, row in df.iterrows():
        pairs = [f"{k.replace('_', ' ')}: {v}"
                 for k, v in row.items() if str(v) and str(v) != "nan"]
        text = "Ghana Election Result — " + "; ".join(pairs) + "."
        chunks.append(Chunk(
            chunk_id=f"election::row::{idx}",
            text=text,
            source="election_csv",
            metadata={"row_index": int(idx), **{k: (str(v) if pd.notna(v) else "")
                                                for k, v in row.items()}},
        ))
    log_stage(log, "data_prep.chunk.csv",
              f"Produced {len(chunks)} row-chunks from CSV")
    return chunks


# -----------------------------------------------------------------------------
# PDF chunking  (2025 Budget Statement)
# -----------------------------------------------------------------------------
def load_pdf_text(path: Path) -> str:
    """Pull raw text from a PDF using pypdf (no framework)."""
    from pypdf import PdfReader

    log_stage(log, "data_prep.pdf.load", f"Reading {path}")
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            log_stage(log, "data_prep.pdf.error",
                      f"Page {i}: {e}", extra={"page": i})
    raw = "\n".join(pages)
    cleaned = clean_text(raw)
    log_stage(log, "data_prep.pdf.cleaned",
              f"{len(pages)} pages, {len(cleaned)} chars after cleaning")
    return cleaned


def _word_tokens(text: str) -> List[str]:
    """Cheap whitespace tokeniser — consistent across chunkers."""
    return text.split()


def fixed_chunker(text: str,
                  size: int = settings.chunk_size,
                  overlap: int = settings.chunk_overlap,
                  source: str = "budget_pdf") -> List[Chunk]:
    """Baseline sliding-window chunker.  Used in the comparative study."""
    words = _word_tokens(text)
    chunks, step = [], max(size - overlap, 1)
    for i in range(0, len(words), step):
        window = words[i:i + size]
        if len(window) < 50:          # skip tiny tails
            break
        chunks.append(Chunk(
            chunk_id=f"{source}::fixed::{i:06d}",
            text=" ".join(window),
            source=source,
            metadata={"strategy": "fixed", "start_word": i,
                      "end_word": i + len(window)},
        ))
    log_stage(log, "data_prep.chunk.fixed",
              f"fixed-window chunker produced {len(chunks)} chunks")
    return chunks


def sentence_chunker(text: str,
                     target_size: int = settings.chunk_size,
                     overlap: int = settings.chunk_overlap,
                     source: str = "budget_pdf") -> List[Chunk]:
    """Sentence-aware greedy packer.

    Splits the document into sentences (NLTK punkt) and greedily packs them
    into chunks whose word count is ~target_size, always finishing on a
    full sentence boundary.  Between consecutive chunks we repeat the tail
    `overlap` words so a sentence straddling a boundary is still retrievable
    from either side.
    """
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    sentences = nltk.sent_tokenize(text)
    chunks: List[Chunk] = []
    buf: List[str] = []
    wc = 0
    idx = 0

    def flush(buf: List[str], idx: int):
        if not buf:
            return
        chunk_text = " ".join(buf).strip()
        chunks.append(Chunk(
            chunk_id=f"{source}::sent::{idx:05d}",
            text=chunk_text,
            source=source,
            metadata={"strategy": "sentence",
                      "sentence_count": len(buf),
                      "word_count": len(_word_tokens(chunk_text))},
        ))

    for s in sentences:
        sw = _word_tokens(s)
        if wc + len(sw) > target_size and buf:
            flush(buf, idx)
            idx += 1
            # carry the tail as overlap (keep last N words as new prefix)
            carry_words = _word_tokens(" ".join(buf))[-overlap:]
            buf = [" ".join(carry_words)] if carry_words else []
            wc = len(carry_words)
        buf.append(s)
        wc += len(sw)
    flush(buf, idx)

    log_stage(log, "data_prep.chunk.sentence",
              f"sentence-aware chunker produced {len(chunks)} chunks "
              f"(target {target_size} words, overlap {overlap})")
    return chunks


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------
def build_all_chunks(election_csv: Optional[Path] = None,
                     budget_pdf  : Optional[Path] = None) -> List[Chunk]:
    """Run the full Part-A pipeline and return a unified chunk list."""
    all_chunks: List[Chunk] = []

    if election_csv and Path(election_csv).exists():
        df = load_and_clean_election_csv(Path(election_csv))
        all_chunks.extend(row_chunker(df))

    if budget_pdf and Path(budget_pdf).exists():
        text = load_pdf_text(Path(budget_pdf))
        all_chunks.extend(sentence_chunker(text))

    log_stage(log, "data_prep.done",
              f"Total chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    log_stage(log, "data_prep.persist", f"Wrote {path}")


def load_chunks(path: Path) -> List[Chunk]:
    with path.open("r", encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]
