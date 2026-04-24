"""
diagnose_retrieval.py — isolate which retrieval stage biases results toward
election_csv vs budget_pdf chunks.

Prints top-10 for each stage: DENSE, SPARSE/BM25, RRF (before domain rerank),
AFTER domain rerank — for two queries. Also reports election/budget counts
at depth 10/25/50 to surface biases further down the ranking.

Tries the real pipeline first. If sentence-transformers / rank_bm25 / sklearn
are unavailable, falls back to:
  * DENSE: pre-computed MiniLM vectors from data/index/vector_store.pkl +
           keyword-centroid query proxy (same 384-d space)
  * BM25: pure-numpy BM25Okapi reimplementation (k1=1.5, b=0.75)
  * RRF + domain rerank: pure-Python copies of the real logic
"""
from __future__ import annotations
import sys, json, math, pickle, re
from pathlib import Path
from collections import Counter
from typing import List

import numpy as np

PROJECT_ROOT = Path("/sessions/amazing-friendly-keller/mnt/AI Work/ai_project")
sys.path.insert(0, str(PROJECT_ROOT))

# -------- try real imports ---------------------------------------------
try:
    from app.data_prep import load_chunks
    from app.config import PROCESSED, settings
    HAVE_APP = True
except Exception as e:
    print(f"[warn] cannot import app modules: {e}")
    HAVE_APP = False

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    HAVE_SBERT = True
except Exception:
    HAVE_SBERT = False

try:
    from rank_bm25 import BM25Okapi  # noqa: F401
    HAVE_BM25 = True
except Exception:
    HAVE_BM25 = False

try:
    from app.rag_pipeline import RAGPipeline  # noqa: F401
    from app.retriever    import BM25Index    # noqa: F401
except Exception:
    RAGPipeline = None
    BM25Index   = None

print(f"[env] HAVE_APP={HAVE_APP}  HAVE_SBERT={HAVE_SBERT}  HAVE_BM25={HAVE_BM25}")

# -------- load chunks ---------------------------------------------------
class LiteChunk:
    __slots__ = ("chunk_id", "text", "source", "metadata")
    def __init__(self, cid, txt, src, md=None):
        self.chunk_id = cid; self.text = txt; self.source = src; self.metadata = md or {}

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
VS_PATH     = PROJECT_ROOT / "data" / "index"     / "vector_store.pkl"

with CHUNKS_PATH.open("r", encoding="utf-8") as fh:
    chunks = [LiteChunk(d["chunk_id"], d["text"], d["source"], d.get("metadata", {}))
              for d in (json.loads(l) for l in fh)]
n_elec = sum(1 for c in chunks if c.source == "election_csv")
n_bud  = len(chunks) - n_elec
print(f"[data] {len(chunks)} chunks  election={n_elec}  budget={n_bud}")

# -------- dense ---------------------------------------------------------
MODE_NOTES = []
REAL_DENSE = False
pipe = None
if HAVE_APP and HAVE_SBERT and RAGPipeline is not None:
    try:
        pipe = RAGPipeline(load_chunks(CHUNKS_PATH))
        REAL_DENSE = True
        MODE_NOTES.append("DENSE: real MiniLM via RAGPipeline")
    except Exception as e:
        print(f"[warn] RAGPipeline build failed: {e}")

if not REAL_DENSE:
    if not VS_PATH.exists():
        print("[fatal] no vector_store.pkl; cannot run dense stage")
        sys.exit(1)
    with VS_PATH.open("rb") as fh:
        bundle = pickle.load(fh)
    corpus_mat = bundle["matrix"].astype(np.float32)
    vs_chunks  = bundle["chunks"]
    id2row = {c["chunk_id"]: i for i, c in enumerate(vs_chunks)}
    order  = [id2row[c.chunk_id] for c in chunks]
    corpus_mat = corpus_mat[order]
    MODE_NOTES.append("DENSE: pre-computed MiniLM vectors + keyword-centroid query proxy")

# -------- bm25 ----------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
def _tok(t): return [x.lower() for x in _TOKEN_RE.findall(t)]

REAL_BM25 = False
bm25_real = None
if HAVE_APP and HAVE_BM25 and BM25Index is not None:
    try:
        bm25_real = BM25Index(chunks)
        REAL_BM25 = True
        MODE_NOTES.append("BM25: real rank_bm25")
    except Exception as e:
        print(f"[warn] real BM25 failed: {e}")

if not REAL_BM25:
    class NPBM25:
        def __init__(self, corpus_tokens, k1=1.5, b=0.75):
            self.k1, self.b = k1, b
            self.N = len(corpus_tokens)
            self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float32)
            self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
            df = Counter(); self.tf = []
            for d in corpus_tokens:
                c = Counter(d); self.tf.append(c)
                for t in c: df[t] += 1
            self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        def scores(self, q_tokens):
            sc = np.zeros(self.N, dtype=np.float32)
            for q in q_tokens:
                idf = self.idf.get(q)
                if idf is None: continue
                for i, tfd in enumerate(self.tf):
                    f = tfd.get(q, 0)
                    if f == 0: continue
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    sc[i] += idf * (f * (self.k1 + 1)) / denom
            return sc
    np_bm25 = NPBM25([_tok(c.text) for c in chunks])
    MODE_NOTES.append("BM25: numpy BM25Okapi reimpl (k1=1.5, b=0.75)")

# -------- RRF -----------------------------------------------------------
RRF_K = 60
def rrf(dense, sparse, top_k=10, k_const=RRF_K):
    table = {}
    for cid, _, rk in dense:
        table[cid] = table.get(cid, 0.0) + 1.0 / (k_const + rk + 1)
    for cid, _, rk in sparse:
        table[cid] = table.get(cid, 0.0) + 1.0 / (k_const + rk + 1)
    merged = sorted(table.items(), key=lambda x: -x[1])[:top_k]
    return [(cid, sc, r) for r, (cid, sc) in enumerate(merged)]

# -------- domain rerank (mirror innovation.py) --------------------------
try:
    from app.innovation import domain_rerank as _app_dr
    from app.vector_store import SearchHit
    HAVE_REAL_DR = True
except Exception:
    HAVE_REAL_DR = False
    SearchHit = None

DOMAIN_TERMS = set(settings.domain_boost_keywords) if HAVE_APP else {
    "ghana","volta","ashanti","greater accra","ndc","npp","parliament",
    "constituency","votes","polling","budget","gdp","cedi","mofep",
    "inflation","tax","revenue","expenditure","deficit","policy",
}
W = settings.domain_boost_weight if HAVE_APP else 0.03

def _dom_density(text):
    toks = text.lower().split()
    if not toks: return 0.0
    hits = sum(1 for t in toks if t.strip(",.;:()%") in DOMAIN_TERMS)
    if hits == 0: return 0.0
    return min((hits / 9.0) ** 0.5, 1.0)

def _q_overlap(q, t):
    qs = set(q.lower().split()); cs = set(t.lower().split())
    return (len(qs & cs) / max(len(qs), 1)) if qs else 0.0

id2chunk = {c.chunk_id: c for c in chunks}

def domain_rerank_local(query, ranked):
    if HAVE_REAL_DR and SearchHit is not None:
        # use real one for maximum fidelity
        hits = []
        for cid, sc, rk in ranked:
            ch = id2chunk[cid]
            # need a real Chunk; we use the dict-based one — real_domain_rerank
            # only touches .chunk_id and .text, but builds SearchHit from hits
            hits.append(SearchHit(chunk=type("C",(),{"chunk_id":cid,"text":ch.text,"source":ch.source})(),
                                  score=sc, rank=rk))
        out = _app_dr(query, hits)
        return [(h.chunk.chunk_id, float(h.score), r) for r, h in enumerate(out)]
    out = []
    for cid, sc, rk in ranked:
        ch = id2chunk[cid]
        boost = W * _dom_density(ch.text)
        ovl   = 0.5 * W * _q_overlap(query, ch.text)
        out.append((cid, sc + boost + ovl, rk))
    out.sort(key=lambda x: -x[1])
    return [(cid, sc, r) for r, (cid, sc, _) in enumerate(out)]

# -------- stage runners -------------------------------------------------
def run_dense(query, k=10):
    if REAL_DENSE:
        q_vec = pipe.embedder.encode(query)
        hits = pipe.vs.search(q_vec, top_k=k, min_score=0.0)
        return [(h.chunk.chunk_id, float(h.score), r) for r, h in enumerate(hits)]
    qtoks = set(_tok(query))
    mask = np.array([bool(set(_tok(c.text)) & qtoks) for c in chunks])
    if mask.sum() == 0: mask[:] = True
    qvec = corpus_mat[mask].mean(axis=0)
    qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
    sims = (corpus_mat @ qvec).astype(np.float32)
    top = np.argsort(-sims)[:k]
    return [(chunks[int(i)].chunk_id, float(sims[int(i)]), r) for r, i in enumerate(top)]

def run_sparse(query, k=10):
    if REAL_BM25:
        hits = bm25_real.search(query, top_k=k)
        return [(h.chunk.chunk_id, float(h.score), r) for r, h in enumerate(hits)]
    scores = np_bm25.scores(_tok(query))
    top = np.argsort(-scores)[:k]
    return [(chunks[int(i)].chunk_id, float(scores[int(i)]), r) for r, i in enumerate(top)]

def run_rrf(query, pool=50, k=10):
    return rrf(run_dense(query, k=pool), run_sparse(query, k=pool), top_k=k)

def run_final(query, pool=50, k=10):
    return domain_rerank_local(query, run_rrf(query, pool=pool, k=pool))[:k]

# -------- pretty print --------------------------------------------------
def fmt_row(rank, cid, score):
    ch = id2chunk[cid]
    tag = "E" if ch.source == "election_csv" else "B"
    txt = ch.text.replace("\n", " ")[:60]
    return f"  {rank:2d}. [{tag}] {score:>8.4f}  {cid:30s} | {txt}"

def dump(label, rows):
    e = sum(1 for cid,_,_ in rows if id2chunk[cid].source == "election_csv")
    b = len(rows) - e
    print(f"\n--- {label}  (E={e}, B={b}) ---")
    for r, (cid, sc, _) in enumerate(rows):
        print(fmt_row(r+1, cid, sc))

def src_counts(rows, depth):
    e = sum(1 for cid,_,_ in rows[:depth] if id2chunk[cid].source == "election_csv")
    return e, depth - e

def diagnose(query):
    print("\n" + "=" * 78)
    print(f"QUERY: {query\!r}")
    print("=" * 78)
    d = run_dense(query,  k=50)
    s = run_sparse(query, k=50)
    r = run_rrf(query,    pool=50, k=50)
    f = run_final(query,  pool=50, k=50)
    dump("1) DENSE top-10",                  d[:10])
    dump("2) SPARSE/BM25 top-10",            s[:10])
    dump("3) RRF-FUSED top-10 (pre-rerank)", r[:10])
    dump("4) AFTER domain rerank top-10",    f[:10])
    print("\n[depth analysis: E/B counts at top 10, 25, 50]")
    for label, lst in [("DENSE", d), ("SPARSE", s), ("RRF", r), ("FINAL", f)]:
        parts = []
        for depth in (10, 25, 50):
            e, b = src_counts(lst, depth)
            parts.append(f"t{depth}: E={e:>2} B={b:>2}")
        print(f"  {label:7s}  " + "   ".join(parts))

# -------- main ----------------------------------------------------------
print("\n[mode] " + "  |  ".join(MODE_NOTES))
for q in [
    "What does the 2025 budget say about inflation?",
    "GDP growth 2025",
]:
    diagnose(q)
print("\n[done]")
