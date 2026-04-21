# Architecture & System Design (Part F)

**ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)**

See also: `architecture/architecture.svg`, `architecture/architecture.mermaid`.

## 1. Data flow (offline)

```
Ghana_Election_Result.csv  ─┐
                            ├─► clean_text() ─► row_chunker() ─┐
2025-Budget-Statement.pdf  ─┘                                  │
                            └─► clean_text() ─► sentence_chunker() ─┤
                                                                    ▼
                                 ┌─────────── Embedder (MiniLM-L6-v2) ────────┐
                                 ▼                                            ▼
                         Custom VectorStore (NumPy, 384-d)              BM25 Index
                                                                  (parallel, same chunks)
```

Both stores are persisted to `data/index/` so the UI boots instantly on subsequent runs.

## 2. Data flow (query-time)

```
User ─► Streamlit UI ─► RAGPipeline.ask()
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
            dense search   BM25 search   domain prior
                 └─────── RRF fusion ─────┘
                          │
                          ▼
                 HybridRetriever result
                          │
               score < 0.15? ──► query rewrite ──► retry
                          │
                          ▼
                 pack_context()  (rank → filter → truncate)
                          │
                          ▼
                 build_prompt()  (guarded + chain-of-thought)
                          │
                          ▼
                 LLMClient.chat (Groq Llama-3.3-70B)
                          │
                          ▼
                 Answer + [#n] citations
                          │
                          ▼
                 Streamlit UI (chunks, scores, prompt, 👍/👎)
                          │
                          ▼                  (feedback loop)
               innovation.record_feedback ─► data/processed/feedback.jsonl
                          │
                          └──► influences future domain_rerank()
```

## 3. Components (files → responsibilities)

| File | Responsibility | Exam part |
|---|---|---|
| `app/data_prep.py` | cleaning, 3 chunkers, chunk IO | A |
| `app/embeddings.py` | MiniLM wrapper + TF-IDF fallback | B |
| `app/vector_store.py` | hand-rolled NumPy cosine store | B |
| `app/retriever.py` | dense + BM25 + RRF + rewrite | B |
| `app/prompt_builder.py` | templates + context packing | C |
| `app/llm_client.py` | Groq chat wrapper | D |
| `app/rag_pipeline.py` | orchestrator with stage logging | D |
| `app/evaluator.py` | adversarial + RAG vs pure LLM | E |
| `app/innovation.py` | domain re-ranker + feedback loop | G |
| `app/streamlit_app.py` | UI | Final |

## 4. Why this design fits Ghana elections + budget

**The two source documents are very different.**

The CSV is structured (one row per constituency result). A natural-language
"row chunk" gives the embedder a tiny but well-formed document that mentions
region, candidate, party, votes, and year in a single sentence. This is
ideal for a bi-encoder.

The budget PDF is long-form prose with numbers and headings. Sentence-
aware chunking at 500 words / 80-word overlap preserves paragraph-level
arguments (e.g., "The Government projects inflation to ease to X% by Q4")
without breaking up the supporting numbers.

**Ghanaian terminology mixes prose with acronyms** (NDC, NPP, EC, MOFEP,
GDP). Dense embeddings handle synonyms poorly for acronyms, so we layer
BM25 — a sparse model that excels at exact-token matches like "NDC". The
two signals are fused with Reciprocal-Rank-Fusion (no score calibration
needed, scale invariant).

**Domain-specific re-ranker.** Even after hybrid retrieval, chunks that
"smell like Ghana politics or budget" deserve a prior. We compute
`domain_density` — the fraction of tokens belonging to our curated
lexicon — and add it as a small bonus. This stops generic sentences
("the government will act") from out-ranking politically loaded ones
("NDC won the Volta Region by a 5:1 margin").

**Abstention + query-rewrite** is critical because the assistant is
public-facing. If the top similarity is below 0.15 (chosen empirically),
we first try rewriting Ghana-specific acronyms to their full forms, then
if that still fails we return:
> "I don't have enough information to answer that from the available sources."

This removes the biggest source of RAG embarrassment: confidently
answering outside the knowledge base.

**Feedback loop.** Every 👍 / 👎 in the UI nudges a chunk's score for
future queries. Over time the retriever adapts to the *actual* preferences
of Academic City users without requiring re-training.

## 5. Trade-offs we explicitly accepted

| Decision | Upside | Downside | Mitigation |
|---|---|---|---|
| NumPy vector store (no FAISS) | fully auditable, zero deps, fits the constraint | O(N·D) query scan | fine for ≤100k chunks; we cap at ~2k |
| MiniLM-L6-v2 (384-d) | small, fast, runs on free Streamlit CPU | weaker than bge-large | hybrid BM25 recovers keyword misses |
| Low temperature (0.1) | grounded answers | less "creative" phrasing | user can override |
| Groq free tier | free & fast (500+ tps) | rate limits on bursty usage | UI retries on 429 |
| Rule-based query rewriter | no extra LLM call | brittle to unseen acronyms | we only trigger on low-confidence |
