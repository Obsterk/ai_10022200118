# Design Decisions

**ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)**

## Guiding rules I gave myself before coding

1. Every component must be **inspectable in <50 lines of code**. If I
   can't fit the logic in a single screen, it's too abstract.
2. No hidden state — every config value lives in `app/config.py`.
3. Ship nothing that depends on a RAG framework. sentence-transformers
   is fine (it is *just* an embedding model), rank_bm25 is fine (just a
   keyword-search utility). LangChain / LlamaIndex / Haystack / Guidance
   are all banned. (Checked: none are in `requirements.txt`.)
4. Structured logging at every stage — grep-friendly, not pretty.
5. Degrade gracefully — if the embedding model fails to download,
   fall back to TF-IDF. If the LLM is down, abstain cleanly.

## Key choices, with reasoning

### 1. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dim output, ~80 MB, runs on Streamlit Cloud's 1 vCPU.
- Out-of-the-box cosine-normalised when we pass `normalize_embeddings=True`.
- Alternative bge-small-en-v1.5 scored ~1 % better in my chunking study
  (see `experiment_logs/chunking_experiments.md`) but doubled boot time.

### 2. Vector store: custom NumPy (N × D) matrix
- Cosine similarity = matrix · query (normalised) = single GEMV call.
- ~2k chunks × 384 dims = ~3 MB — sub-millisecond search.
- FAISS/Chroma would be overkill and forbidden.

### 3. Chunking: one strategy per source
- CSV → row_chunker (1 row = 1 chunk, structured → prose conversion).
- PDF → sentence_chunker (NLTK punkt + greedy pack, 500 words, 80 overlap).
- A fixed-window chunker is kept for the ablation study only.

### 4. Hybrid retrieval with Reciprocal-Rank-Fusion
- Dense signal = semantic match.
- Sparse (BM25) signal = exact keyword match (critical for NDC/NPP/Volta/…).
- RRF is scale-free — I don't have to calibrate cosines vs BM25 raw scores.
- Reference: Cormack, Clarke, Büttcher (SIGIR 2009).

### 5. Failure-case fix = abstain + rule-based rewrite
- If top-1 fused score < 0.15 I expand Ghanaian acronyms and retry once.
- If *still* < 0.15 → refuse to answer. Better silent than confidently wrong.

### 6. Prompt = guarded + chain-of-thought
- System prompt forbids outside knowledge, demands [#n] citations, and
  specifies the exact refusal string so accuracy is programmatically
  checkable.
- I chose a short `<scratchpad>…</scratchpad>` CoT because it measurably
  reduced citation errors on my 10-query dev set (see
  `experiment_logs/prompt_experiments.md`).

### 7. Innovation (Part G): domain re-ranker + feedback loop
- Domain lexicon is just 22 hand-picked terms — cheap but effective.
- Feedback is dumped to `data/processed/feedback.jsonl`, aggregated on
  every query, capped at ±0.10 so it can't dominate cosine scores.

### 8. UI: Streamlit
- Pushes the query, shows all retrieved chunks with scores, shows the
  final prompt the LLM saw, exposes feedback buttons.
- Deploys free to Streamlit Community Cloud with one git push.

## Constraint audit

```
$ grep -EiR "langchain|llama_index|haystack|guidance|dspy" . --include=*.py --include=requirements.txt
  (no results)
```

All the plumbing (chunking, embedding invocation, vector search, BM25 wiring,
rank fusion, re-ranking, prompt packaging, LLM call, logging) is implemented
in this repo directly. Line counts:

| Module | LOC |
|---|---|
| app/data_prep.py | ~180 |
| app/embeddings.py | ~95 |
| app/vector_store.py | ~90 |
| app/retriever.py | ~135 |
| app/prompt_builder.py | ~95 |
| app/rag_pipeline.py | ~115 |
| app/evaluator.py | ~160 |
| app/innovation.py | ~115 |
| app/llm_client.py | ~55 |
| app/logger_config.py | ~60 |
| **Total core** | **~1100 LOC** |

Small enough that every line is defensible.
