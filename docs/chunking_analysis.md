# Chunking Strategy — Justification & Comparative Analysis (Part A)

**ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)**

## 1. Why we need two chunkers

The exam gives us two very different sources:

| Source | Structure | Intent when queried |
|---|---|---|
| `Ghana_Election_Result.csv` | structured tabular — constituency / candidate / votes | "Who won the X constituency?" — factual pointy questions |
| `2025-Budget-Statement.pdf` | long-form prose, bullets, tables | "What does the budget say about inflation?" — paragraph answers |

A single generic splitter (as a framework would provide) would serve both
badly. Dumping a CSV row into a 1 000-character window wastes space; splitting
the PDF row-by-row loses the surrounding argument. I therefore implemented
*three* chunkers and pick per source.

## 2. Strategies implemented

### 2.1 `row_chunker()` — one row, one chunk (CSV)
Each row becomes a single natural-language sentence:
```
Ghana Election Result — region: Volta; constituency: Ho Central;
candidate: Richard Kwami Sefe; party: NDC; votes: 44321; year: 2020.
```
Why: the bi-encoder can embed it meaningfully, BM25 can match
"Volta"/"NDC" exactly, and any query targeting a specific fact
retrieves precisely the right row with no neighbouring noise.

### 2.2 `sentence_chunker()` — sentence-aware greedy packer (PDF)
Uses NLTK `sent_tokenize` to split the cleaned PDF into sentences,
then greedily packs sentences into chunks of **≈500 words** until the
budget is exceeded, always finishing at a sentence boundary.

Between consecutive chunks we carry an **80-word overlap** (≈1 paragraph)
so sentences that straddle a boundary remain retrievable from both sides.

### 2.3 `fixed_chunker()` — baseline, not used in production
A classic word-window splitter (500 words, 80 overlap). Kept in the repo
so the ablation study below is reproducible.

## 3. Justification of 500 words / 80-word overlap

| Parameter | Choice | Rationale |
|---|---|---|
| chunk_size | 500 words (~380 tokens) | A paragraph is usually 80–200 words; 500 lets us keep 2-3 paragraphs in one chunk → enough context for a retriever, small enough that the LLM can still see 5–7 of them in a single prompt. |
| overlap | 80 words | Roughly one full sentence. Guarantees a sentence that straddles a chunk border is still intact in the next chunk. |
| target token budget (LLM context) | 3 500 tokens | Leaves ~4 500 tokens for the answer within Llama-3.3-70B's 128K window — cheap even at top_k=10. |

## 4. Comparative study (manual experiment)

Setup: 10 queries on the sample CSV + a 40-page extract from the budget
PDF. For each chunker, I ran retrieval with `top_k=5` and counted
**Retrieval Quality (RQ)** = manual relevance of the top-1 hit (yes=1/no=0).
Numbers are from a 2026-04-18 notebook session.

| Strategy | Avg chunk len | #chunks | RQ@1 (top-1 correct) | Avg latency (ms) | Comment |
|---|---|---|---|---|---|
| fixed 500/80 (PDF) | 500 w | 86 | 0.70 (7/10) | 7.1 | breaks mid-sentence, diluted matches |
| sentence 500/80 (PDF) | 488 w | 79 | **0.90 (9/10)** | 7.4 | preserved argument structure |
| row-chunker (CSV) | 20 w | 30 | **1.00 (10/10)** | 0.6 | perfect match for factual queries |
| sentence 300/50 (PDF, smaller) | 293 w | 134 | 0.80 (8/10) | 11.9 | chunks too small → missing context |
| sentence 900/120 (PDF, larger) | 872 w | 42 | 0.60 (6/10) | 7.0 | chunks too big → wrong paragraph leaks in |

**Conclusion**: sentence 500/80 is the sweet spot for the PDF;
row-chunker is unambiguously correct for the CSV.

## 5. Pitfalls I hit (and fixed)

1. **Hyphenation line-breaks** in the PDF ("eco-\nnomic" → "economic")
   were splitting words. Fix in `clean_text()`: `re.sub(r"-\n", "", text)`.
2. **Page numbers** were becoming their own chunk. Fix: `_PAGE_NUM` regex
   strips any line that is purely digits.
3. **LG-Public header stamp** appeared on every page and polluted BM25
   with irrelevant noise. Fix: `_HEADER_NOISE` regex.
4. NLTK on Streamlit Cloud doesn't have `punkt` by default. Fix:
   `nltk.download("punkt", quiet=True)` guarded by `try/except LookupError`.
   Also download `punkt_tab` which newer NLTK needs.

## 6. Code locations

- Chunkers and cleaning — `app/data_prep.py`
- Config (size/overlap) — `app/config.py` (`chunk_size`, `chunk_overlap`)
- Logs produced by each chunker — `logs/rag_pipeline.log`
  (look for `stage="data_prep.chunk.*"`)
