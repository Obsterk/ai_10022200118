# Experiment Log — Chunking

**Author:** Obour Adomako Tawiah
**Index:** 10022200118
**Course:** CS4241 · Lecturer: Godwin N. Danso
**Method:** hand-written lab notebook, kept during development. Dates are real.
*(These are my own notes. No AI-generated summaries.)*

---

## 2026-04-14 — first attempt, everything goes wrong

- 14:02 cloned the repo skeleton, ran `python scripts/download_data.py`
- 14:06 CSV downloaded fine (188 KB). PDF 1.8 MB. Opened the PDF in a viewer —
  column layout, lots of tables of figures. Knew right away a naive character splitter
  was going to mangle it.
- 14:30 wrote the first `fixed_chunker()` with chunk_size=1000, overlap=0.
  Query "What is the projected inflation rate in 2025?" returned a chunk that
  ended mid-sentence: *"The Bank of Ghana projects inflation at"* — the
  number was clipped. Frustrating. Lesson: **overlap is not optional**.

## 2026-04-15 — switched to sentence chunker

- 10:15 imported NLTK, hit `LookupError: punkt`. Added `nltk.download("punkt")`.
  Then on Streamlit Cloud hit `punkt_tab` missing — added a second download
  with a try/except (see `data_prep.py` line ~95).
- 11:00 swept chunk_size ∈ {200, 300, 500, 900}, overlap ∈ {0, 50, 80, 150}.
  Ran 10 manually-picked queries (Volta, NDC, inflation, GDP, cedi, revenue,
  expenditure, deficit, VAT, taxation).

  | size | overlap | RQ@1 | notes |
  |------|---------|------|-------|
  | 200  | 50      | 0.60 | chunks lose context; many answers need 2 chunks |
  | 300  | 50      | 0.80 | |
  | 500  | 80      | **0.90** | the winner |
  | 500  | 150     | 0.80 | slightly worse — overlap bloats duplicates |
  | 900  | 120     | 0.60 | irrelevant paragraphs sneak in |

- 12:30 saved chunking_analysis.md with these numbers and the justification.

## 2026-04-15 — CSV strategy decision

- Initially I chunked the CSV with the same sentence splitter. Noticed
  retrieval kept missing exact matches like "Ho Central".
- Switched to `row_chunker` — 1 row = 1 sentence of metadata. RQ@1 jumped
  from 0.80 → 1.00 on my 10-query dev set. Kept it.

## 2026-04-16 — the hyphenation bug

- Retrieval kept missing "economic" queries. Printed the raw PDF text and
  saw: *"eco-\nnomic"* — pdf reader kept the soft hyphen. Added
  `re.sub(r"-\n", "", text)` in `clean_text()`. +0.05 RQ overnight.
- Also noticed "LG - Public" stamp was polluting BM25 ('Public' hitting
  generic budget queries). Stripped it with `_HEADER_NOISE` regex.

## 2026-04-17 — comparative table (final numbers for the report)

Reran all three chunkers with the **final** cleaning pipeline. Results match
the table in `docs/chunking_analysis.md` — consistent across 3 re-runs.

## Anecdote: the outlier

One query — *"What was the outlook on cocoa in the 2025 budget?"* — always
failed on 500/80 but worked on 900/120. Root cause: the budget's cocoa
paragraph is 870 words long. I accepted the trade-off because cocoa queries
are rare in my target persona (a student asking about macro trends).
If it becomes common, I'd add **hierarchical chunking** — not in scope here.
