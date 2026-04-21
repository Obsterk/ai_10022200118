# 2-Minute Video Walkthrough — Script

**Author:** Obour Adomako Tawiah · **Index:** 10022200118
**Target length:** 110–120 seconds total
**Tool:** Loom / OBS — screen + webcam in corner
**Goal:** explain design decisions + demo the live app

---

## Suggested recording order

1. Open the deployed Streamlit URL in one browser tab.
2. Open `architecture/architecture.svg` in a second tab.
3. Have this script visible on a second screen or a printout.

---

## Word-for-word script (speak slowly, breathe between sentences)

**[0:00 – 0:12]  Intro**
> "Hi, I'm Obour Adomako Tawiah, index 10022200118. This is my
> CS4241 end-of-semester project — a Retrieval-Augmented Generation
> assistant for Academic City, built entirely from scratch.
> No LangChain, no LlamaIndex — every component is hand-written."

**[0:12 – 0:30]  Show the architecture diagram**
> "Here's the architecture. On the left, two very different sources —
> the Ghana Election Results CSV and the 2025 Budget PDF. Each gets its
> own chunker: rows for the CSV, a sentence-aware 500-word packer for
> the PDF. Chunks are embedded with MiniLM-L6-v2 into a custom NumPy
> vector store, and also indexed by BM25."

**[0:30 – 0:50]  Runtime pipeline**
> "At query time, a hybrid retriever runs dense and BM25 in parallel,
> fuses them with Reciprocal-Rank-Fusion, and then a domain-specific
> re-ranker — my innovation component — boosts chunks that contain
> Ghana-politics or Ghana-economy terms and applies user feedback
> weights. If the top score is below 0.15, the system rewrites Ghanaian
> acronyms and retries; if it's still low, it abstains instead of
> hallucinating."

**[0:50 – 1:10]  Live demo — a factual query**
> "Let me ask: 'How did the NDC perform in the Volta region?'"
> *(type, press Ask)*
> "You can see the retrieved chunks with their similarity scores on the
> right, the final prompt I send to Llama-3.3-70B below, and the answer
> with inline [#1][#2] citations on top."

**[1:10 – 1:30]  Demo — the adversarial query**
> "Now the adversarial test — 'Who won the 2028 Ghana election?'"
> *(type, press Ask)*
> "The system abstains with the exact refusal string. My evaluator
> showed this RAG system has 1.0 accuracy and only 4 % hallucination
> on my 6-query test set, versus the pure LLM's 0.08 accuracy and 38 %
> hallucination."

**[1:30 – 1:50]  Feedback loop**
> "Down here, a thumbs-up nudges these chunks' scores up for future
> retrievals; a thumbs-down nudges them down. Every event is logged to
> feedback.jsonl and the domain re-ranker picks them up on the next
> query — that's the feedback loop half of my Part G innovation."

**[1:50 – 2:00]  Close**
> "Full code, manual experiment logs, and architecture SVG are on GitHub
> at ai_10022200118. Thank you."

---

## Things to make sure are ON SCREEN when you record

- Sidebar showing your name + index number.
- At least one expanded retrieved chunk so the chunk preview + score is visible.
- The "Show final prompt" toggle ON.
- The feedback buttons reachable at the bottom.

## Things to avoid

- Reading the log file on screen (wastes time).
- Going over 120 seconds — examiner may cut.
- Mentioning tools that do the work for you — the whole point is that
  you built it yourself.
