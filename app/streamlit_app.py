"""
streamlit_app.py  —  the chat UI.
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Modernised build: gradient header, chat-style history, animated score
bars, source badges, and example-query chips. Functionality is identical
to the plain version — only presentation changed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from html import escape

import streamlit as st

# --- make `app` importable whether we run with  `streamlit run app/streamlit_app.py`
# or with `python -m streamlit run ...` from different working directories.
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings, RAW_DIR, PROCESSED, SAMPLE_DIR  # noqa: E402
from app.data_prep import build_all_chunks, load_chunks, save_chunks  # noqa: E402
from app.rag_pipeline import RAGPipeline  # noqa: E402
from app.innovation import record_feedback, feedback_summary  # noqa: E402


# -----------------------------------------------------------------------------
# page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Academic City RAG — CS4241",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# custom CSS — modern, polished look
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* base typography */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
}

/* hero header card */
.hero {
    background: linear-gradient(135deg, #1f3a93 0%, #6dd5ed 100%);
    color: white;
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(31, 58, 147, 0.18);
}
.hero h1 { color: white; margin: 0 0 6px 0; font-size: 2.1rem; font-weight: 700; }
.hero p  { color: rgba(255,255,255,0.92); margin: 0; font-size: 1.05rem; }

/* metric row */
.metric-card {
    background: #ffffff;
    border: 1px solid #e6e9f0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.06); }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #1f3a93; margin: 0; }
.metric-label { font-size: 0.78rem; color: #6b7280; text-transform: uppercase;
                letter-spacing: 0.08em; margin: 4px 0 0 0; }

/* answer card */
.answer-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-left: 4px solid #22c55e;
    padding: 18px 22px;
    border-radius: 10px;
    margin: 12px 0;
    font-size: 1.02rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.answer-card.abstain { border-left-color: #f59e0b; background: #fffbeb; }

/* retrieval chunk card */
.chunk-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-right: 8px;
}
.badge-election { background: #dbeafe; color: #1e40af; }
.badge-budget   { background: #fef3c7; color: #92400e; }
.badge-other    { background: #e5e7eb; color: #374151; }

/* score bar */
.score-bar-wrap {
    background: #e5e7eb;
    border-radius: 8px;
    height: 6px;
    width: 100%;
    margin-top: 6px;
    overflow: hidden;
}
.score-bar-fill {
    background: linear-gradient(90deg, #3b82f6, #22c55e);
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* sidebar polish */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    border-right: 1px solid #e6e9f0;
}
[data-testid="stSidebar"] h3 { color: #1f3a93; }

/* primary button override */
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #1f3a93, #3b82f6);
    border: none;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(31, 58, 147, 0.25);
}

/* example chip */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }

/* footer */
.footer {
    font-size: 0.82rem;
    color: #6b7280;
    text-align: center;
    padding: 18px 0;
    border-top: 1px solid #e6e9f0;
    margin-top: 32px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# session state
# -----------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {query, response}
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""


# -----------------------------------------------------------------------------
# sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎓 Academic City RAG")
    st.markdown("**Obour Adomako Tawiah**")
    st.markdown("CS4241 · 2026")
    st.markdown("Index: `10022200118`")
    st.divider()

    # Streamlit Cloud users put their key in Secrets — read it into env.
    # Locally, st.secrets raises if no secrets.toml exists, so guard it.
    try:
        for k in ("LLM_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
            if k in st.secrets:
                os.environ[k] = st.secrets[k]
                if not settings.llm_api_key:
                    settings.llm_api_key = st.secrets[k]
                if not settings.groq_api_key:
                    settings.groq_api_key = st.secrets[k]
    except Exception:
        pass

    st.markdown("#### ⚙️ Retrieval")
    top_k = st.slider("Top-K chunks", 3, 10, settings.top_k)
    prompt_variant = st.selectbox(
        "Prompt style",
        options=["guarded+cot", "guarded", "naive"],
        index=0,
        help="guarded+cot is the final production choice (lowest hallucination).",
    )
    show_prompt = st.checkbox("Show final prompt", value=False,
                              help="Reveal the exact system + user message.")
    st.divider()

    fb = feedback_summary()
    st.markdown("#### 💬 Feedback loop")
    st.metric("Events recorded", fb["total"],
              delta=f"+{fb['positive']} / -{fb['negative']}")

    st.divider()
    if st.session_state.history:
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# -----------------------------------------------------------------------------
# data loading (cached per session)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building RAG index (first run only)…")
def _build_pipeline() -> RAGPipeline:
    chunk_cache = PROCESSED / "chunks.jsonl"

    if chunk_cache.exists():
        chunks = load_chunks(chunk_cache)
    else:
        csv = RAW_DIR / "Ghana_Election_Result.csv"
        pdf = RAW_DIR / "2025-Budget-Statement.pdf"
        if not csv.exists():
            csv = SAMPLE_DIR / "Ghana_Election_Result_sample.csv"
        if not pdf.exists():
            pdf = None
        chunks = build_all_chunks(election_csv=csv, budget_pdf=pdf)
        save_chunks(chunks, chunk_cache)

    if not chunks:
        st.error("No data found. Please run `python scripts/download_data.py` "
                 "and then `python scripts/build_index.py`.")
        st.stop()
    return RAGPipeline(chunks, prompt_variant="guarded+cot")


pipeline = _build_pipeline()
total_chunks = len(pipeline.chunks) if hasattr(pipeline, "chunks") else 0


# -----------------------------------------------------------------------------
# hero header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🎓 Academic City RAG Assistant</h1>
  <p>Ask anything about the Ghana election results or the 2025 Budget
     Statement. Every answer is grounded in retrieved passages — no
     hallucinations, no guesswork.</p>
  <p style="font-size:0.75rem;opacity:0.6;margin-top:0.5rem;">build: retrieval-v3 · per-source</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# metrics row
# -----------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
      <p class="metric-value">{total_chunks:,}</p>
      <p class="metric-label">Chunks indexed</p>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
      <p class="metric-value">{len(st.session_state.history)}</p>
      <p class="metric-label">Questions asked</p>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
      <p class="metric-value">{fb['total']}</p>
      <p class="metric-label">Feedback events</p>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
      <p class="metric-value">{top_k}</p>
      <p class="metric-label">Top-K retrieved</p>
    </div>""", unsafe_allow_html=True)

st.write("")


# -----------------------------------------------------------------------------
# query input + example chips
# -----------------------------------------------------------------------------
EXAMPLES = [
    "How did the NDC perform in the Volta region?",
    "What does the 2025 budget say about inflation?",
    "Which region had the largest NPP margin?",
    "Who won the 2028 Ghana election?",
]

st.markdown("#### 🔍 Ask a question")
st.caption("Try an example or type your own:")
chip_cols = st.columns(len(EXAMPLES))
for col, ex in zip(chip_cols, EXAMPLES):
    if col.button(ex, key=f"chip-{ex}", use_container_width=True):
        st.session_state.pending_query = ex
        st.rerun()

query = st.text_input(
    "Your question",
    value=st.session_state.pending_query,
    placeholder="e.g., What did the 2025 budget say about inflation?",
    key="query_box",
    label_visibility="collapsed",
)
st.session_state.pending_query = ""

submit = st.button("✨  Ask", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _source_badge(source: str) -> str:
    s = (source or "").lower()
    if "election" in s:
        return '<span class="chunk-badge badge-election">Election CSV</span>'
    if "budget" in s or "pdf" in s:
        return '<span class="chunk-badge badge-budget">2025 Budget</span>'
    return f'<span class="chunk-badge badge-other">{escape(source or "unknown")}</span>'


def _render_response(entry):
    """Render a single (query, response) entry from history."""
    resp = entry["response"]
    q    = entry["query"]

    st.markdown(f"##### 💬 Question\n> {escape(q)}")

    # --- answer -------------------------------------------------------------
    if resp.abstained:
        st.markdown(
            f'<div class="answer-card abstain">⚠️ {escape(resp.answer)}'
            '<br><small>The system abstained to prevent hallucination.</small>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="answer-card">{escape(resp.answer)}</div>',
            unsafe_allow_html=True,
        )

    if resp.rewritten_query:
        st.info(f"🔁 Original retrieval was weak — rewrote query as: "
                f"`{resp.rewritten_query}`")

    # --- retrieved chunks ---------------------------------------------------
    hits = resp.retrieval.hits
    if hits:
        max_score = max((h.score for h in hits), default=1.0) or 1.0
        st.markdown(f"##### 📚 Retrieved context — top {len(hits)}")
        for h in hits:
            pct = min(100, int(round(100 * h.score / max_score)))
            header_html = (
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<b>[{h.rank + 1}]</b> {_source_badge(h.chunk.source)}'
                f'<span style="color:#6b7280;font-size:0.85rem;">'
                f'id={escape(h.chunk.chunk_id)} · score={h.score:.3f}</span>'
                f'</div>'
                f'<div class="score-bar-wrap"><div class="score-bar-fill" '
                f'style="width:{pct}%"></div></div>'
            )
            with st.expander(f"[{h.rank+1}]  {h.chunk.source}  ·  {h.score:.3f}"):
                st.markdown(header_html, unsafe_allow_html=True)
                st.write("")
                st.write(h.chunk.text)
                if h.chunk.metadata:
                    st.json(h.chunk.metadata, expanded=False)

    # --- final prompt -------------------------------------------------------
    if show_prompt and not resp.abstained:
        with st.expander("🧠 Final prompt sent to the LLM"):
            st.code(
                "=== SYSTEM ===\n" + resp.system_prompt +
                "\n\n=== USER ===\n" + resp.user_prompt,
                language="markdown",
            )

    # --- feedback loop (Part G) --------------------------------------------
    st.markdown("###### Was this answer helpful?")
    fc1, fc2, _ = st.columns([1, 1, 4])
    key_suffix = entry.get("id", "x")
    if fc1.button("👍  Helpful", key=f"up-{key_suffix}"):
        record_feedback(q, resp.used_chunk_ids, helpful=True)
        st.toast("Thanks — retrieval will learn from this ✔")
    if fc2.button("👎  Missed",  key=f"dn-{key_suffix}"):
        record_feedback(q, resp.used_chunk_ids, helpful=False)
        st.toast("Recorded — similar chunks will be down-weighted.")


# -----------------------------------------------------------------------------
# handle a new submission
# -----------------------------------------------------------------------------
if submit and query.strip():
    try:
        with st.spinner("Retrieving passages, fusing scores, generating answer…"):
            resp = pipeline.ask(query, top_k=top_k, prompt_variant=prompt_variant)
        st.session_state.history.append({
            "id": len(st.session_state.history),
            "query": query,
            "response": resp,
        })
    except RuntimeError as e:
        msg = str(e)
        if "rate-limited" in msg.lower() or "429" in msg:
            st.warning(
                "⏳ **The free LLM tier is busy right now.** "
                "OpenRouter allows 16 requests per minute and 50 per day "
                "on free models. Please wait 60 seconds and press **Ask** "
                "again — the system auto-retries with back-off so usually "
                "the second try succeeds."
            )
        else:
            st.error(f"Something went wrong: {msg}")


# -----------------------------------------------------------------------------
# render history (newest first)
# -----------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    for entry in reversed(st.session_state.history):
        _render_response(entry)
        st.divider()


# -----------------------------------------------------------------------------
# footer
# -----------------------------------------------------------------------------
st.markdown(
    '<div class="footer">'
    'Built from scratch — no LangChain/LlamaIndex. '
    'Embeddings: <b>MiniLM-L6-v2</b> · '
    'Retrieval: hybrid dense + BM25 RRF + domain re-ranker · '
    'LLM: hosted via OpenRouter.'
    '</div>',
    unsafe_allow_html=True,
)
