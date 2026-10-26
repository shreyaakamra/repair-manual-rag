"""
streamlit_app.py

FixMate - Streamlit front-end for the repair-manual RAG pipeline.
Deploy on Streamlit Community Cloud by pointing it at this file (keep it in
the project root, next to config.py, search.py and generate.py).

Secrets (Settings -> Secrets): GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY.
QDRANT_URL must be a hosted Qdrant instance (e.g. Qdrant Cloud); the
localhost default only exists on your own machine.

Only the look of the app lives here. Retrieval (search.py) and generation
(generate.py) are called exactly as before.
"""

from html import escape

import streamlit as st

from config import (
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    MYFIXIT_CATEGORIES,
    QDRANT_URL,
    RETRIEVAL_K,
)
from generate import generate_answer_from_chunks, GenerationError
from search import search

st.set_page_config(
    page_title="FixMate",
    page_icon="🛠️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Styling. HTML strings below are deliberately flush-left with no blank
# lines: Streamlit's markdown parser treats indented lines as code blocks and
# a blank line as the end of an HTML block.
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=DM+Sans:wght@400;500;600;700&display=swap');
:root { --ink: #1b1035; --violet: #7c5cff; --deep: #5b3fd9; --lilac: #e4daff; --mint: #c9f5e8; --peach: #ffe0ce; --pink: #ffd3e8; --lemon: #ffe45c; }
.stApp { background: linear-gradient(180deg, #efe9ff 0%, #f8f5ff 55%, #fbf9ff 100%); isolation: isolate; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main { background: transparent !important; }
header[data-testid="stHeader"] { background: transparent; }
.stApp::before { content: ""; position: fixed; right: -18vw; top: -18vh; width: 60vw; height: 60vh; background: radial-gradient(closest-side, rgba(124, 92, 255, 0.28), rgba(124, 92, 255, 0) 70%); filter: blur(30px); z-index: -1; pointer-events: none; }
.stApp::after { content: ""; position: fixed; left: -18vw; bottom: -18vh; width: 60vw; height: 60vh; background: radial-gradient(closest-side, rgba(43, 196, 166, 0.22), rgba(43, 196, 166, 0) 70%); filter: blur(30px); z-index: -1; pointer-events: none; }
@keyframes fmUp { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes fmPop { 0% { opacity: 0; transform: scale(0.94) translateY(10px); } 70% { transform: scale(1.015); } 100% { opacity: 1; transform: none; } }
@keyframes fmBob { from { transform: translateY(0) rotate(var(--r, 0deg)); } to { transform: translateY(-14px) rotate(calc(var(--r, 0deg) + 8deg)); } }
@keyframes fmWiggle { 0%, 100% { transform: rotate(var(--r, 0deg)); } 50% { transform: rotate(calc(var(--r, 0deg) + 2.5deg)) translateY(-2px); } }
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button { font-family: 'DM Sans', -apple-system, 'Segoe UI', sans-serif; }
.stApp, .stApp p, .stApp li { color: var(--ink); }
.block-container { max-width: 820px; padding-top: 2.4rem; padding-bottom: 3rem; position: relative; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
.fm-hero { position: relative; overflow: hidden; border: 2px solid var(--ink); border-radius: 32px; padding: 2rem 1.8rem 1.7rem 1.8rem; margin-bottom: 1.6rem; background: linear-gradient(135deg, #5b3fd9 0%, #7c5cff 55%, #9b7bff 100%); box-shadow: 8px 8px 0 var(--ink); animation: fmPop 0.7s both; }
.fm-float { position: absolute; font-size: 2.1rem; z-index: 0; pointer-events: none; opacity: 0.92; animation: fmBob 3.6s ease-in-out infinite alternate; }
.fm-f1 { right: 7%; top: 14%; --r: -12deg; }
.fm-f2 { right: 22%; top: 6%; --r: 10deg; animation-delay: 0.6s; font-size: 1.7rem; }
.fm-f3 { right: 11%; bottom: 20%; --r: 16deg; animation-delay: 1.1s; }
.fm-f4 { right: 28%; bottom: 14%; --r: -8deg; animation-delay: 1.6s; font-size: 1.6rem; }
.fm-conf { position: absolute; z-index: 0; pointer-events: none; width: 14px; height: 14px; animation: fmBob 4.4s ease-in-out infinite alternate; }
.fm-c1 { left: 52%; top: 12%; background: var(--lemon); border-radius: 3px; --r: 20deg; }
.fm-c2 { left: 60%; bottom: 18%; background: #2bc4a6; border-radius: 50%; animation-delay: 0.8s; }
.fm-c3 { right: 3%; top: 48%; background: #ff8a5b; border-radius: 3px; --r: -25deg; animation-delay: 1.3s; }
.fm-c4 { left: 46%; bottom: 10%; background: var(--pink); border-radius: 50%; animation-delay: 2s; }
.fm-tag { position: relative; z-index: 1; display: inline-block; padding: 5px 14px; border: 2px solid var(--ink); border-radius: 999px; background: #fff; color: var(--ink); font-size: 0.82rem; font-weight: 700; margin-bottom: 1rem; }
.fm-title { position: relative; z-index: 1; font-family: 'Bricolage Grotesque', 'Trebuchet MS', sans-serif; font-weight: 800; font-size: clamp(3rem, 13vw, 5rem); line-height: 1; letter-spacing: -0.03em; color: #fff; text-shadow: 3px 3px 0 var(--ink); }
.fm-title span { color: var(--lemon); }
.fm-lead { position: relative; z-index: 1; max-width: 560px; margin: 0.9rem 0 1.2rem 0; font-size: 1.05rem; font-weight: 500; line-height: 1.6; color: #fff; }
.fm-stickers { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 10px; }
.fm-sticker { display: inline-block; padding: 6px 14px; border: 2px solid var(--ink); border-radius: 14px; font-size: 0.86rem; font-weight: 700; color: var(--ink); box-shadow: 3px 3px 0 var(--ink); animation: fmWiggle 3s ease-in-out infinite; }
.fm-s1 { background: var(--mint); --r: -2deg; }
.fm-s2 { background: var(--lemon); --r: 1.5deg; animation-delay: 0.4s; }
.fm-s3 { background: var(--pink); --r: -1deg; animation-delay: 0.8s; }
.fm-label { font-family: 'Bricolage Grotesque', 'Trebuchet MS', sans-serif; font-weight: 800; font-size: 1.05rem; color: var(--ink); margin: 1.2rem 0 0.6rem 0; }
.fm-h2 { font-family: 'Bricolage Grotesque', 'Trebuchet MS', sans-serif; font-weight: 800; font-size: 1.7rem; letter-spacing: -0.02em; margin: 2.4rem 0 0.2rem 0; }
.fm-h2-sub { font-size: 0.98rem; opacity: 0.75; margin-bottom: 1rem; }
.fm-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.4rem 0 1rem 0; animation: fmUp 0.5s both; }
.fm-chip { display: inline-block; padding: 3px 12px; border: 2px solid var(--ink); border-radius: 999px; font-size: 0.8rem; font-weight: 700; color: var(--ink); background: var(--lemon); }
.fm-chip.mint { background: var(--mint); }
.fm-chip.lilac { background: var(--lilac); }
.fm-chip-row { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; }
.fm-answer-h { font-family: 'Bricolage Grotesque', 'Trebuchet MS', sans-serif; font-weight: 800; font-size: 1.5rem; letter-spacing: -0.02em; margin-bottom: 0.4rem; }
div[data-testid="stVerticalBlockBorderWrapper"]:has(.fm-answer-mark) { border: 2px solid var(--ink); border-radius: 24px; background: #fff; box-shadow: 6px 6px 0 #2bc4a6; animation: fmPop 0.55s both; }
.fm-src { display: flex; align-items: center; gap: 14px; padding: 13px 16px; margin-bottom: 12px; border: 2px solid var(--ink); border-radius: 18px; box-shadow: 3px 3px 0 var(--ink); text-decoration: none !important; color: var(--ink) !important; animation: fmPop 0.5s both; transition: transform 0.15s ease, box-shadow 0.15s ease; }
.fm-src:nth-child(5n+1) { background: var(--lilac); }
.fm-src:nth-child(5n+2) { background: var(--mint); animation-delay: 0.07s; }
.fm-src:nth-child(5n+3) { background: var(--peach); animation-delay: 0.14s; }
.fm-src:nth-child(5n+4) { background: var(--pink); animation-delay: 0.21s; }
.fm-src:nth-child(5n+5) { background: #fff3b0; animation-delay: 0.28s; }
a.fm-src:hover { transform: translate(-2px, -2px); box-shadow: 5px 5px 0 var(--ink); }
.fm-src-num { flex: 0 0 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 0.9rem; color: #fff; background: var(--ink); }
.fm-src-body { display: flex; flex-direction: column; flex: 1 1 auto; min-width: 0; }
.fm-src-title { font-weight: 700; line-height: 1.35; }
.fm-src-meta { font-size: 0.82rem; opacity: 0.7; }
.fm-src-go { font-size: 0.85rem; font-weight: 700; white-space: nowrap; text-decoration: underline; }
.fm-how { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; }
.fm-how-card { position: relative; padding: 16px 18px 18px 18px; border: 2px solid var(--ink); border-radius: 22px; box-shadow: 4px 4px 0 var(--ink); transition: transform 0.18s ease, box-shadow 0.18s ease; }
.fm-how-card:hover { transform: translate(-3px, -3px) rotate(-0.6deg); box-shadow: 7px 7px 0 var(--ink); }
.fm-how-card.h1 { background: var(--lilac); }
.fm-how-card.h2 { background: var(--mint); }
.fm-how-card.h3 { background: var(--peach); }
.fm-how-card.h4 { background: var(--pink); }
.fm-how-card.h5 { background: #fff3b0; }
.fm-how-card.h6 { background: #d7ecff; }
.fm-how-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.fm-how-num { width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; color: #fff; background: var(--ink); }
.fm-how-ico { font-size: 1.8rem; }
.fm-how-title { font-family: 'Bricolage Grotesque', 'Trebuchet MS', sans-serif; font-weight: 800; font-size: 1.15rem; margin-bottom: 4px; }
.fm-how-text { font-size: 0.92rem; line-height: 1.5; opacity: 0.85; }
.fm-note { margin-top: 1.2rem; padding: 14px 18px; border: 2px dashed var(--ink); border-radius: 18px; background: rgba(255, 255, 255, 0.6); font-size: 0.92rem; line-height: 1.6; }
.fm-foot { margin-top: 1.6rem; font-size: 0.82rem; line-height: 1.6; opacity: 0.65; text-align: center; }
div[data-testid="stForm"] { border: 2px solid var(--ink); border-radius: 28px; background: #fff; box-shadow: 6px 6px 0 var(--violet); padding: 1.4rem 1.5rem 1rem 1.5rem; }
div[data-testid="stWidgetLabel"] p { font-weight: 700; font-size: 0.95rem; color: var(--ink); }
div[data-baseweb="input"], div[data-baseweb="select"] > div { background: #fbf9ff !important; border: 2px solid var(--ink) !important; border-radius: 14px !important; }
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within { box-shadow: 3px 3px 0 var(--violet); }
.stFormSubmitButton > button { border-radius: 999px; border: 2px solid var(--ink) !important; padding: 0.6rem 2rem; font-weight: 700; font-size: 1rem; color: #fff !important; background: var(--ink) !important; box-shadow: 4px 4px 0 var(--violet); transition: transform 0.15s ease, box-shadow 0.15s ease; }
.stFormSubmitButton > button:hover { transform: translate(-2px, -2px); box-shadow: 6px 6px 0 var(--violet); }
.stFormSubmitButton > button:active { transform: translate(3px, 3px); box-shadow: 0 0 0 var(--violet); }
.stButton > button { border-radius: 999px !important; border: 2px solid var(--ink) !important; background: #fff !important; color: var(--ink) !important; font-weight: 600; padding: 0.4rem 1.1rem; box-shadow: 3px 3px 0 var(--ink); transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease; }
.stButton > button:hover { transform: translate(-2px, -2px); box-shadow: 5px 5px 0 var(--ink); background: var(--lemon) !important; }
.stButton > button:active { transform: translate(3px, 3px); box-shadow: 0 0 0 var(--ink); }
div[data-testid="stColumn"]:nth-child(1) .stButton > button, div[data-testid="column"]:nth-child(1) .stButton > button { background: var(--lilac) !important; }
div[data-testid="stColumn"]:nth-child(2) .stButton > button, div[data-testid="column"]:nth-child(2) .stButton > button { background: var(--mint) !important; }
div[data-testid="stColumn"]:nth-child(3) .stButton > button, div[data-testid="column"]:nth-child(3) .stButton > button { background: var(--peach) !important; }
details { border: 2px solid var(--ink) !important; border-radius: 16px !important; background: #fff; }
@media (max-width: 640px) {
div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.6rem !important; }
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"], div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.fm-float { opacity: 0.5; }
}
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    """<div class="fm-hero">
<span class="fm-float fm-f1">🔧</span><span class="fm-float fm-f2">⚙️</span><span class="fm-float fm-f3">🪛</span><span class="fm-float fm-f4">🔩</span>
<span class="fm-conf fm-c1"></span><span class="fm-conf fm-c2"></span><span class="fm-conf fm-c3"></span><span class="fm-conf fm-c4"></span>
<div class="fm-tag">✨ Repair help from real manuals</div>
<div class="fm-title">Fix<span>Mate</span></div>
<div class="fm-lead">Tell me what's broken. I look through real repair guides, pull out the steps that fit, and explain the fix with sources you can check.</div>
<div class="fm-stickers"><span class="fm-sticker fm-s1">📖 Real iFixit guides</span><span class="fm-sticker fm-s2">🔎 Hybrid search</span><span class="fm-sticker fm-s3">🔗 Cited answers</span></div>
</div>""",
    unsafe_allow_html=True,
)

missing = []
if not GROQ_API_KEY:
    missing.append("GROQ_API_KEY")
if QDRANT_URL.startswith("http://localhost"):
    missing.append("QDRANT_URL (still pointing at localhost - won't work when deployed)")
if missing:
    st.warning(
        "Missing/unconfigured for deployment: " + ", ".join(missing) +
        ". Set these in the app's Secrets panel. See streamlit_app.py's "
        "module docstring for details."
    )

EXAMPLES = [
    "washing machine won't drain",
    "iPhone SE battery replacement",
    "my phone no longer vibrates",
]
embedder_name = escape(str(EMBEDDING_MODEL_NAME).split("/")[-1])


def _use_example(text: str) -> None:
    st.session_state["query_text"] = text


def _safe_url(url) -> str:
    """Only allow http(s) links in the rendered source cards."""
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        return url
    return ""


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------
st.markdown('<div class="fm-label">Try one of these</div>', unsafe_allow_html=True)
example_cols = st.columns(len(EXAMPLES))
for i, (col, text) in enumerate(zip(example_cols, EXAMPLES)):
    col.button(text, key=f"example_{i}", on_click=_use_example, args=(text,))

st.markdown('<div style="height: 0.8rem"></div>', unsafe_allow_html=True)

with st.form("query_form"):
    category = st.selectbox("Device category (optional filter)", ["Any"] + MYFIXIT_CATEGORIES)
    query = st.text_input(
        "What's the problem?",
        placeholder="e.g. how do I replace the battery",
        key="query_text",
    )
    with st.expander("Options"):
        k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=RETRIEVAL_K)
        show_context = st.checkbox("Show retrieved source text", value=False)
    submitted = st.form_submit_button("Ask FixMate", type="primary")

if submitted and query.strip():
    selected_category = None if category == "Any" else category
    result = None
    gen_error = None
    chunks = []

    with st.status("FixMate is on it...", expanded=True) as status:
        st.write("🔎 Searching the repair guides (meaning + keywords)...")
        chunks = search(query, category=selected_category, k=k)
        n_guides = len({c.get("guide_id") for c in chunks})
        st.write(f"📚 Found {len(chunks)} repair step(s) from {n_guides} guide(s).")
        st.write("✍️ Writing an answer from those steps...")
        try:
            result = generate_answer_from_chunks(query, chunks)
            status.update(label="Done", state="complete", expanded=False)
        except GenerationError as e:
            gen_error = str(e)
            status.update(label="Something went wrong", state="error", expanded=True)

    if gen_error:
        st.error(gen_error)

    if show_context and chunks:
        with st.expander(f"Retrieved {len(chunks)} source chunk(s)", expanded=False):
            for i, c in enumerate(chunks, start=1):
                st.markdown(
                    f"**[{i}] {c['guide_title']}** (step {c['step_order']}, "
                    f"{c['category']})"
                )
                st.write(c["text"])
                st.divider()

    if result:
        st.markdown(
            f'<div class="fm-chips"><span class="fm-chip mint">⚡ hybrid retrieval</span>'
            f'<span class="fm-chip">📚 {len(chunks)} step(s) · {n_guides} guide(s)</span></div>',
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown(
                '<div class="fm-answer-h fm-answer-mark">🛠️ Your fix</div>',
                unsafe_allow_html=True,
            )
            st.write(result["answer"])

        if result["sources"]:
            st.markdown('<div class="fm-label">Where this came from</div>', unsafe_allow_html=True)
            cards = []
            for s in result["sources"]:
                url = _safe_url(s.get("url"))
                inner = (
                    f'<span class="fm-src-num">{escape(str(s["index"]))}</span>'
                    f'<span class="fm-src-body">'
                    f'<span class="fm-src-title">{escape(str(s["guide_title"]))}</span>'
                    f'<span class="fm-src-meta">Step {escape(str(s["step_order"]))}'
                    f' · {escape(str(s["category"]))}</span></span>'
                )
                if url:
                    cards.append(
                        f'<a class="fm-src" href="{escape(url, quote=True)}" '
                        f'target="_blank" rel="noopener noreferrer">{inner}'
                        f'<span class="fm-src-go">Open guide ↗</span></a>'
                    )
                else:
                    cards.append(f'<div class="fm-src">{inner}</div>')
            st.markdown('<div class="fm-srcs">' + "".join(cards) + "</div>", unsafe_allow_html=True)
        else:
            st.info("No matching repair steps were found for that query/category.")
elif submitted:
    st.info("Type a question first.")

# --------------------------------------------------------------------------
# How it works (pipeline)
# --------------------------------------------------------------------------
st.markdown(
    """<div class="fm-h2">How FixMate works</div>
<div class="fm-h2-sub">Six steps between your question and a cited answer.</div>""",
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="fm-how">
<div class="fm-how-card h1"><div class="fm-how-top"><span class="fm-how-num">1</span><span class="fm-how-ico">💬</span></div><div class="fm-how-title">You ask</div><div class="fm-how-text">A problem in plain language, with an optional device-category filter.</div></div>
<div class="fm-how-card h2"><div class="fm-how-top"><span class="fm-how-num">2</span><span class="fm-how-ico">🧮</span></div><div class="fm-how-title">Embed the question</div><div class="fm-how-text">Your question becomes a vector, so it can be matched by meaning.</div><div class="fm-chip-row"><span class="fm-chip lilac">{embedder_name}</span></div></div>
<div class="fm-how-card h3"><div class="fm-how-top"><span class="fm-how-num">3</span><span class="fm-how-ico">🔎</span></div><div class="fm-how-title">Hybrid retrieval</div><div class="fm-how-text">Vector search finds steps with similar meaning. BM25 finds steps that share your exact words.</div><div class="fm-chip-row"><span class="fm-chip lilac">Qdrant</span><span class="fm-chip lilac">BM25</span></div></div>
<div class="fm-how-card h4"><div class="fm-how-top"><span class="fm-how-num">4</span><span class="fm-how-ico">🧩</span></div><div class="fm-how-title">Fuse the rankings</div><div class="fm-how-text">Reciprocal rank fusion merges both lists and keeps the top repair steps.</div><div class="fm-chip-row"><span class="fm-chip lilac">RRF</span></div></div>
<div class="fm-how-card h5"><div class="fm-how-top"><span class="fm-how-num">5</span><span class="fm-how-ico">✍️</span></div><div class="fm-how-title">Write the answer</div><div class="fm-how-text">An LLM gets those steps and a prompt that forbids answering outside them.</div><div class="fm-chip-row"><span class="fm-chip lilac">Groq</span></div></div>
<div class="fm-how-card h6"><div class="fm-how-top"><span class="fm-how-num">6</span><span class="fm-how-ico">🔗</span></div><div class="fm-how-title">Cite the sources</div><div class="fm-how-text">The answer comes back with numbered citations that link to the original guides.</div></div>
</div>""",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="fm-note"><b>What it knows:</b> repair guides from the MyFixit dataset (iFixit). '
    f"Indexed categories: {escape(', '.join(MYFIXIT_CATEGORIES))}. It can only help with devices "
    f"that appear in those guides.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="fm-foot">FixMate writes its answers from retrieved guide steps and can still be '
    "incomplete or wrong. Check the linked guide before you start a repair, and disconnect "
    "power and batteries first.</div>",
    unsafe_allow_html=True,
)