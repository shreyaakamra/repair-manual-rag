"""
streamlit_app.py

Streamlit front-end for the repair-manual RAG pipeline. Deploy this on
Streamlit Community Cloud by pointing it at this file as the app's entry
point.

BEFORE DEPLOYING, two things won't work with local-machine defaults:

1. Qdrant: "http://localhost:6333" only exists on your own machine. On
   Streamlit Cloud you need a reachable Qdrant instance - the free tier
   at https://cloud.qdrant.io is the easiest path. Set QDRANT_URL and
   QDRANT_API_KEY in the app's Secrets panel (Settings -> Secrets) to
   point at it, and re-run ingest_myfixit_to_qdrant.py once locally
   against that same remote cluster so it has data before you deploy.

"""

import streamlit as st

from config import MYFIXIT_CATEGORIES, GROQ_API_KEY, QDRANT_URL, RETRIEVAL_K
from generate import generate_answer_from_chunks, GenerationError
from search import search

st.set_page_config(page_title="Repair Manual Assistant", page_icon="🔧", layout="centered")

st.title("🔧 Repair Manual Assistant")
st.caption(
    "Ask a repair question and get an answer grounded in real iFixit repair "
    "guides (MyFixit dataset), with numbered source citations."
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

with st.sidebar:
    st.header("About this pipeline")
    st.write(
        "Retrieval: query embedded with all-MiniLM-L6-v2, filtered semantic "
        "search over Qdrant.\n\n"
        "Generation: retrieved steps passed to Groq with a system prompt "
        "that forbids answering outside the provided context."
    )
    st.write(f"**Indexed categories:** {', '.join(MYFIXIT_CATEGORIES)}")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=RETRIEVAL_K)
    show_context = st.checkbox("Show retrieved source text", value=False)

with st.form("query_form"):
    category = st.selectbox("Device category (optional filter)", ["Any"] + MYFIXIT_CATEGORIES)
    query = st.text_input("What's the problem?", placeholder="e.g. how do I replace the battery")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and query.strip():
    selected_category = None if category == "Any" else category

    with st.spinner("Retrieving relevant repair steps..."):
        chunks = search(query, category=selected_category, k=k)

    if show_context and chunks:
        with st.expander(f"Retrieved {len(chunks)} source chunk(s)", expanded=False):
            for i, c in enumerate(chunks, start=1):
                st.markdown(
                    f"**[{i}] {c['guide_title']}** (step {c['step_order']}, "
                    f"{c['category']}, score {c['score']:.3f})"
                )
                st.write(c["text"])
                st.divider()

    with st.spinner("Generating a grounded answer..."):
        try:
            result = generate_answer_from_chunks(query, chunks)
        except GenerationError as e:
            st.error(str(e))
            result = None

    if result:
        st.markdown("### Answer")
        st.write(result["answer"])

        if result["sources"]:
            st.markdown("### Sources")
            for s in result["sources"]:
                st.markdown(
                    f"**[{s['index']}]** {s['guide_title']} "
                    f"(step {s['step_order']}, {s['category']}) - "
                    f"[source]({s['url']})"
                )
        else:
            st.info("No matching repair steps were found for that query/category.")
elif submitted:
    st.info("Type a question first.")
