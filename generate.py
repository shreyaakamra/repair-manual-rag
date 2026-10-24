"""
generate.py

The piece the README flagged as "not yet built": takes retrieved chunks
and turns them into an actual natural-language answer, grounded in those
chunks, via Groq's chat completions API (official groq SDK, client
cached in clients.py).

Design choices:
- System prompt explicitly forbids answering from outside the provided
  context, and requires the model to say so if the context doesn't cover
  the question - this is the main lever against hallucination in a RAG
  system, so it's worth being explicit about it rather than trusting the
  model's default behavior.
- Sources are numbered [1], [2]... in the prompt and the model is asked
  to cite them inline, so an answer can be checked against its sources
  step by step - this also makes the eval script's job easier.
- generate_answer_from_chunks() is split out from generate_answer() so
  callers that already have chunks (the Streamlit app displaying
  retrieved context, the eval script computing groundedness) don't have
  to re-run retrieval just to get the same chunks a second time.
"""

from clients import get_groq_client
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE
from search import search
import re

_FULLWIDTH_CITATION = re.compile(r"【(\d+)(?:†[^】]*)?】")


def normalize_citations(text: str) -> str:
    """Convert 【2】 or 【2†L0-L2】 style markers into plain [2]."""
    return _FULLWIDTH_CITATION.sub(r"[\1]", text)

SYSTEM_PROMPT = """You are a repair assistant. Answer the user's repair question \
using ONLY the numbered source steps provided below. Follow these rules strictly:

1. Only use information present in the sources. Do not add steps, tools, or \
warnings that are not stated in the sources.
2. Cite the source number(s) you used inline, immediately after each claim, \
using ONLY plain square brackets with a number and no other characters \
inside them - like [1] or [1][2]. Do not use any other bracket style, \
footnote markers, superscripts, or symbols (for example, never write \
【1】, [1†L2], or (1)).
3. If the sources do not contain enough information to answer the question, \
say so plainly instead of guessing.
4. Keep the answer concise and in the order the repair steps should be \
performed.
"""


class GenerationError(Exception):
    pass


def _build_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] (Guide: {c['guide_title']} | Category: {c['category']} | "
            f"Step {c['step_order']})\n{c['text']}"
        )
    return "\n\n".join(lines)


def generate_answer_from_chunks(query: str, chunks: list[dict]) -> dict:
    """
    Generation only - assumes retrieval already happened. Use this when
    you already have `chunks` (e.g. the Streamlit app wants to display
    the retrieved context alongside the answer, or the eval script needs
    the raw chunk text to check groundedness) and don't want to search
    twice.
    """
    if not GROQ_API_KEY:
        raise GenerationError(
            "GROQ_API_KEY is not set. Add it to your .env file locally, or to "
            "this app's Secrets panel if deployed on Streamlit Community Cloud."
        )

    if not chunks:
        return {"query": query, "answer": "I couldn't find any matching repair steps.", "sources": []}

    context = _build_context(chunks)
    user_prompt = f"Question: {query}\n\nSources:\n{context}"

    client = get_groq_client()
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        raise GenerationError(f"Groq API call failed: {e}") from e

    answer_text = normalize_citations(completion.choices[0].message.content.strip())

    return {
        "query": query,
        "answer": answer_text,
        "sources": [
            {
                "index": i + 1,
                "guide_title": c["guide_title"],
                "url": c["url"],
                "category": c["category"],
                "step_order": c["step_order"],
            }
            for i, c in enumerate(chunks)
        ],
    }


def generate_answer(
    query: str,
    category: str | None = None,
    subject: str | None = None,
    k: int = 5,
) -> dict:
    """Runs retrieval, then generation. See generate_answer_from_chunks()
    if you already have chunks and want to skip the search call."""
    chunks = search(query, category=category, subject=subject, k=k)
    if not chunks:
        return {
            "query": query,
            "answer": (
                "I couldn't find any repair steps matching that question"
                + (f" in category '{category}'." if category else ".")
            ),
            "sources": [],
        }
    return generate_answer_from_chunks(query, chunks)


if __name__ == "__main__":
    result = generate_answer("how do I replace the battery", category="Phone")
    print(f"\nQ: {result['query']}\n")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['index']}] {s['guide_title']} (step {s['step_order']}) - {s['url']}")
