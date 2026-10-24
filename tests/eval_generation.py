"""
tests/eval_generation.py

Evaluates the GENERATION layer end-to-end (retrieval + Groq call) against
tests/test_queries.json. Costs real API calls, so keep the query set
small and intentional rather than exhaustive.

Two checks per answer:

1. Citation presence: does the answer actually contain inline citation
   markers like [1]? If not, the model ignored the system prompt's
   citation instruction, which is worth catching regardless of whether
   the content itself is accurate.

2. Groundedness (LLM-as-judge): a second Groq call is given the answer
   and the exact source chunks it was built from, and asked to score,
   1-5, whether every claim in the answer is actually supported by those
   sources. This is the standard "faithfulness" check used in RAG
   evaluation - it does NOT check whether the answer is a *good* repair
   procedure, only whether it's honest about what its sources say.

For the deliberately off-domain query (expect_low_relevance in the test
set), the check instead looks for the model admitting it can't answer,
since that's the correct behavior when nothing relevant was retrieved.

Run:
    python tests/eval_generation.py
"""

import json
import re
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients import get_groq_client
from generate import generate_answer_from_chunks, GenerationError
from search import search
from config import GROQ_API_KEY, GROQ_MODEL, TESTS_DIR

CITATION_PATTERN = re.compile(r"\[\d+\]")
ADMITS_NO_ANSWER_PATTERN = re.compile(
    r"\b(don't have|do not have|cannot find|couldn't find|no information|"
    r"not (?:enough|sufficient) information|unable to answer|"
    r"(?:do|does) not contain|(?:do|does)n't contain)\b",
    re.IGNORECASE,
)
ABSTAIN_PATTERN = re.compile(
    r"(do(?:es)? not (?:contain|include)|don't (?:contain|include)|"
    r"only (?:state|mention|contain)|not enough information|"
    r"couldn't find|cannot find|no information)",
    re.IGNORECASE,
)

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given an \
answer and the source text it was supposed to be based on. Score, from 1 to 5, \
whether EVERY claim in the answer is directly supported by the source text:

5 = every claim is directly supported
3 = mostly supported, minor unsupported detail
1 = contains claims not found in the sources at all

Respond with ONLY a JSON object: {"score": <int>, "reason": "<one sentence>"}
"""


def load_test_queries():
    with open(TESTS_DIR / "test_queries.json", encoding="utf-8") as f:
        return json.load(f)


def judge_groundedness(answer: str, source_texts: list[str]) -> dict:
    context = "\n\n".join(f"[{i+1}] {text}" for i, text in enumerate(source_texts))
    prompt = f"Answer to check:\n{answer}\n\nSource text it should be grounded in:\n{context}"

    client = get_groq_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = completion.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "reason": f"judge returned unparseable output: {raw[:200]}"}


def evaluate():
    if not GROQ_API_KEY:
        print("[skip] GROQ_API_KEY not set - can't run generation eval. "
              "Set it in .env (local) or Secrets (Streamlit Cloud) to run this.")
        return False

    queries = load_test_queries()
    citation_checks = 0
    citation_passes = 0
    groundedness_scores = []
    refusal_checks = 0
    abstained_count = 0
    refusal_passes = 0

    for q in queries:
        if not q.get("expected_keywords") and not q.get("expect_low_relevance"):
            continue

        print(f"\n--- #{q['id']}: {q['query']} ---")
        chunks = search(q["query"], category=q.get("category"))

        try:
            result = generate_answer_from_chunks(q["query"], chunks)
        except GenerationError as e:
            print(f"[error] {e}")
            continue

        print(result["answer"])

        if q.get("expect_low_relevance"):
            refusal_checks += 1
            admits = bool(ADMITS_NO_ANSWER_PATTERN.search(result["answer"]))
            refusal_passes += int(admits)
            print(f"[{'PASS' if admits else 'FAIL'}] model {'admitted' if admits else 'did NOT admit'} "
                  f"it lacks relevant info")
            continue
        if ABSTAIN_PATTERN.search(result["answer"]) and not CITATION_PATTERN.search(result["answer"]):
            abstained_count += 1
            print("[ABSTAIN] sources judged insufficient - excluded from citation/groundedness scoring")
            continue
        citation_checks += 1
        has_citation = bool(CITATION_PATTERN.search(result["answer"]))
        citation_passes += int(has_citation)
        print(f"[{'PASS' if has_citation else 'FAIL'}] citation present: {has_citation}")

        source_texts = [c["text"] for c in chunks]
        judgment = judge_groundedness(result["answer"], source_texts)
        print(f"Groundedness score: {judgment.get('score')} - {judgment.get('reason')}")
        if judgment.get("score") is not None:
            groundedness_scores.append(judgment["score"])

    print("\n--- Summary ---")
    if citation_checks:
        print(f"Citation presence: {citation_passes}/{citation_checks}")
        
    if refusal_checks:
        print(f"Correct refusal on off-domain query: {refusal_passes}/{refusal_checks}")
    if groundedness_scores:
        avg = sum(groundedness_scores) / len(groundedness_scores)
        print(f"Average groundedness score: {avg:.1f}/5 (n={len(groundedness_scores)})")
        print(f"Abstained on in-domain queries (insufficient context): {abstained_count}/{len(queries) - refusal_checks}")
    return True


if __name__ == "__main__":
    evaluate()
