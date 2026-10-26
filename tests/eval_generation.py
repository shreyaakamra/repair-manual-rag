"""
tests/eval_generation.py (tiered version)

Evaluates the GENERATION layer end-to-end (retrieval + Groq call) against
the tiered tests/test_queries.json. Costs real API calls.

What is scored depends on each query's expected_behavior:

- "answer" (easy, hard): the model should answer from the sources.
    * abstained     - it declined although an answer was expected. Reported
                      separately, NOT counted as a citation failure.
    * citation      - the answer has at least one [n] marker.
    * valid citation- every [n] points at a real source (1..number of sources).
    * cites right guide - (only when the query has labeled guide IDs) at least
                      one cited source belongs to an expected guide.
    * groundedness  - 1-5 from a second model call acting as a fact-checker.
- "abstain" (not_covered, off_domain): the model should decline. Answering
  with cited steps counts as a failure (a possible hallucination).
- "observe" (ambiguous): not scored; the answer is printed for you to read.

Every summary line starts with RESULT so you can filter runs easily.

Run:
    python -m tests.eval_generation                  # dev split (default)
    python -m tests.eval_generation --split holdout  # final numbers only
    python -m tests.eval_generation --mode guide     # Experiment 2: whole guide
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clients import get_groq_client
from generate import generate_answer_from_chunks, GenerationError
from search import search, pick_best_guide, fetch_guide_steps
from config import GROQ_API_KEY, GROQ_MODEL, QDRANT_COLLECTION, TESTS_DIR

CITATION_PATTERN = re.compile(r"\[(\d+)\]")
ABSTAIN_PATTERN = re.compile(
    r"(don't have|do not have|cannot find|can't find|couldn't find|no information|"
    r"not (?:enough|sufficient) information|unable to answer|"
    r"(?:do|does)(?: not|n't) (?:contain|include|cover)|"
    r"only (?:state|mention|contain|say))",
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


def _retry_wait(message: str, attempt: int) -> float:
    """Use the wait time the API suggests ("try again in 772ms"), else back off."""
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", message)
    if m:
        seconds = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
        return seconds + 1.0
    return min(2 ** attempt, 30)


def with_rate_limit_retry(fn, *args, max_tries: int = 8, **kwargs):
    """Call fn; if Groq answers 429 (tokens-per-minute limit), wait and retry."""
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            limited = "429" in msg or "rate limit" in msg.lower()
            if not limited or attempt == max_tries:
                raise
            wait = _retry_wait(msg, attempt)
            print(f"  [rate limit] waiting {wait:.1f}s (attempt {attempt}/{max_tries})...")
            time.sleep(wait)


def load_queries(split: str) -> list[dict]:
    with open(TESTS_DIR / "test_queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    if split == "all":
        return queries
    return [q for q in queries if q.get("split") == split]


def judge_groundedness(answer: str, source_texts: list[str]) -> dict:
    context = "\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(source_texts))
    prompt = f"Answer to check:\n{answer}\n\nSource text it should be grounded in:\n{context}"

    client = get_groq_client()
    try:
        completion = with_rate_limit_retry(
            client.chat.completions.create,
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        return {"score": None, "reason": f"judge call failed: {str(e)[:120]}"}
    raw = completion.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "reason": f"judge returned unparseable output: {raw[:200]}"}


def frac(values: list) -> str:
    if not values:
        return "n/a"
    return f"{sum(values)}/{len(values)}"


def evaluate(split: str = "dev", mode: str = "chunks") -> bool:
    if not GROQ_API_KEY:
        print("[skip] GROQ_API_KEY not set - can't run generation eval.")
        return False

    queries = load_queries(split)
    stats = defaultdict(lambda: defaultdict(list))  # tier -> metric -> values
    print(f"Collection: {QDRANT_COLLECTION}")
    print(f"Mode: {mode}")
    errors = []
    abstained_ids = defaultdict(list)
    answered_wrongly = defaultdict(list)
    print(f"Running {len(queries)} '{split}' queries...")

    for q in queries:
        tier = q["tier"]
        behavior = q.get("expected_behavior", "answer")
        print(f"\n--- #{q['id']} [{tier}] {q['query']} ---")

        chunks = search(q["query"], category=q.get("category"))
        if mode == "guide" and chunks:
            # Experiment 2: use the search results only to pick the best guide,
            # then give the model ALL of that guide's steps, in order.
            picked = pick_best_guide(chunks)
            chunks = fetch_guide_steps(picked)
            if chunks:
                print(f"[guide mode] picked guide {picked}: "
                      f"{chunks[0]['guide_title']} ({len(chunks)} steps)")
        try:
            result = with_rate_limit_retry(generate_answer_from_chunks, q["query"], chunks)
        except GenerationError as e:
            print(f"[error] {str(e)[:200]}")
            errors.append(q["id"])
            continue

        answer = result["answer"]
        print(answer)

        cited = [int(n) for n in CITATION_PATTERN.findall(answer)]
        abstained = bool(ABSTAIN_PATTERN.search(answer)) and not cited

        if behavior == "abstain":
            stats[tier]["correct_abstain"].append(abstained)
            if not abstained:
                answered_wrongly[tier].append(q["id"])
            print(f"[{'PASS' if abstained else 'FAIL'}] "
                  f"{'abstained' if abstained else 'ANSWERED when it should abstain'}")
            continue

        if behavior == "observe":
            stats[tier]["abstained_observed"].append(abstained)
            print(f"[INFO] {'abstained' if abstained else 'answered'} (not scored - read it)")
            continue

        # behavior == "answer"
        stats[tier]["abstained"].append(abstained)
        if abstained:
            abstained_ids[tier].append(q["id"])
            print("[ABSTAIN] declined although an answer was expected "
                  "(counted separately, not as a citation failure)")
            continue

        has_citation = bool(cited)
        stats[tier]["citation"].append(has_citation)
        valid = has_citation and all(1 <= n <= len(chunks) for n in cited)
        stats[tier]["valid_citation"].append(valid)
        distinct = {chunks[n - 1]["guide_id"] for n in set(cited) if 1 <= n <= len(chunks)}
        stats[tier]["distinct_guides"].append(len(distinct))
        print(f"[{'PASS' if has_citation else 'FAIL'}] citation present: {has_citation}"
              f" | all citations valid: {valid}")

        if q.get("expected_guide_ids"):
            wanted = set(q["expected_guide_ids"])
            right = any(
                chunks[n - 1]["guide_id"] in wanted
                for n in set(cited)
                if 1 <= n <= len(chunks)
            )
            stats[tier]["cites_right_guide"].append(right)
            print(f"[{'PASS' if right else 'FAIL'}] cites an expected guide: {right}")

        judgment = judge_groundedness(answer, [c["text"] for c in chunks])
        score = judgment.get("score")
        print(f"Groundedness score: {score} - {judgment.get('reason')}")
        if isinstance(score, (int, float)):
            stats[tier]["groundedness"].append(score)

    print("\n--- Summary ---")
    print(f"RESULT mode: {mode}")
    if errors:
        print(f"RESULT WARNING: run INCOMPLETE, queries skipped: {errors} - do not compare")
    else:
        print("RESULT errors: 0 (all queries ran)")
    all_scores = []
    for tier, m in stats.items():
        print(f"RESULT [{tier}]")
        if "abstained" in m:
            print(f"RESULT   abstained on answerable queries: "
                  f"{sum(m['abstained'])}/{len(m['abstained'])}")
            if abstained_ids[tier]:
                print(f"RESULT   abstained ids: {abstained_ids[tier]}")
        if m["citation"]:
            print(f"RESULT   citation present (answered): {frac(m['citation'])}")
            print(f"RESULT   all citations valid:         {frac(m['valid_citation'])}")
            print(f"RESULT   distinct guides cited (mean): "
                  f"{sum(m['distinct_guides']) / len(m['distinct_guides']):.1f}")
        if m["cites_right_guide"]:
            print(f"RESULT   cites an expected guide:     {frac(m['cites_right_guide'])}")
        if m["groundedness"]:
            avg = sum(m["groundedness"]) / len(m["groundedness"])
            all_scores += m["groundedness"]
            print(f"RESULT   mean groundedness:           {avg:.1f}/5 (n={len(m['groundedness'])})")
        if "correct_abstain" in m:
            print(f"RESULT   correct abstention:          {frac(m['correct_abstain'])}")
            if answered_wrongly[tier]:
                print(f"RESULT   answered but should abstain: {answered_wrongly[tier]}")
        if "abstained_observed" in m:
            print(f"RESULT   abstained (observed only):   "
                  f"{sum(m['abstained_observed'])}/{len(m['abstained_observed'])}")
    if all_scores:
        print(f"RESULT [overall] mean groundedness: "
              f"{sum(all_scores) / len(all_scores):.1f}/5 (n={len(all_scores)})")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--mode", choices=["chunks", "guide"], default="chunks",
                        help="chunks = top-5 steps (baseline); guide = whole best guide")
    args = parser.parse_args()
    evaluate(args.split, args.mode)