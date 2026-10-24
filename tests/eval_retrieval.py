"""
tests/eval_retrieval.py

Evaluates the RETRIEVAL layer only (no generation, no API cost) against
tests/test_queries.json. Two metrics, both automatable without a
human-labeled ground-truth answer set:

1. Category-filter accuracy: every result returned when a category filter
   is applied must actually belong to that category. This should be 100%
   by construction (Qdrant enforces the filter) - it's here as a
   regression guard, not a discovery tool. If this ever fails, the filter
   logic in search.py broke.

2. Keyword-recall@k: for each query, do ANY of its expected_keywords
   (domain terms a correct answer should surface) appear in the text of
   the top-k retrieved chunks? This is a proxy for "did we retrieve
   something topically relevant," not a substitute for a real labeled
   relevance judgment - see the README's honesty about this in mind.
   Queries with expect_low_relevance=true are scored differently: they
   pass if the top score is BELOW a low-relevance threshold, since the
   point of that query is that nothing good should match.

Run:
    python tests/eval_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import search
from config import RETRIEVAL_K, TESTS_DIR

LOW_RELEVANCE_SCORE_THRESHOLD = 0.42  # empirically, off-domain queries can score up to ~0.41 due to
                                       # generic procedural-language overlap (e.g. "remove/lift a small
                                       # part" language shared between a car timing belt and a phone
                                       # ribbon-cable latch) - see decisions.md


def load_test_queries():
    with open(TESTS_DIR / "test_queries.json", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    queries = load_test_queries()

    category_checks = 0
    category_passes = 0
    keyword_checks = 0
    keyword_passes = 0
    low_relevance_checks = 0
    low_relevance_passes = 0

    print(f"Running {len(queries)} retrieval eval queries (k={RETRIEVAL_K})...\n")

    for q in queries:
        results = search(q["query"], category=q.get("category"), k=RETRIEVAL_K)
        top_score = results[0]["score"] if results else 0.0

        # --- category-filter correctness ---
        if q.get("category"):
            category_checks += 1
            ok = all(r["source_category"] == q["category"] for r in results)
            category_passes += int(ok)
            status = "PASS" if ok else "FAIL"
            if not ok:
                print(f"[{status}] #{q['id']} category filter leaked a wrong category!")

        # --- low-relevance sanity check ---
        if q.get("expect_low_relevance"):
            low_relevance_checks += 1
            ok = top_score < LOW_RELEVANCE_SCORE_THRESHOLD
            low_relevance_passes += int(ok)
            status = "PASS" if ok else "FAIL"
            print(
                f"[{status}] #{q['id']} '{q['query']}' -> top score {top_score:.3f} "
                f"({'below' if ok else 'ABOVE'} threshold {LOW_RELEVANCE_SCORE_THRESHOLD})"
            )
            continue

        # --- keyword recall ---
        if q.get("expected_keywords"):
            keyword_checks += 1
            combined_text = " ".join(r["text"].lower() for r in results)
            hit = any(kw.lower() in combined_text for kw in q["expected_keywords"])
            keyword_passes += int(hit)
            status = "PASS" if hit else "FAIL"
            print(
                f"[{status}] #{q['id']} '{q['query']}' -> top score {top_score:.3f}, "
                f"keyword hit: {hit}"
            )

    print("\n--- Summary ---")
    if category_checks:
        print(f"Category-filter accuracy: {category_passes}/{category_checks} "
              f"({100 * category_passes / category_checks:.0f}%)")
    if keyword_checks:
        print(f"Keyword-recall@{RETRIEVAL_K}:      {keyword_passes}/{keyword_checks} "
              f"({100 * keyword_passes / keyword_checks:.0f}%)")
    if low_relevance_checks:
        print(f"Low-relevance sanity check: {low_relevance_passes}/{low_relevance_checks} "
              f"({100 * low_relevance_passes / low_relevance_checks:.0f}%)")

    total_checks = category_checks + keyword_checks + low_relevance_checks
    total_passes = category_passes + keyword_passes + low_relevance_passes
    all_pass = total_passes == total_checks
    print(f"\nOverall: {total_passes}/{total_checks} checks passed.")
    return all_pass


if __name__ == "__main__":
    success = evaluate()
    sys.exit(0 if success else 1)
