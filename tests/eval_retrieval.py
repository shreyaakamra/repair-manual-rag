"""
tests/eval_retrieval.py (tiered version)

Evaluates RETRIEVAL only (no generation, no API cost) against the tiered
tests/test_queries.json.

What each metric means:
- Guide hit@k: did a guide the query was labeled with show up in the top-k
  results? Used for 'easy' queries and any 'hard' query with guide IDs.
- MRR: 1 / (position of the first correct guide), averaged. 1.0 means the
  right guide was always first; 0.2 means it was typically fifth.
- Subject hit@k: for symptom queries labeled by part patterns (regexes such
  as 'vibrat' or 'charg|usb'), did any result's part name match? Looser than a guide hit:
  it proves the right PART was found, not the right guide.
- Category correctness: when a category was chosen, every result must be
  from that category. Empty results FAIL (an empty set must never pass).
- Off-domain: top score must stay below LOW_RELEVANCE_SCORE_THRESHOLD.
- 'ambiguous' and 'not_covered' queries are not scored here; they print
  what came back so you can read it (not_covered is judged at generation).

Run:
    python -m tests.eval_retrieval                  # dev split (default)
    python -m tests.eval_retrieval --split holdout  # only for final numbers
    python -m tests.eval_retrieval --split all
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from search import search
from config import QDRANT_COLLECTION, RETRIEVAL_K, TESTS_DIR

LOW_RELEVANCE_SCORE_THRESHOLD = 0.42
EASY_HIT_TARGET = 0.70  # acceptance criterion: 70% on easy queries


def load_queries(split: str) -> list[dict]:
    with open(TESTS_DIR / "test_queries.json", encoding="utf-8") as f:
        queries = json.load(f)
    if split == "all":
        return queries
    return [q for q in queries if q.get("split") == split]


def unique_guides_in_order(results: list[dict]) -> list:
    """Retrieval returns chunks; several can come from one guide."""
    seen = []
    for r in results:
        if r["guide_id"] not in seen:
            seen.append(r["guide_id"])
    return seen


def short(results: list[dict], n: int = 3) -> str:
    return "; ".join(
        f"{r['guide_title']} (id {r['guide_id']}, {r['score']:.2f})" for r in results[:n]
    )


def rate(values: list) -> str:
    return f"{sum(values)}/{len(values)} ({100 * sum(values) / len(values):.0f}%)"


def evaluate(split: str = "dev", k: int = RETRIEVAL_K) -> bool:
    queries = load_queries(split)
    metrics = defaultdict(lambda: defaultdict(list))  # tier -> metric -> values
    print(f"Collection: {QDRANT_COLLECTION}")
    print(f"Running {len(queries)} '{split}' queries (k={k})...\n")

    for q in queries:
        tier = q["tier"]
        results = search(q["query"], category=q.get("category"), k=k)
        top_score = results[0]["top_vector_score"] if results else 0.0
        tag = f"#{q['id']} [{tier}] '{q['query']}'"

        # category correctness: never let an empty result set pass
        if q.get("category"):
            ok = bool(results) and all(r["source_category"] == q["category"] for r in results)
            metrics[tier]["category"].append(ok)
            if not ok:
                print(f"[FAIL] {tag} -> category check failed ({len(results)} results)")

        if tier in ("easy", "hard") and q.get("expected_guide_ids"):
            guides = unique_guides_in_order(results)
            wanted = set(q["expected_guide_ids"])
            pos = next((i for i, g in enumerate(guides, 1) if g in wanted), None)
            hit = pos is not None
            metrics[tier]["guide_hit"].append(hit)
            metrics[tier]["rr"].append(1 / pos if pos else 0.0)
            print(f"[{'PASS' if hit else 'FAIL'}] {tag} -> first correct guide at rank {pos} "
                  f"(top score {top_score:.2f})")
            if not hit:
                print(f"       got: {short(results)}")

        elif tier == "hard" and (q.get("expected_subject_patterns") or q.get("expected_subjects")):
            if q.get("expected_subject_patterns"):
                pats = [re.compile(p, re.I) for p in q["expected_subject_patterns"]]
                hit = any(p.search(r.get("subject") or "") for p in pats for r in results)
            else:
                wanted = {s.lower() for s in q["expected_subjects"]}
                hit = any((r.get("subject") or "").lower() in wanted for r in results)
            metrics[tier]["subject_hit"].append(hit)
            print(f"[{'PASS' if hit else 'FAIL'}] {tag} -> right part found: {hit} "
                  f"(top score {top_score:.2f})")
            if not hit:
                print(f"       got subjects: {[r.get('subject') for r in results]}")

        elif tier == "off_domain":
            ok = top_score < LOW_RELEVANCE_SCORE_THRESHOLD
            metrics[tier]["low_relevance"].append(ok)
            print(f"[{'PASS' if ok else 'FAIL'}] {tag} -> top score {top_score:.3f} "
                  f"(threshold {LOW_RELEVANCE_SCORE_THRESHOLD})")

        else:  # ambiguous, not_covered: observe only
            print(f"[INFO] {tag} -> top score {top_score:.3f}")
            print(f"       got: {short(results)}")

    print("\n--- Summary by tier ---")
    for tier, m in metrics.items():
        print(f"\n{tier}")
        if m["guide_hit"]:
            print(f"  guide hit@{k}:  {rate(m['guide_hit'])}")
            print(f"  MRR:           {sum(m['rr']) / len(m['rr']):.2f}")
        if m["subject_hit"]:
            print(f"  subject hit@{k}: {rate(m['subject_hit'])}")
        if m["category"]:
            print(f"  category ok:   {rate(m['category'])}")
        if m["low_relevance"]:
            print(f"  low relevance: {rate(m['low_relevance'])}")

    easy = metrics["easy"]["guide_hit"]
    passed = True
    if easy:
        easy_rate = sum(easy) / len(easy)
        ok = easy_rate >= EASY_HIT_TARGET
        passed = passed and ok
        print(f"\nAcceptance: easy guide hit@{k} {easy_rate:.0%} vs target "
              f"{EASY_HIT_TARGET:.0%} -> {'PASS' if ok else 'FAIL'}")
    if any(not v for m in metrics.values() for v in m.get("category", [])):
        passed = False
        print("Acceptance: category correctness -> FAIL")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    args = parser.parse_args()
    sys.exit(0 if evaluate(args.split) else 1)
    