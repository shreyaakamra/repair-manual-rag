"""
run_eval.py

Runs the full evaluation suite: retrieval metrics (free, no API calls)
then generation metrics (costs Groq API calls, skipped automatically if
GROQ_API_KEY isn't set).

Usage:
    python run_eval.py                 # both suites
    python run_eval.py --retrieval-only
"""

import argparse
import sys

sys.path.insert(0, ".")

from tests.eval_retrieval import evaluate as eval_retrieval
from tests.eval_generation import evaluate as eval_generation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("RETRIEVAL EVAL")
    print("=" * 60)
    eval_retrieval()

    if not args.retrieval_only:
        print("\n" + "=" * 60)
        print("GENERATION EVAL")
        print("=" * 60)
        eval_generation()


if __name__ == "__main__":
    main()
