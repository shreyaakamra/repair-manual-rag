"""
rag_pipeline.py

Single entry point for the full RAG pipeline: retrieval (search.py) ->
generation (generate.py) -> formatted, cited answer. This is the file to
point at in a demo or interview - it shows the whole pipeline in one
readable place instead of scattered across scripts.

Usage:
    python rag_pipeline.py "how do I replace the battery" --category Phone
    python rag_pipeline.py "screen is cracked" --category Phone --k 3
"""

import argparse

from generate import generate_answer, GenerationError


def run(query: str, category: str | None = None, subject: str | None = None, k: int = 5):
    try:
        result = generate_answer(query, category=category, subject=subject, k=k)
    except GenerationError as e:
        print(f"[error] {e}")
        return None

    print(f"\nQ: {result['query']}")
    if category:
        print(f"(scoped to category: {category})")
    print(f"\n{result['answer']}\n")

    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['index']}] {s['guide_title']} (step {s['step_order']}) - {s['url']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Ask a repair question.")
    parser.add_argument("query", help="Natural-language repair question")
    parser.add_argument("--category", default=None, help="e.g. Phone, Appliance")
    parser.add_argument("--subject", default=None, help="e.g. Battery, Screen")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    run(args.query, category=args.category, subject=args.subject, k=args.k)


if __name__ == "__main__":
    main()
