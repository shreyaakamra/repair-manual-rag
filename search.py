"""
search.py

Filtered semantic search over the MyFixit-indexed Qdrant collection.
Filter by category (e.g. "Phone") and/or subject (e.g. "Battery") to scope
retrieval.

Returns plain dicts (not raw Qdrant ScoredPoint objects) so downstream
consumers - generate.py, eval scripts - don't need to know anything about
the Qdrant client.
"""

from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_COLLECTION, RETRIEVAL_K
from clients import get_embedder, get_qdrant_client


def search(
    query: str,
    category: str | None = None,
    subject: str | None = None,
    k: int = RETRIEVAL_K,
) -> list[dict]:
    embedder = get_embedder()
    client = get_qdrant_client()

    query_vector = embedder.encode(query).tolist()

    conditions = []
    if category:
        conditions.append(FieldCondition(key="source_category", match=MatchValue(value=category)))
    if subject:
        conditions.append(FieldCondition(key="subject", match=MatchValue(value=subject)))

    query_filter = Filter(must=conditions) if conditions else None

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=k,
    )

    return [
        {
            "score": r.score,
            "text": r.payload["text"],
            "guide_title": r.payload["guide_title"],
            "guide_id": r.payload["guide_id"],
            "category": r.payload["category"],
            "source_category": r.payload["source_category"],
            "subject": r.payload.get("subject"),
            "step_order": r.payload["step_order"],
            "url": r.payload["url"],
        }
        for r in response.points
    ]


def print_results(results: list[dict]):
    for r in results:
        print(
            f"\n[score={r['score']:.3f}] {r['guide_title']} "
            f"(step {r['step_order']}, category: {r['category']})"
        )
        print(f"  {r['text'][:200]}...")
        print(f"  source: {r['url']}")


if __name__ == "__main__":
    results = search("how do I replace the battery", category="Phone")
    print_results(results)
