"""
search.py

Semantic search over the MyFixit-indexed Qdrant collection, with an optional
hybrid mode that fuses it with BM25 keyword search (reciprocal rank fusion).
Filter by category (e.g. "Phone") and/or subject (e.g. "Battery") to scope
retrieval.

Returns plain dicts (not raw Qdrant ScoredPoint objects) so downstream
consumers - generate.py, eval scripts - don't need to know anything about
the Qdrant client.

hybrid=True (default): vector + BM25, merged with reciprocal rank fusion.
hybrid=False: vector search only.

Score fields on every result:
  score            vector-only: cosine score. hybrid: RRF score (about 0.01-0.033).
  vector_score     this chunk's cosine score (None if only BM25 found it).
  top_vector_score best cosine score for the whole query. Use THIS for any
                   "is the query relevant at all" threshold - it means the same
                   thing in both modes.
"""

from collections import defaultdict

from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import QDRANT_COLLECTION, RETRIEVAL_K
from clients import get_embedder, get_qdrant_client

# How many candidates each method contributes before fusion.
CANDIDATES_PER_METHOD = 20
# BM25 is not filtered inside the index, so over-fetch and filter afterwards.
BM25_POOL = 100


def _to_result(point_id, payload, score, vector_score=None) -> dict:
    return {
        "id": point_id,
        "score": score,
        "vector_score": vector_score,
        "top_vector_score": None,
        "text": payload["text"],
        "guide_title": payload["guide_title"],
        "guide_id": payload["guide_id"],
        "category": payload["category"],
        "source_category": payload["source_category"],
        "subject": payload.get("subject"),
        "step_order": payload["step_order"],
        "url": payload["url"],
    }


def _vector_search(query, category, subject, k) -> list[dict]:
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
    results = [_to_result(r.id, r.payload, r.score, r.score) for r in response.points]
    top = results[0]["vector_score"] if results else 0.0
    for r in results:
        r["top_vector_score"] = top
    return results


def _keyword_search(query, category, subject, known: dict, k) -> list[dict]:
    """BM25 hits, filtered by category/subject. `known` = results we already have by id."""
    from bm25_search import bm25_search  # imported here so hybrid=False never needs bm25s

    hits = bm25_search(query, k=BM25_POOL)  # [(id, score)], best first
    if not hits:
        return []

    missing = [i for i, _ in hits if i not in known]
    payloads = {}
    if missing:
        client = get_qdrant_client()
        for rec in client.retrieve(
            collection_name=QDRANT_COLLECTION,
            ids=missing,
            with_payload=True,
            with_vectors=False,
        ):
            payloads[rec.id] = rec.payload

    out = []
    for i, bm_score in hits:
        if i in known:
            r = known[i]
        elif i in payloads:
            r = _to_result(i, payloads[i], bm_score)
        else:
            continue
        if category and r["source_category"] != category:
            continue
        if subject and r["subject"] != subject:
            continue
        out.append(r)
        if len(out) >= k:
            break
    return out


def search(
    query: str,
    category: str | None = None,
    subject: str | None = None,
    k: int = RETRIEVAL_K,
    hybrid: bool = True,
) -> list[dict]:
    if not hybrid:
        return _vector_search(query, category, subject, k)

    from bm25_search import rrf_fuse

    vec = _vector_search(query, category, subject, CANDIDATES_PER_METHOD)
    known = {r["id"]: r for r in vec}
    kw = _keyword_search(query, category, subject, known, CANDIDATES_PER_METHOD)
    for r in kw:
        known.setdefault(r["id"], r)

    top_vector = vec[0]["vector_score"] if vec else 0.0
    fused = rrf_fuse([[r["id"] for r in vec], [r["id"] for r in kw]], k=k)
    results = []
    for cid, rrf_score in fused:
        r = dict(known[cid])
        r["score"] = rrf_score
        r["top_vector_score"] = top_vector
        results.append(r)
    return results


def print_results(results: list[dict]):
    for r in results:
        print(
            f"\n[score={r['score']:.3f}] {r['guide_title']} "
            f"(step {r['step_order']}, category: {r['category']})"
        )
        print(f"  {r['text'][:200]}...")
        print(f"  source: {r['url']}")


def pick_best_guide(chunks: list[dict]):
    """Guide with the most chunks in the results; ties go to the higher score."""
    stats = defaultdict(lambda: [0, 0.0])  # guide_id -> [chunk count, best score]
    for c in chunks:
        s = stats[c["guide_id"]]
        s[0] += 1
        s[1] = max(s[1], c["score"])
    return max(stats, key=lambda g: (stats[g][0], stats[g][1]))


def fetch_guide_steps(guide_id: int) -> list[dict]:
    """All steps of one guide, in order, as dicts shaped like search results."""
    client = get_qdrant_client()
    flt = Filter(must=[FieldCondition(key="guide_id", match=MatchValue(value=guide_id))])
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=flt,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points += batch
        if offset is None:
            break
    steps = [_to_result(p.id, p.payload, None) for p in points]
    return sorted(steps, key=lambda s: s["step_order"])


if __name__ == "__main__":
    results = search("how do I replace the battery", category="Phone")
    print_results(results)