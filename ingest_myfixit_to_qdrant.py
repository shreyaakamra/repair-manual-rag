"""
ingest_myfixit_to_qdrant.py

Reads downloaded MyFixit category JSON files, turns each repair step into
a chunk, embeds it, and upserts into Qdrant with metadata (category,
subject, guide title, step order, source URL).
"""

import uuid
INGEST_NAMESPACE = uuid.UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
import time

def upsert_with_retry(client, collection_name, points, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 2 ** attempt  # 2s, 4s, 8s, 16s, 32s
            print(f"  upsert failed ({e.__class__.__name__}), retrying in {wait}s "
                  f"(attempt {attempt}/{max_retries})...")
            time.sleep(wait)

from qdrant_client.models import Distance, VectorParams, PointStruct

from config import (
    MYFIXIT_CATEGORIES,
    MYFIXIT_JSONS_DIR,
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
)
from clients import get_embedder, get_qdrant_client
from myfixit_processing import process_category_file


def ensure_collection(client, recreate: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if recreate and QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        existing.remove(QDRANT_COLLECTION)
        print(f"Dropped existing collection '{QDRANT_COLLECTION}' (--recreate)")

        if QDRANT_COLLECTION not in existing:
            client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name="source_category", field_schema="keyword")
        client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name="category", field_schema="keyword")
        client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name="subject", field_schema="keyword")
        print(f"Created Qdrant collection '{QDRANT_COLLECTION}' with payload indexes")
    else:
        print(f"Using existing Qdrant collection '{QDRANT_COLLECTION}'")


def ingest(recreate: bool = False) -> int:
    """Runs the full ingest and returns the total number of chunks indexed."""
    embedder = get_embedder()
    client = get_qdrant_client()
    ensure_collection(client, recreate=recreate)

    total_chunks = 0
    for category in MYFIXIT_CATEGORIES:
        json_path = MYFIXIT_JSONS_DIR / f"{category}.json"
        if not json_path.exists():
            print(f"[skip] {json_path} not found - run download_myfixit.py first")
            continue

        print(f"\nProcessing category: {category}")
        chunks = process_category_file(json_path, source_category=category)
        print(f"  {len(chunks)} steps extracted")

        # Embed and upsert in batches to keep memory/requests reasonable
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            vectors = embedder.encode(texts, show_progress_bar=False)

            points = [
                PointStruct(
                            id=str(uuid.uuid5(INGEST_NAMESPACE, f"{chunk.category}-{chunk.guide_id}-{chunk.step_id}")),
                    vector=vector.tolist(),
                                        payload={
                        "text": chunk.text,
                        "guide_title": chunk.guide_title,
                        "guide_id": chunk.guide_id,
                        "category": chunk.category,
                        "subject": chunk.subject,
                        "source_category": chunk.source_category,
                        "step_order": chunk.step_order,
                        "step_id": chunk.step_id,
                        "url": chunk.url,
                    },
                )
                for chunk, vector in zip(batch, vectors)
            ]
            upsert_with_retry(client, QDRANT_COLLECTION, points)
            total_chunks += len(points)
            print(f"  upserted batch {i // batch_size + 1} ({len(points)} chunks)")

    print(f"\nDone. {total_chunks} total chunks indexed in '{QDRANT_COLLECTION}'.")
    return total_chunks


if __name__ == "__main__":
    import sys

    ingest(recreate="--recreate" in sys.argv)
