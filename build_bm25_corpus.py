"""
build_bm25_corpus.py - export the text BM25 will search, one line per chunk.

Why a file? The deployed app cannot see your local data/ folder (it is not in
GitHub), so BM25 reads this small compressed file instead. Each line holds the
chunk's Qdrant point ID, its category, and the SAME enriched text that was
embedded (device + part + guide title + step), so keyword search and vector
search look at identical text.

Run once, and again whenever the embedded text or the data changes:
    python build_bm25_corpus.py

Writes index/chunks.jsonl.gz (commit this file).
"""

import gzip
import json
import uuid
from pathlib import Path

from config import MYFIXIT_CATEGORIES, MYFIXIT_JSONS_DIR
from myfixit_processing import process_category_file
from ingest_myfixit_to_qdrant import INGEST_NAMESPACE, build_embedding_text

OUT = Path(__file__).resolve().parent / "index" / "chunks.jsonl.gz"
OUT.parent.mkdir(exist_ok=True)

seen = set()
written = 0
with gzip.open(OUT, "wt", encoding="utf-8") as f:
    for category in MYFIXIT_CATEGORIES:
        path = MYFIXIT_JSONS_DIR / f"{category}.json"
        for c in process_category_file(path, source_category=category):
            # must be exactly the ID formula used at ingest time
            pid = str(uuid.uuid5(INGEST_NAMESPACE, f"{c.category}-{c.guide_id}-{c.step_id}"))
            if pid in seen:
                continue
            seen.add(pid)
            f.write(json.dumps(
                {"id": pid, "source_category": category, "text": build_embedding_text(c)},
                ensure_ascii=False,
            ) + "\n")
            written += 1

print(f"wrote {written} chunks to {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")