"""Keyword (BM25) search + rank fusion for the repair-manual RAG.

Reads index/chunks.jsonl.gz (written by build_bm25_corpus.py), builds a BM25
index in memory on first use (a few seconds for ~47k chunks), and offers:

    bm25_search(query, k, category=None) -> [(chunk_id, score), ...]
    rrf_fuse([ids_from_vector, ids_from_bm25], k=5) -> [(chunk_id, rrf_score), ...]
"""
import gzip
import json
from pathlib import Path

import bm25s
import numpy as np

CORPUS_PATH = Path(__file__).parent / "index" / "chunks.jsonl.gz"
ID_KEY = "id"              # <- change if your jsonl uses another key name
TEXT_KEY = "text"          # <- change if your jsonl uses another key name
CATEGORY_KEY = "category"  # <- only used when you pass category=...

_state = {}


def _load():
    if _state:
        return _state
    ids, texts, cats = [], [], []
    with gzip.open(CORPUS_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if ID_KEY not in row or TEXT_KEY not in row:
                raise KeyError(
                    f"Expected keys '{ID_KEY}' and '{TEXT_KEY}' but the file has "
                    f"{list(row.keys())}. Edit ID_KEY / TEXT_KEY at the top of this file."
                )
            ids.append(row[ID_KEY])
            texts.append(row[TEXT_KEY])
            cats.append(row.get(CATEGORY_KEY))
    tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    _state.update(ids=ids, cats=np.array(cats, dtype=object), retriever=retriever)
    return _state


def bm25_search(query, k=5, category=None):
    s = _load()
    q_tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
    weights = None
    if category is not None:
        weights = (s["cats"] == category).astype(np.float32)  # 0 for other categories
    n = len(s["ids"])
    docs, scores = s["retriever"].retrieve(
        q_tokens, k=min(k, n), weight_mask=weights, show_progress=False
    )
    out = []
    for i, sc in zip(docs[0], scores[0]):
        if sc > 0:  # zero score = no query word matched at all
            out.append((s["ids"][i], float(sc)))
    return out


def rrf_fuse(ranked_lists, k=5, c=60):
    """Reciprocal Rank Fusion. ranked_lists = [[id, id, ...], [id, ...]], best first."""
    scores = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (c + rank)
    return sorted(scores.items(), key=lambda x: -x[1])[:k]