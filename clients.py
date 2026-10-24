"""
clients.py

Single source of truth for the embedding model and Qdrant client.
Both are expensive to construct (model load, connection handshake), so
every other module (ingest, search, generate, eval scripts) should pull
them from here instead of instantiating their own copies.

Using functools.lru_cache instead of module-level globals so the objects
are created lazily on first use, not at import time (helps startup time
for scripts that only need one of the two, e.g. download_myfixit.py).
"""

from functools import lru_cache

from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)
