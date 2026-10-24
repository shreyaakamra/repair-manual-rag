"""
config.py

NOTE on paths: this assumes config.py lives at the project root
(alongside download_myfixit.py, search.py, etc.), so BASE_DIR is just
this file's own directory.

NOTE on secrets: locally, values come from a .env file (via
python-dotenv). On Streamlit Community Cloud, there is no .env file -
secrets are set in the app's "Secrets" panel and exposed via
st.secrets, not os.environ. _get_secret() checks os.environ first
(local dev) and falls back to st.secrets (deployed), so the same
config.py works in both places without changes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default=None):
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        # Not running under Streamlit, or no secrets.toml configured -
        # fine for local CLI usage as long as .env covers it instead.
        return default


# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MYFIXIT_JSONS_DIR = DATA_DIR / "myfixit_jsons"  # put downloaded category .json files here
TESTS_DIR = BASE_DIR / "tests"

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Qdrant
# IMPORTANT if deploying to Streamlit Community Cloud: "localhost:6333"
# only works when Qdrant is running on the same machine (e.g. your laptop
# via docker run). Streamlit Cloud has no persistent local Qdrant - point
# QDRANT_URL at a hosted instance instead (Qdrant Cloud has a free tier:
# https://cloud.qdrant.io) and set QDRANT_API_KEY to match.
QDRANT_URL = _get_secret("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = _get_secret("QDRANT_API_KEY")
QDRANT_COLLECTION = "myfixit_manuals"

# Retrieval
RETRIEVAL_K = 5

# Generation (Groq, via the official groq SDK - see clients.py)
GROQ_API_KEY = _get_secret("GROQ_API_KEY")
# llama-3.3-70b-versatile was deprecated by Groq (decommissioned Aug 2026).
# openai/gpt-oss-120b is the current recommended general-purpose model;
# override via .env / Streamlit secrets if Groq's lineup changes again -
# check https://console.groq.com/docs/models for the current list.
GROQ_MODEL = _get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = 0  # low temperature: we want grounded, not creative, answers

# Which MyFixit category files to ingest (filenames from the repo's jsons/ folder)
# Full list of categories: Mac, Car and Truck, Household, Computer Hardware,
# Appliance, Camera, PC, Electronics, Phone, Game Console, Skills, Vehicle,
# Media Player, Apparel, Tablet
MYFIXIT_CATEGORIES = ["Appliance", "Phone"]  # start small, add more once validated

MYFIXIT_BASE_URL = "https://raw.githubusercontent.com/rub-ksv/MyFixit-Dataset/master/jsons"
