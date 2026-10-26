# Repair Manual RAG (MyFixit)

A retrieval-augmented generation pipeline that answers natural-language repair
questions ("how do I replace my phone's battery?"), scoped to the correct
device category, sourced from real iFixit repair guides via the
[MyFixit dataset](https://github.com/rub-ksv/MyFixit-Dataset).

## What changed since the retrieval-only version

- **Generation is now implemented** (`generate.py`) — retrieved chunks are
  passed to Groq (via the official `groq` SDK) with a system prompt that
  forbids answering outside the provided context and requires inline
  citations.
- **A Streamlit UI** (`streamlit_app.py`) — deployable to Streamlit
  Community Cloud. See the Deployment section below.
- **Two known bugs fixed**: `config.py` now defines the constants
  `download_myfixit.py` needs, and `myfixit_processing.py` has a working
  `process_category_file`. The Qdrant collection is `myfixit_manuals`, not
  the old `samsung_manuals` leftover name.
- **Fixed a path bug**: `PROJECT_ROOT = Path(__file__).parent.parent` pointed
  one directory too high if `config.py` sits at the repo root — replaced with
  `BASE_DIR = Path(__file__).resolve().parent`.
- **Shared client factories** (`clients.py`) — the embedding model and
  Qdrant client are now constructed once and reused, instead of every
  script/function creating its own.
- **Schema validation** — `myfixit_processing.py` now warns (once) if the
  dataset's actual JSON keys don't match what the parser expects, instead of
  silently producing empty chunks. Run `inspect_schema.py` after downloading
  to eyeball the raw keys yourself before trusting a full ingest.
- **An actual eval suite** (`tests/` + `run_eval.py`) — see below.

## Architecture

```
download_myfixit.py ──► data/myfixit_jsons/<Category>.json
        │
        ▼
myfixit_processing.py ──► Chunk objects (one per repair step)
        │
        ▼
ingest_myfixit_to_qdrant.py ──► embeds (all-MiniLM-L6-v2) + upserts to Qdrant
        │
        ▼
search.py ──► embeds query, applies category/subject filter, returns top-k
        │
        ▼
generate.py ──► builds numbered-source context, calls Groq, returns
                cited, grounded answer
        │
        ▼
rag_pipeline.py ──► CLI entry point tying retrieval + generation together
streamlit_app.py ──► web UI, same retrieval + generation calls
```

`clients.py` is the shared dependency both `ingest_myfixit_to_qdrant.py` and
`search.py`/`generate.py` pull the embedding model and Qdrant client from.

## Setup

```bash
pip install -r requirements.txt
docker run -p 6333:6333 qdrant/qdrant   # run Qdrant locally
cp .env.example .env                     # then fill in GROQ_API_KEY
```

Edit `config.py` to set `MYFIXIT_CATEGORIES` (defaults to `["Appliance", "Phone"]`
— start small, add more once validated).

```bash
python download_myfixit.py       # pulls category JSON files
python inspect_schema.py         # sanity-check field names before trusting the parser
python ingest_myfixit_to_qdrant.py
python rag_pipeline.py "how do I replace the battery" --category Phone
```

Run the web UI locally with:

```bash
streamlit run streamlit_app.py
```

## Deploying to Streamlit Community Cloud

Streamlit Cloud can't run a local Docker container, so `localhost:6333`
won't exist there. Two things need to change from local dev:

1. **Use a hosted Qdrant instance** instead of `docker run qdrant/qdrant`.
   [Qdrant Cloud](https://cloud.qdrant.io) has a free tier that's enough for
   this project's scale. Create a cluster, then run
   `ingest_myfixit_to_qdrant.py` **locally, once**, pointed at that cluster's
   URL/API key (set them in your local `.env` temporarily) so the remote
   collection is populated before you deploy.
2. **Set secrets in the Streamlit dashboard**, not a `.env` file — Streamlit
   Cloud doesn't read `.env`. Go to your app's Settings → Secrets and add:

   ```toml
   GROQ_API_KEY = "gsk_..."
   QDRANT_URL = "https://xxxx-xxxx.cloud.qdrant.io"
   QDRANT_API_KEY = "..."
   ```

   `config.py` checks environment variables first (local dev) and falls
   back to `st.secrets` automatically, so no code changes are needed
   between the two environments.

Then on [share.streamlit.io](https://share.streamlit.io), point a new app at
this repo with `streamlit_app.py` as the entry point. `requirements.txt`
already includes `streamlit`, so no extra setup is needed there.

One resource note: `sentence-transformers` pulls in `torch`, which is a
fairly heavy dependency for Streamlit Cloud's free-tier resources. It fits,
but expect a slower first load while the embedding model downloads and the
container spins up.

## Evaluation

```bash
python run_eval.py                  # retrieval + generation eval
python run_eval.py --retrieval-only # no API calls, just retrieval metrics
```

- **Retrieval eval** (`tests/eval_retrieval.py`, free): category-filter
  correctness (a regression guard — Qdrant's filter should make this 100% by
  construction) and keyword-recall@k against `tests/test_queries.json`, plus
  a sanity check that a deliberately off-domain query scores low rather than
  matching confidently.
- **Generation eval** (`tests/eval_generation.py`, costs Groq calls):
  checks that answers include inline citations, that the off-domain query
  gets an honest "can't answer" instead of a hallucinated one, and scores
  **groundedness** (does every claim trace back to a retrieved source?)
  using a second LLM call as judge.

### Honest limitations of this eval

This is not a rigorous, published benchmark. `expected_keywords` in the test
set are hand-picked domain terms, not human relevance judgments, so
keyword-recall is a proxy for topical relevance, not a precision/recall
number you'd cite in a paper. The groundedness judge is itself an LLM and can
be wrong. Treat this as "enough structure to catch regressions and demonstrate
methodology," not a claim of formally validated accuracy — worth saying
plainly if asked in an interview.

## Remaining limitations

- No reranking (single-stage cosine similarity over bi-encoder embeddings).
- The Streamlit app is a single-user demo UI, not a production service —
  no auth, no rate limiting, no persistence of past queries.
- Only the categories listed in `MYFIXIT_CATEGORIES` are indexed.
- Dataset field names (`Guidid`, `Text_raw`, `Order`, `StepId`) are based on
  the MyFixit README/paper and validated once at runtime via a schema check —
  confirm with `inspect_schema.py` if the dataset repo has changed format.

## Citation

Nabizadeh, N., Kolossa, D., & Heckmann, M. (2020). MyFixit: An Annotated
Dataset, Annotation Tool, and Baseline Methods for Information Extraction
from Repair Manuals. *Proceedings of The 12th Language Resources and
Evaluation Conference (LREC 2020)*, 2120-2128.
