# SHL Assessment Recommendation Agent

FastAPI conversational RAG service that recommends only SHL assessments from `app/data/catalog.json`.

## Features

- `GET /health` returns `{"status":"ok"}`
- `POST /chat` returns a stable schema with `reply`, `recommendations`, and `end_of_conversation`
- Clarification, refinement, comparison, off-topic refusal, and prompt-injection refusal
- Catalog-grounded responses only; recommendations are emitted from `catalog.json`
- SentenceTransformers embeddings with FAISS indexing and deterministic fallback embeddings for offline/test environments
- Gemini 2.5 Flash integration through `GEMINI_API_KEY`
- Playwright and BeautifulSoup scraper for rebuilding the SHL catalog

## Run Locally

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Build The Catalog And Index

```powershell
playwright install chromium
python -m app.services.scraper
python scripts/build_index.py
```

The scraper writes `app/data/catalog.json`. The index builder writes `app/vectorstore/faiss.index` when FAISS is available.

## API

```json
{
  "messages": [
    {"role": "user", "content": "Recommend SHL assessments for a software developer role."}
  ]
}
```

## Tests

```powershell
pytest
```

## Docker

```powershell
docker build -t shl-ai-recommender .
docker run --env-file .env -p 8000:8000 shl-ai-recommender
```

## Railway Deployment

1. Push the repository to GitHub.
2. Create a new Railway project from the GitHub repo.
3. Add `GEMINI_API_KEY` as a Railway variable.
4. Railway can use the included Dockerfile automatically.
5. Set the health check path to `/health`.

## Approach

The service treats `catalog.json` as the source of truth. User messages pass through guardrails, intent detection, optional clarification, semantic retrieval, optional Gemini generation, and response validation. Even when Gemini is enabled, the API recommendation objects are built from retrieved catalog records, preventing invented assessment names or URLs.
