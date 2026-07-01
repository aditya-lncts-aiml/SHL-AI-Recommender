# SHL Assessment Recommendation Agent

A FastAPI-based conversational AI service that recommends **only SHL assessments** from a curated SHL catalog. The application uses a lightweight hybrid retrieval pipeline combining **TF-IDF**, **BM25**, **RapidFuzz**, and rule-based ranking to provide grounded recommendations.

---

## Features

- Conversational assessment recommendations
- Hybrid retrieval using:
  - TF-IDF semantic search
  - BM25 keyword ranking
  - RapidFuzz fuzzy matching
  - Rule-based score boosting
- Assessment comparison
- Clarification for ambiguous requests
- Prompt injection and off-topic guardrails
- Grounded responses generated only from the local SHL catalog
- Gemini 2.5 Flash integration for natural language responses
- Playwright + BeautifulSoup scraper for rebuilding the SHL assessment catalog
- Fast and lightweight deployment with no transformer or FAISS dependencies

---

## Project Structure

```
app/
├── data/
│   └── catalog.json
├── routers/
├── services/
│   ├── agent.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── comparison.py
│   ├── guardrails.py
│   └── scraper.py
└── main.py
```

---

## Run Locally

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

copy .env.example .env      # Windows
# cp .env.example .env      # Linux/macOS

uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## Environment Variables

Create a `.env` file.

```env
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-2.5-flash
```

---

## Build the SHL Catalog

Install Playwright:

```bash
playwright install chromium
```

Rebuild the catalog:

```bash
python -m app.services.scraper
```

The scraper updates:

```
app/data/catalog.json
```

No vector index generation is required.

---

## API

### Health Check

```
GET /health
```

Response

```json
{
  "status": "ok"
}
```

---

### Chat Endpoint

```
POST /chat
```

Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Recommend SHL assessments for a Java developer."
    }
  ]
}
```

Example Response

```json
{
  "reply": "I found these SHL assessments from the catalog...",
  "recommendations": [
    {
      "name": "SHL Coding Interview",
      "url": "...",
      "test_type": "Skills"
    }
  ],
  "end_of_conversation": false
}
```

---

## Evaluation

Run the evaluation suite:

```bash
python -m evaluation.evaluate
```

Metrics include:

- Precision@10
- Recall@10
- Hit Rate@10
- Mean Reciprocal Rank (MRR)
- Groundedness
- Hallucination Rate
- Clarification Accuracy
- Comparison Accuracy
- Average Latency

---

## Tests

```bash
pytest
```

---

## Docker

Build the image:

```bash
docker build -t shl-ai-recommender .
```

Run the container:

```bash
docker run --env-file .env -p 8000:8000 shl-ai-recommender
```

---

## Deployment

The application can be deployed on platforms such as:

- Render
- Railway
- AWS EC2
- Docker-compatible cloud providers

Required environment variables:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`

Set the health check endpoint to:

```
/health
```

---

## Retrieval Pipeline

The recommendation engine uses a lightweight hybrid retrieval strategy.

```
User Query
      │
      ▼
TF-IDF Similarity
      │
      ▼
BM25 Ranking
      │
      ▼
RapidFuzz Matching
      │
      ▼
Rule-based Score Boosting
      │
      ▼
Top SHL Assessments
      │
      ▼
Gemini (Optional)
      │
      ▼
Grounded Response
```

Unlike embedding-based RAG systems, this implementation does **not** require transformer models or vector databases, making it faster, lighter, and suitable for free-tier deployments.

---

## Tech Stack

- Python 3.11
- FastAPI
- scikit-learn
- BM25 (rank-bm25)
- RapidFuzz
- Google Gemini API
- Playwright
- BeautifulSoup4
- Docker

---

## License

This project is intended for educational and assessment purposes.