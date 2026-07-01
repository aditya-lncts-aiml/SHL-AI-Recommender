from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import chat, health
from app.services.embeddings import get_embedding_service


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading embedding model...")
    get_embedding_service()._load_model()
    logging.info("Embedding model loaded.")
    yield


app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    version="1.0.0",
    description="Conversational RAG API for recommending SHL assessments.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)