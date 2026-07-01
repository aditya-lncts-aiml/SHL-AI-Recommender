from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.agent import RecommendationAgent


logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])
agent = RecommendationAgent()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return agent.respond(request)
    except ValueError as exc:
        logger.warning("Invalid chat request: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="Unable to process chat request") from exc
