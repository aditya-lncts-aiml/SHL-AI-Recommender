from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(tags=["health"])

@router.get("/")
def root():
    return {
        "message": "SHL AI Assessment Recommendation API",
        "docs": "/docs",
        "health": "/health"
    }

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
