from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import numpy as np


logger = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_DIMENSION = 384


class EmbeddingService:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: object | None = None
        self._model_attempted = False

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, FALLBACK_DIMENSION), dtype=np.float32)

        model = self._load_model()
        if model is not None:
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vectors, dtype=np.float32)

        return np.vstack([self._hash_embedding(text) for text in texts]).astype(np.float32)
    
    def _load_model(self) :
        return None

    # def _load_model(self) -> object | None:
    #     if self._model is not None:
    #         return self._model
    #     if self._model_attempted:
    #         return None
    #     self._model_attempted = True
    #     try:
    #         from sentence_transformers import SentenceTransformer

    #         self._model = SentenceTransformer(self.model_name)
    #         return self._model
    #     except Exception as exc:
    #         logger.warning("SentenceTransformer unavailable, using deterministic fallback embeddings: %s", exc)
    #         return None

    @staticmethod
    @lru_cache(maxsize=4096)
    def _hash_embedding(text: str) -> np.ndarray:
        vector = np.zeros(FALLBACK_DIMENSION, dtype=np.float32)
        tokens = [token for token in text.lower().replace("/", " ").replace("-", " ").split() if token]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % FALLBACK_DIMENSION
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return vector / norm


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()