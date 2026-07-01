from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.embeddings import get_embedding_service
from app.services.retrieval import CatalogRepository, INDEX_PATH


def build_index() -> Path:
    catalog = CatalogRepository().load()
    if not catalog:
        raise RuntimeError("Cannot build index because app/data/catalog.json is empty")

    vectors = get_embedding_service().encode([assessment.search_text() for assessment in catalog]).astype(np.float32)
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(INDEX_PATH))
        return INDEX_PATH
    except Exception as exc:
        fallback_path = INDEX_PATH.with_suffix(".npy")
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(fallback_path, vectors)
        print(f"FAISS unavailable, saved fallback vectors to {fallback_path}: {exc}")
        return fallback_path


if __name__ == "__main__":
    path = build_index()
    print(f"Index written to {path}")
