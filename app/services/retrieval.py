from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.embeddings import (
    EmbeddingService,
    get_embedding_service,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = ROOT / "app" / "data" / "catalog.json"

INDEX_PATH = ROOT / "app" / "vectorstore" / "faiss.index"


# ==========================================================
# Assessment Model
# ==========================================================

@dataclass(frozen=True)
class Assessment:

    name: str

    url: str

    test_type: str

    description: str

    category: str = ""

    duration: str = ""

    languages: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Assessment":

        return cls(
            name=str(data["name"]).strip(),
            url=str(data["url"]).strip(),
            test_type=str(
                data.get("test_type")
                or data.get("assessment_type")
                or "Assessment"
            ).strip(),
            description=str(
                data.get("description") or ""
            ).strip(),
            category=str(
                data.get("category") or ""
            ).strip(),
            duration=str(
                data.get("duration") or ""
            ).strip(),
            languages=list(
                data.get("languages") or []
            ),
        )

    def search_text(self) -> str:
        """
        Weighted search document.

        Repeating important fields improves
        embedding quality.
        """

        languages = " ".join(self.languages or [])

        return " ".join(
            [

                # Highest weight
                self.name,
                self.name,
                self.name,

                # Medium weight
                self.test_type,
                self.test_type,

                self.category,
                self.category,

                # Lower weight
                self.duration,

                languages,

                # Important context
                self.description,
                self.description,
            ]
        )

    def keyword_tokens(self) -> set[str]:
        """
        Token set used by keyword search.
        """

        text = self.search_text().lower()

        return set(
            re.findall(
                r"[a-z0-9]+",
                text,
            )
        )

    def recommendation_payload(self):

        return {
            "name": self.name,
            "url": self.url,
            "test_type": self.test_type,
        }


# ==========================================================
# Search Result
# ==========================================================

@dataclass(frozen=True)
class SearchResult:

    assessment: Assessment

    score: float

    semantic_score: float = 0.0

    keyword_score: float = 0.0


# ==========================================================
# Hybrid Score
# ==========================================================

@dataclass(frozen=True)
class HybridWeights:

    semantic: float = 0.75

    keyword: float = 0.25


DEFAULT_WEIGHTS = HybridWeights()


# ==========================================================
# Catalog Repository
# ==========================================================

class CatalogRepository:
    """
    Loads and validates catalog.json.

    Responsibilities
    ----------------
    - Load JSON
    - Validate entries
    - Remove duplicates
    - Return Assessment objects
    """

    def __init__(self, path: Path = CATALOG_PATH):
        self.path = path

    def load(self) -> list[Assessment]:

        if not self.path.exists():
            logger.warning(
                "Catalog not found at %s",
                self.path,
            )
            return []

        try:

            with self.path.open(
                "r",
                encoding="utf-8",
            ) as f:

                raw = json.load(f)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid catalog.json: {exc}"
            ) from exc

        if not isinstance(raw, list):
            raise ValueError(
                "catalog.json must contain a list."
            )

        assessments: list[Assessment] = []

        seen_urls: set[str] = set()

        seen_names: set[str] = set()

        for item in raw:

            if not isinstance(item, dict):
                continue

            if "name" not in item:
                continue

            if "url" not in item:
                continue

            try:

                assessment = Assessment.from_dict(item)

            except Exception as exc:

                logger.warning(
                    "Skipping invalid catalog entry: %s",
                    exc,
                )

                continue

            url = assessment.url.lower()

            name = assessment.name.lower()

            # Duplicate URL
            if url in seen_urls:
                continue

            # Duplicate Name
            if name in seen_names:
                continue

            seen_urls.add(url)

            seen_names.add(name)

            assessments.append(
                assessment
            )

        assessments.sort(
            key=lambda x: x.name.lower()
        )

        logger.info(
            "Loaded %s assessments.",
            len(assessments),
        )

        return assessments

    def size(self) -> int:
        return len(self.load())

    def names(self) -> list[str]:
        return [
            a.name
            for a in self.load()
        ]

    def urls(self) -> list[str]:
        return [
            a.url
            for a in self.load()
        ]
    

# ==========================================================
# Retriever
# ==========================================================

class Retriever:

    def __init__(
        self,
        repository: CatalogRepository | None = None,
        embedding_service: EmbeddingService | None = None,
        index_path: Path = INDEX_PATH,
        weights: HybridWeights = DEFAULT_WEIGHTS,
    ):

        self.repository = repository or CatalogRepository()

        self.embedding_service = (
            embedding_service
            or get_embedding_service()
        )

        self.index_path = index_path

        self.weights = weights

        # Load catalog once
        self._catalog = self.repository.load()

        # Cached embeddings
        self._vectors: np.ndarray | None = None

        # Cached FAISS index
        self._faiss_index = None

        # Cached keyword tokens
        self._keyword_cache: list[set[str]] = [
            assessment.keyword_tokens()
            for assessment in self._catalog
        ]

    @property
    def catalog(self) -> list[Assessment]:
        return self._catalog

    @property
    def catalog_size(self) -> int:
        return len(self._catalog)

    # ------------------------------------------------------

    def _catalog_vectors(self) -> np.ndarray:

        if self._vectors is not None:
            return self._vectors

        logger.info(
            "Generating embeddings for %s assessments...",
            len(self._catalog),
        )

        self._vectors = self.embedding_service.encode(
            [
                assessment.search_text()
                for assessment in self._catalog
            ]
        ).astype(np.float32)

        # Normalize vectors for cosine similarity
        try:
            import faiss

            faiss.normalize_L2(self._vectors)

        except Exception:
            pass

        return self._vectors

    # ------------------------------------------------------

    def _load_faiss_index(self):

        if self._faiss_index is not None:
            return self._faiss_index

        if not self.index_path.exists():

            logger.warning(
                "FAISS index not found. "
                "Falling back to in-memory retrieval."
            )

            return None

        try:

            import faiss

            index = faiss.read_index(
                str(self.index_path)
            )

            if index.ntotal != len(self._catalog):

                logger.warning(
                    "Ignoring stale FAISS index "
                    "(%s vectors, catalog has %s)",
                    index.ntotal,
                    len(self._catalog),
                )

                return None

            self._faiss_index = index

            logger.info(
                "Loaded FAISS index with %s vectors.",
                index.ntotal,
            )

            return self._faiss_index

        except Exception as exc:

            logger.warning(
                "Unable to load FAISS index: %s",
                exc,
            )

            return None

    # ------------------------------------------------------

    @staticmethod
    def _normalize_query(
        vector: np.ndarray,
    ) -> np.ndarray:

        try:

            import faiss

            faiss.normalize_L2(vector)

        except Exception:
            pass

        return vector

    # ------------------------------------------------------

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        return re.findall(
            r"[a-z0-9]+",
            text.lower(),
        )
    

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Hybrid Retrieval

        Final Score

            =
            Semantic Similarity
            +
            Keyword Score
            +
            Exact Match Boost
        """

        if not self._catalog:
            return []

        top_k = min(top_k, len(self._catalog))

        # --------------------------------------------------
        # Query Embedding
        # --------------------------------------------------

        query_vector = self.embedding_service.encode(
            [query]
        ).astype(np.float32)

        query_vector = self._normalize_query(
            query_vector
        )

        # --------------------------------------------------
        # Semantic Search
        # --------------------------------------------------

        semantic_scores = np.zeros(
            len(self._catalog),
            dtype=np.float32,
        )

        faiss_index = self._load_faiss_index()

        if faiss_index is not None:

            scores, indices = faiss_index.search(
                query_vector,
                len(self._catalog),
            )

            for score, idx in zip(
                scores[0],
                indices[0],
            ):

                if 0 <= idx < len(self._catalog):
                    semantic_scores[idx] = float(score)

        else:

            vectors = self._catalog_vectors()

            semantic_scores = np.dot(
                vectors,
                query_vector[0],
            )

        # --------------------------------------------------
        # Keyword Search
        # --------------------------------------------------

        query_tokens = Counter(
            self._tokenize(query)
        )

        keyword_scores = np.zeros(
            len(self._catalog),
            dtype=np.float32,
        )

        normalized_query = query.lower()

        for idx, assessment in enumerate(
            self._catalog
        ):

            tokens = self._keyword_cache[idx]

            overlap = sum(
                query_tokens[token]
                for token in tokens
                if token in query_tokens
            )

            score = float(overlap)

            # ------------------------------------------
            # Exact Name Match
            # ------------------------------------------

            if assessment.name.lower() == normalized_query:
                score += 10

            elif assessment.name.lower() in normalized_query:
                score += 6

            elif normalized_query in assessment.name.lower():
                score += 6

            # ------------------------------------------
            # Category Match
            # ------------------------------------------

            if assessment.category:

                if assessment.category.lower() in normalized_query:
                    score += 3

            # ------------------------------------------
            # Test Type Match
            # ------------------------------------------

            if assessment.test_type:

                if assessment.test_type.lower() in normalized_query:
                    score += 2

            # ------------------------------------------
            # Description Match
            # ------------------------------------------

            description = assessment.description.lower()

            if "java" in normalized_query and "java" in description:
                score += 2

            if "python" in normalized_query and "python" in description:
                score += 2

            if "personality" in normalized_query:

                if (
                    "personality" in description
                    or
                    "opq" in assessment.name.lower()
                ):
                    score += 3

            if "cognitive" in normalized_query:

                if (
                    "cognitive" in description
                    or
                    "verify" in assessment.name.lower()
                ):
                    score += 3

            keyword_scores[idx] = score

        # --------------------------------------------------
        # Normalize Keyword Scores
        # --------------------------------------------------

        max_keyword = keyword_scores.max()

        if max_keyword > 0:

            keyword_scores /= max_keyword

        # --------------------------------------------------
        # Hybrid Score
        # --------------------------------------------------

        final_scores = (
            self.weights.semantic
            * semantic_scores
            +
            self.weights.keyword
            * keyword_scores
        )

        ranked = np.argsort(
            final_scores
        )[::-1][:top_k]

        results = []

        for idx in ranked:

            results.append(

                SearchResult(

                    assessment=self._catalog[int(idx)],

                    score=float(
                        final_scores[int(idx)]
                    ),

                    semantic_score=float(
                        semantic_scores[int(idx)]
                    ),
    
                    keyword_score=float(
                        keyword_scores[int(idx)]
                    ),
                ),
            )

        

        return results

    # Name Lookup (Improved)
    # ------------------------------------------------------
    # ------------------------------------------------------
    def find_by_names(

        self,
        names: list[str],
    ) -> list[SearchResult]:
        """
        Find assessments by name using:

        1. Exact match
        2. Substring match
        3. Acronym match
        4. Fuzzy similarity
        """

        if not names:
            return []

        matches: dict[str, SearchResult] = {}

        for query_name in names:

            normalized = self._normalize_name(query_name)

            for assessment in self._catalog:

                assessment_name = self._normalize_name(
                    assessment.name
                )

                score = 0.0

                # --------------------------------------
                # Exact Match
                # --------------------------------------

                if assessment_name == normalized:
                    score = 1.0

                # --------------------------------------
                # Substring Match
                # --------------------------------------

                elif normalized in assessment_name:
                    score = 0.95

                elif assessment_name in normalized:
                    score = 0.90

                # --------------------------------------
                # Acronym Match
                # --------------------------------------

                elif self._acronym_match(
                    normalized,
                    assessment_name,
                ):
                    score = 0.88

                # --------------------------------------
                # Fuzzy Match
                # --------------------------------------

                else:

                    similarity = self._string_similarity(
                        normalized,
                        assessment_name,
                    )

                    if similarity >= 0.80:
                        score = similarity

                if score > 0:

                    current = matches.get(assessment.url)

                    if (
                        current is None
                        or score > current.score
                    ):
                        matches[assessment.url] = SearchResult(
                            assessment=assessment,
                            score=score,
                            semantic_score=score,
                            keyword_score=score,
                        )

        if matches:
            return sorted(
                matches.values(),
                key=lambda r: r.score,
                reverse=True,
            )[:10]

        # fallback
        return self.search(
            " ".join(names),
            top_k=min(
                10,
                max(3, len(names) * 3),
            ),
        )


    # ------------------------------------------------------
    # Helpers
    # ------------------------------------------------------

    @staticmethod
    def _normalize_name(
        text: str,
    ) -> str:

        return re.sub(
            r"[^a-z0-9]",
            "",
            text.lower(),
        )


    @staticmethod
    def _acronym_match(
        query: str,
        assessment: str,
    ) -> bool:

        acronyms = {
            "opq": "occupationalpersonalityquestionnaire",
            "mq": "motivationalquestionnaire",
            "sjt": "situationaljudgementtest",
            "verify": "verify",
            "adept": "adept",
        }

        if query in acronyms:

            return acronyms[query] in assessment

        return False


    @staticmethod
    def _string_similarity(
        a: str,
        b: str,
    ) -> float:

        from difflib import SequenceMatcher

        return SequenceMatcher(
            None,
            a,
            b,
        ).ratio()


    # ------------------------------------------------------
    # Debug Helper
    # ------------------------------------------------------

    def debug_search(
        self,
        query: str,
        top_k: int = 5,
    ):

        results = self.search(
            query,
            top_k,
        )

        print()

        print("=" * 70)

        print("Hybrid Retrieval Debug")

        print("=" * 70)

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print()
    
            print(f"{rank}. {result.assessment.name}")
    
            print(
                f"Semantic : {result.semantic_score:.3f}"
            )
    
            print(
                f"Keyword  : {result.keyword_score:.3f}"
            )
    
            print(
                f"Final     : {result.score:.3f}"
            )
    
        print("=" * 70)