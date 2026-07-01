from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi



EXACT_MATCH_BOOST = 10
PARTIAL_MATCH_BOOST = 6
CATEGORY_BOOST = 3
TYPE_BOOST = 2
SKILL_BOOST = 2


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = ROOT / "app" / "data" / "catalog.json"



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
    ):

        self.repository = repository or CatalogRepository()


        # Load catalog once
        self._catalog = self.repository.load()

        documents = [
            assessment.search_text()
            for assessment in self._catalog
        ]

        # TF-IDF
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            documents
        )

        # BM25
        self.bm25 = BM25Okapi(
            [
                doc.lower().split()
                for doc in documents
            ]
        )


    @property
    def catalog(self) -> list[Assessment]:
        return self._catalog

    @property
    def catalog_size(self) -> int:
        return len(self._catalog)

    # ------------------------------------------------------


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

        query_vector = self.vectorizer.transform(
            [query]
        )

        semantic_scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix,
        )[0]

        # --------------------------------------------------
        # Keyword Search
        # --------------------------------------------------


        keyword_scores = np.zeros(
            len(self._catalog),
            dtype=np.float32,
        )

        normalized_query = query.lower().strip()

        bm25_scores = self.bm25.get_scores(
                re.findall(
                    r"[a-z0-9]+",
                    query.lower(),
                )
            )
        if bm25_scores.max() > 0:
            bm25_scores /= bm25_scores.max()

        fuzzy_scores = np.zeros(
            len(self._catalog),
            dtype=np.float32,
        )
        for idx, assessment in enumerate(
            self._catalog
        ):


            

            fuzzy_score = (
                fuzz.token_set_ratio(
                    query,
                    assessment.name,
                )
                / 100.0
            )
            fuzzy_scores[idx] = fuzzy_score

            # ------------------------------------------
            # Exact Name Match
            # ------------------------------------------

            score = bm25_scores[idx]

            if assessment.name.lower() == normalized_query:
                score += EXACT_MATCH_BOOST
            elif assessment.name.lower() in normalized_query:
                score += PARTIAL_MATCH_BOOST
            elif normalized_query in assessment.name.lower():
                score += PARTIAL_MATCH_BOOST

            # ------------------------------------------
            # Category Match
            # ------------------------------------------

            if assessment.category:

                if assessment.category.lower() in normalized_query:
                    score += CATEGORY_BOOST

            # ------------------------------------------
            # Test Type Match
            # ------------------------------------------

            if assessment.test_type:

                if assessment.test_type.lower() in normalized_query:
                    score += TYPE_BOOST

            # ------------------------------------------
            # Description Match
            # ------------------------------------------

            description = assessment.search_text().lower()

            if "java" in normalized_query and "java" in description:
                score += SKILL_BOOST

            if "python" in normalized_query and "python" in description:
                score += SKILL_BOOST

            if "personality" in normalized_query:
                if (
                    "personality" in description
                    or "opq" in assessment.name.lower()
                ):
                    score += CATEGORY_BOOST

            if "cognitive" in normalized_query:
                if (
                    "cognitive" in description
                    or "verify" in assessment.name.lower()
                ):
                    score += CATEGORY_BOOST

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

        SEMANTIC_WEIGHT = 0.4
        KEYWORD_WEIGHT = 0.4
        FUZZY_WEIGHT = 0.2

        final_scores = (
            SEMANTIC_WEIGHT * semantic_scores
            + KEYWORD_WEIGHT * keyword_scores
            + FUZZY_WEIGHT * fuzzy_scores
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
                        semantic_scores[idx]
                    ),

                    keyword_score=float(
                        keyword_scores[idx]
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

        logger.debug()

        logger.debug("=" * 70)

        logger.debug("Hybrid Retrieval Debug")

        logger.debug("=" * 70)

        for rank, result in enumerate(
            results,
            start=1,
        ):

            logger.debug()
    
            logger.debug(f"{rank}. {result.assessment.name}")
    
            logger.debug(
                f"Semantic : {result.semantic_score:.3f}"
            )
    
            logger.debug(
                f"Keyword  : {result.keyword_score:.3f}"
            )
    
            logger.debug(
                f"Final     : {result.score:.3f}"
            )
    
        logger.debug("=" * 70)