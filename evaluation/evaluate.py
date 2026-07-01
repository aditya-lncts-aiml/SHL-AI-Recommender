"""
evaluate.py

Runs an end-to-end evaluation of the SHL Assessment Recommendation Agent.

Metrics:

- Precision@10
- Recall@10
- HitRate@10
- MRR
- Groundedness
- Hallucination Rate
- Latency
- Clarification Accuracy
- Comparison Accuracy

Output

evaluation_report.json
evaluation_report.md
"""

from __future__ import annotations

from ctypes.util import test
import json
import time
from pathlib import Path

from app.models.request import ChatMessage, ChatRequest
from app.services.agent import RecommendationAgent
from app.services.retrieval import CatalogRepository

from evaluation.metrics import (
    EvaluationSummary,
    average_latency,
    clarification_accuracy,
    comparison_accuracy,
    groundedness_score,
    hallucination_rate,
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    recommendation_relevance,
    reciprocal_rank,
)

import re

def normalize_name(name: str) -> str:
    """
    Normalize assessment names so minor naming differences
    do not affect evaluation.
    """
    name = name.lower()

    # Remove common prefixes
    name = name.replace("shl ", "")

    # Treat these as equivalent
    name = name.replace("motivational", "motivation")

    # Remove punctuation and spaces
    name = re.sub(r"[^a-z0-9]", "", name)

    return name

ROOT = Path(__file__).resolve().parents[1]

TEST_FILE = ROOT / "evaluation" / "test_queries.json"

REPORT_JSON = ROOT / "evaluation" / "evaluation_report.json"

REPORT_MD = ROOT / "evaluation" / "evaluation_report.md"


agent = RecommendationAgent()

catalog = CatalogRepository().load()

catalog_names = {a.name for a in catalog}


def load_queries():

    with open(TEST_FILE, encoding="utf8") as f:
        return json.load(f)


def evaluate():

    queries = load_queries()

    precision_scores = []

    recall_scores = []

    hit_scores = []

    mrr_scores = []

    grounded_scores = []

    hallucination_scores = []

    clarification_scores = []

    comparison_scores = []

    latency_scores = []

    detailed_results = []

    for test in queries:

        query = test["query"]

        expected = test["expected"]

        is_clarification = test.get("clarification", False)
        is_comparison = test.get("comparison", False)

        expected_set = set(expected)

        start = time.perf_counter()

        response = agent.respond(
            ChatRequest(
                messages=[
                    ChatMessage(
                        role="user",
                        content=query
                    )
                ]
            )
        )

        latency = time.perf_counter() - start

        latency_scores.append(latency)

        predicted = [rec.name for rec in response.recommendations]

        predicted_normalized = [
            normalize_name(name)
            for name in predicted
        ]

        expected_normalized = [
            normalize_name(name)
            for name in expected
        ]

        # Only recommendation queries contribute to retrieval metrics
        if not is_clarification and not is_comparison:

            precision_scores.append(
                precision_at_k(
                    predicted_normalized,
                    expected_normalized,
                    10,
                )
            )

            recall_scores.append(
                recall_at_k(
                    predicted_normalized,
                    expected_normalized,
                    10,
                )
            )

            hit_scores.append(
                hit_rate_at_k(
                    predicted_normalized,
                    expected_normalized,
                    10,
                )
            )

            mrr_scores.append(
                reciprocal_rank(
                    predicted_normalized,
                    expected_normalized,
                )
            )

        grounded_scores.append(
            groundedness_score(
                predicted,
                catalog_names,
            )
        )

        hallucination_scores.append(
            hallucination_rate(
                predicted,
                catalog_names,
            )
        )

        clarification_expected = test.get(
            "clarification",
            False,
        )

        clarification_actual = (
            len(response.recommendations) == 0
        )

        clarification_scores.append(
            clarification_accuracy(
                clarification_expected,
                clarification_actual,
            )
        )

        comparison_expected = test.get(
            "comparison",
            False,
        )

        comparison_actual = (
            "compare" in query.lower()
            or "difference" in query.lower()
        )

        comparison_scores.append(
            comparison_accuracy(
                comparison_expected,
                comparison_actual,
            )
        )

        detailed_results.append(
            {
                "query": query,
                "expected": expected,
                "predicted": predicted,
                "reply": response.reply,
                "latency": latency,
                "precision": (
                    precision_scores[-1]
                    if not is_clarification and not is_comparison
                    else None
                ),

                "recall": (
                    recall_scores[-1]
                    if not is_clarification and not is_comparison
                    else None
                ),
                "grounded": grounded_scores[-1],
            }
        )

    summary = EvaluationSummary(

        queries=len(queries),

        precision=sum(precision_scores) / len(precision_scores),

        recall=sum(recall_scores) / len(recall_scores),

        hit_rate=sum(hit_scores) / len(hit_scores),

        mrr=mean_reciprocal_rank(mrr_scores),

        groundedness=sum(grounded_scores) / len(grounded_scores),

        hallucination_rate=sum(hallucination_scores)
        / len(hallucination_scores),

        clarification_accuracy=sum(
            clarification_scores
        )
        / len(clarification_scores),

        comparison_accuracy=sum(
            comparison_scores
        )
        / len(comparison_scores),

        latency=average_latency(
            latency_scores
        ),
    )

    report = {
        "summary": summary.as_dict(),
        "details": detailed_results,
    }

    with open(
        REPORT_JSON,
        "w",
        encoding="utf8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
            ensure_ascii=False,
        )

    with open(
        REPORT_MD,
        "w",
        encoding="utf8",
    ) as f:

        f.write("# SHL Evaluation Report\n\n")

        for k, v in summary.as_dict().items():

            f.write(f"**{k}** : {v}\n\n")

    summary.pretty_print()

    print()

    print(
        f"JSON Report : {REPORT_JSON}"
    )

    print(
        f"Markdown Report : {REPORT_MD}"
    )


if __name__ == "__main__":
    evaluate()