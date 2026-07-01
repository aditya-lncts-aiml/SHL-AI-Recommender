"""
metrics.py

Evaluation metrics for the SHL Assessment Recommendation Agent.

This module provides reusable metrics for evaluating retrieval quality,
recommendation quality, groundedness, hallucination rate, and latency.

Author: Codex + ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List, Sequence, Set


# ==========================================================
# Basic Retrieval Metrics
# ==========================================================


def precision_at_k(
    predicted: Sequence[str],
    relevant: Sequence[str],
    k: int = 10,
) -> float:
    """
    Precision@K

    Precision = Relevant Retrieved / Retrieved

    Example

    predicted:
    [A,B,C,D]

    relevant:
    [A,C,E]

    Precision@4 = 2 / 4 = 0.5
    """

    predicted = predicted[:k]

    if not predicted:
        return 0.0

    relevant_set = set(relevant)

    hits = sum(
        1
        for item in predicted
        if item in relevant_set
    )

    return hits / len(predicted)


def recall_at_k(
    predicted: Sequence[str],
    relevant: Sequence[str],
    k: int = 10,
) -> float:
    """
    Recall@K

    Recall = Relevant Retrieved / Total Relevant
    """

    if not relevant:
        return 1.0

    predicted = predicted[:k]

    relevant_set = set(relevant)

    hits = sum(
        1
        for item in predicted
        if item in relevant_set
    )

    return hits / len(relevant_set)


def hit_rate_at_k(
    predicted: Sequence[str],
    relevant: Sequence[str],
    k: int = 10,
) -> float:
    """
    Hit Rate@K

    Returns

    1.0 -> at least one relevant recommendation

    0.0 -> none found
    """

    predicted = predicted[:k]

    relevant = set(relevant)

    return float(any(x in relevant for x in predicted))


def reciprocal_rank(
    predicted: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """
    Reciprocal Rank

    Example

    predicted

    A
    B
    C

    relevant

    B

    RR = 1/2
    """

    relevant = set(relevant)

    for index, item in enumerate(predicted, start=1):
        if item in relevant:
            return 1 / index

    return 0.0


def mean_reciprocal_rank(
    scores: Iterable[float],
) -> float:
    """
    Mean Reciprocal Rank
    """

    scores = list(scores)

    if not scores:
        return 0.0

    return mean(scores)


# ==========================================================
# Recommendation Quality
# ==========================================================


def recommendation_relevance(
    predicted: Sequence[str],
    relevant: Sequence[str],
) -> float:
    """
    Convenience wrapper.

    Currently uses Recall.

    Easy to replace later.
    """

    return recall_at_k(predicted, relevant)


# ==========================================================
# Groundedness
# ==========================================================


def groundedness_score(
    recommended_names: Sequence[str],
    catalog_names: Set[str],
) -> float:
    """
    Measures whether every recommendation
    exists inside catalog.json.
    """

    if not recommended_names:
        return 1.0

    grounded = sum(
        1
        for name in recommended_names
        if name in catalog_names
    )

    return grounded / len(recommended_names)


def hallucination_rate(
    recommended_names: Sequence[str],
    catalog_names: Set[str],
) -> float:
    """
    Percentage of hallucinated recommendations.
    """

    return 1.0 - groundedness_score(
        recommended_names,
        catalog_names,
    )


# ==========================================================
# Clarification Accuracy
# ==========================================================


def clarification_accuracy(
    expected: bool,
    actual: bool,
) -> float:
    """
    Simple binary accuracy.
    """

    return float(expected == actual)


# ==========================================================
# Comparison Accuracy
# ==========================================================


def comparison_accuracy(
    expected: bool,
    actual: bool,
) -> float:
    """
    Binary comparison evaluation.
    """

    return float(expected == actual)


# ==========================================================
# Latency
# ==========================================================


def average_latency(
    latencies: Sequence[float],
) -> float:
    """
    Average response time.
    """

    if not latencies:
        return 0.0

    return mean(latencies)


# ==========================================================
# Aggregate Report
# ==========================================================


@dataclass
class EvaluationSummary:

    queries: int

    precision: float

    recall: float

    hit_rate: float

    mrr: float

    groundedness: float

    hallucination_rate: float

    clarification_accuracy: float

    comparison_accuracy: float

    latency: float

    def as_dict(self):

        return {
            "queries": self.queries,
            "precision@10": round(self.precision, 4),
            "recall@10": round(self.recall, 4),
            "hit_rate@10": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "groundedness": round(self.groundedness, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "clarification_accuracy": round(
                self.clarification_accuracy,
                4,
            ),
            "comparison_accuracy": round(
                self.comparison_accuracy,
                4,
            ),
            "latency": round(self.latency, 4),
        }

    def pretty_print(self):

        print()

        print("=" * 50)

        print("SHL Evaluation Summary")

        print("=" * 50)

        for key, value in self.as_dict().items():
            print(f"{key:30} {value}")

        print("=" * 50)