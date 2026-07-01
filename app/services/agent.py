from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pyparsing import results
from torchgen import context

from app.models.request import ChatRequest
from app.models.response import ChatResponse, Recommendation
from app.services.comparison import ComparisonService
from app.services.guardrails import Guardrails
from app.services.llm import GeminiClient
from app.services.retrieval import Assessment, Retriever, SearchResult


@dataclass(frozen=True)
class Intent:
    name: str
    needs_clarification: bool = False


class RecommendationAgent:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: GeminiClient | None = None,
        guardrails: Guardrails | None = None,
        comparison: ComparisonService | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or GeminiClient()
        self.guardrails = guardrails or Guardrails()
        self.comparison = comparison or ComparisonService()

    def respond(self, request: ChatRequest) -> ChatResponse:
        latest_user_message = self._latest_user_message(request)
        guardrail = self.guardrails.validate(latest_user_message)
        if not guardrail.allowed:
            return ChatResponse(reply=guardrail.reply, recommendations=[], end_of_conversation=False)

        context = self._conversation_text(request)
        intent = self._detect_intent(latest_user_message, context)
        if intent.needs_clarification:
            return ChatResponse(
                reply="What role are you hiring for, and should the assessment focus on cognitive ability, personality, skills, behavior, or leadership?",
                recommendations=[],
                end_of_conversation=False,
            )

        if intent.name == "comparison":
            return self._comparison_response(latest_user_message)

        query = self._query_from_conversation(request)
        results = self.retriever.search(query, top_k=10)
        selected = self._filter_results(results, latest_user_message)
        return self._recommendation_response(latest_user_message, request, selected)

    @staticmethod
    def _latest_user_message(request: ChatRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content
        raise ValueError("messages must include at least one user message")

    @staticmethod
    def _conversation_text(request: ChatRequest) -> str:
        return " ".join(message.content for message in request.messages)

    @staticmethod
    def _query_from_conversation(request: ChatRequest) -> str:
        user_messages = [message.content for message in request.messages if message.role == "user"]
        return " ".join(user_messages[-4:])

    def _detect_intent(self, latest: str, context: str) -> Intent:
        normalized = latest.lower()

        # Comparison intent
        if any(word in normalized for word in ["compare", "difference", "versus", " vs "]):
            return Intent("comparison")

        # Has the user already given enough context?
        has_specific_context = any(
            term in context.lower()
            for term in [
                "developer",
                "manager",
                "sales",
                "graduate",
                "personality",
                "cognitive",
                "skills",
                "leadership",
                "behavior",
                "coding",
                "technical",
                "java",
                "python",
                "analyst",
                "engineer"
            ]
        )

        # Generic requests should trigger clarification
        generic_request = (
            "assessment" in normalized
            or "recommend" in normalized
            or "test" in normalized
            or "screening" in normalized
        )
    
        if generic_request and not has_specific_context:
            return Intent("recommendation", needs_clarification=True)

        return Intent("recommendation")

    def _comparison_response(self, latest_user_message: str) -> ChatResponse:
        names = self._extract_possible_assessment_names(latest_user_message)
        results = self.retriever.find_by_names(names) if names else self.retriever.search(latest_user_message, top_k=4)
        assessments = self._dedupe_assessments([result.assessment for result in results])[:4]
        reply = self.comparison.compare(assessments)
        return ChatResponse(
            reply=reply,
            recommendations=[Recommendation(**assessment.recommendation_payload()) for assessment in assessments[:10]],
            end_of_conversation=False,
        )

    def _recommendation_response(self, latest_user_message: str, request: ChatRequest, results: list[SearchResult]) -> ChatResponse:
        assessments = [result.assessment for result in results[:10]]
        payloads = [self._assessment_payload(assessment) for assessment in assessments]
        llm_text = self.llm.generate(
            latest_user_message,
            payloads,
            [{"role": message.role, "content": message.content} for message in request.messages],
        )
        reply = self._safe_llm_reply(llm_text, assessments) if llm_text else self._deterministic_reply(assessments)
        return ChatResponse(
            reply=reply,
            recommendations=[Recommendation(**assessment.recommendation_payload()) for assessment in assessments],
            end_of_conversation=False,
        )
    def _filter_results(self, results: list[SearchResult], latest_message: str) -> list[SearchResult]:
        if not results:
            return []

        normalized = latest_message.lower()

        # Search across the full catalog instead of only the retrieved results.
        catalog_results = [
            SearchResult(assessment, 1.0)
            for assessment in self.retriever.catalog
        ]

        filtered = results

        if "personality" in normalized:
            personality = [
                result
                for result in catalog_results
                if (
                    "personality" in result.assessment.search_text().lower()
                    or "opq" in result.assessment.name.lower()
                )
            ]
            filtered = personality or filtered

        if "cognitive" in normalized:
            cognitive = [
                result
                for result in catalog_results
                if (
                    "cognitive" in result.assessment.search_text().lower()
                    or "verify" in result.assessment.name.lower()
                )
            ]
            filtered = cognitive or filtered

        if "coding" in normalized or "developer" in normalized:
            coding = [
                result
                for result in catalog_results
                if any(
                    keyword in result.assessment.search_text().lower()
                    for keyword in (
                        "coding",
                        "programming",
                        "developer",
                        "technical",
                    )
                )
            ]
            filtered = coding or filtered

        return filtered[:10]

    @staticmethod
    def _deterministic_reply(assessments: list[Assessment]) -> str:
        if not assessments:
            return "I could not find a matching SHL assessment in the local catalog. Please add more SHL catalog entries or run the scraper and index builder."
        names = ", ".join(assessment.name for assessment in assessments[:5])
        return f"I found these SHL assessments from the catalog: {names}. I can refine the list by role, skill area, seniority, duration, or assessment type."

    @staticmethod
    def _safe_llm_reply(text: str | None, assessments: list[Assessment]) -> str:
        if not text:
            return RecommendationAgent._deterministic_reply(assessments)
        allowed_names = {assessment.name for assessment in assessments}
        allowed_urls = {assessment.url for assessment in assessments}
        extracted_urls = [url.rstrip(".,)") for url in re.findall(r"https?://\S+", text)]
        if any(url not in allowed_urls for url in extracted_urls):
            return RecommendationAgent._deterministic_reply(assessments)
        if not any(name in text for name in allowed_names):
            return RecommendationAgent._deterministic_reply(assessments)
        return text

    @staticmethod
    def _assessment_payload(assessment: Assessment) -> dict[str, Any]:
        return {
            "name": assessment.name,
            "url": assessment.url,
            "test_type": assessment.test_type,
            "description": assessment.description,
            "category": assessment.category,
            "duration": assessment.duration,
            "languages": assessment.languages or [],
        }

    @staticmethod
    def _extract_possible_assessment_names(text: str) -> list[str]:
        compact = re.sub(r"\b(compare|difference between|differences between|versus|vs\.?|and)\b", "|", text, flags=re.IGNORECASE)
        parts = [part.strip(" .,:;!?") for part in compact.split("|")]
        names = [part for part in parts if len(part) >= 2 and not part.lower().startswith("what")]
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        names.extend(item for pair in quoted for item in pair if item)
        important = re.findall(r"\b(OPQ|Verify|MQ|ADEPT|SJT|Motivation Questionnaire)\b", text, flags=re.IGNORECASE)
        names.extend(important)
        return list(dict.fromkeys(names))

    @staticmethod
    def _dedupe_assessments(assessments: list[Assessment]) -> list[Assessment]:
        unique: dict[str, Assessment] = {}
        for assessment in assessments:
            unique[assessment.url] = assessment
        return list(unique.values())

    @staticmethod
    def parse_structured_llm_response(text: str) -> dict[str, Any]:
        return json.loads(text)
