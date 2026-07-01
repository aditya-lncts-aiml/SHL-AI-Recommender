from __future__ import annotations

import re
from dataclasses import dataclass


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"forget (all )?(previous|prior|above) instructions",
    r"system prompt",
    r"developer message",
    r"jailbreak",
    r"bypass (the )?(rules|guardrails|policy)",
    r"reveal.*(prompt|instructions|secrets|api key)",
]

OFF_TOPIC_PATTERNS = {
    "politics": [r"\belection\b", r"\bpresident\b", r"\bprime minister\b", r"\bpolitical\b", r"\bparty\b"],
    "medical": [r"\bdiagnos(e|is)\b", r"\bmedicine\b", r"\btreatment\b", r"\bsymptom\b", r"\bdoctor\b"],
    "legal": [r"\blawsuit\b", r"\blegal advice\b", r"\bcontract\b", r"\battorney\b", r"\bcourt\b"],
    "general hiring advice": [r"\binterview questions\b", r"\bsalary\b", r"\bjob description\b", r"\bhiring strategy\b"],
}

SHL_RELATED_TERMS = [
    "shl",
    "assessment",
    "assessments",
    "test",
    "tests",
    "opq",
    "verify",
    "personality",
    "cognitive",
    "behavioral",
    "skills",
    "coding",
    "sales",
    "graduate",
    "manager",
    "leadership",
    "candidate",
    "role",
    "hiring",
    "recommend",
    "compare",
    "include",
    "exclude",
    "duration",
]


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str = ""
    reply: str = ""


class Guardrails:
    def validate(self, text: str) -> GuardrailResult:
        normalized = self._normalize(text)
        if self._matches_any(normalized, PROMPT_INJECTION_PATTERNS):
            return GuardrailResult(
                allowed=False,
                reason="prompt_injection",
                reply="I can help only with SHL assessment recommendations and comparisons, and I cannot follow requests to ignore my instructions or reveal hidden prompts.",
            )

        for topic, patterns in OFF_TOPIC_PATTERNS.items():
            if self._matches_any(normalized, patterns):
                return GuardrailResult(
                    allowed=False,
                    reason=topic,
                    reply="I can help only with SHL assessment recommendations, refinements, and comparisons. Please ask about SHL assessments.",
                )

        if not any(term in normalized for term in SHL_RELATED_TERMS):
            return GuardrailResult(
                allowed=False,
                reason="off_topic",
                reply="I can help only with SHL assessment recommendations, refinements, and comparisons. What SHL assessment need are you working on?",
            )

        return GuardrailResult(allowed=True)

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)
