from __future__ import annotations

from app.services.retrieval import Assessment


class ComparisonService:
    def compare(self, assessments: list[Assessment]) -> str:
        if len(assessments) < 2:
            return "I found one likely SHL match. Please name another SHL assessment to compare it with."

        rows = []
        for assessment in assessments[:4]:
            details = [assessment.test_type]
            if assessment.duration:
                details.append(f"duration: {assessment.duration}")
            if assessment.category:
                details.append(f"category: {assessment.category}")
            rows.append(f"{assessment.name}: {'; '.join(details)}. {assessment.description}")

        return "Here is a grounded comparison from the SHL catalog: " + " ".join(rows)
