from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "app" / "prompts" / "system_prompt.txt"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiClient:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        load_dotenv()
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.system_prompt = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists() else ""

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def generate(self, user_message: str, retrieved: list[dict[str, Any]], conversation: list[dict[str, str]]) -> str | None:
        if not self.enabled:
            return None
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name, system_instruction=self.system_prompt)
            prompt = self._build_prompt(user_message, retrieved, conversation)
            response = model.generate_content(prompt)
            text = getattr(response, "text", None)
            return text.strip() if text else None
        except Exception as exc:
            logger.warning("Gemini generation failed, using deterministic response: %s", exc)
            return None

    @staticmethod
    def _build_prompt(user_message: str, retrieved: list[dict[str, Any]], conversation: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "task": "Answer the latest user message using only the retrieved SHL assessments.",
                "latest_user_message": user_message,
                "conversation": conversation[-8:],
                "retrieved_assessments": retrieved,
                "response_contract": {
                    "reply": "string",
                    "recommendations": [{"name": "catalog name", "url": "catalog URL", "test_type": "catalog test_type"}],
                    "end_of_conversation": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
