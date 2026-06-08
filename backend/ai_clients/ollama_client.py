from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

from .base import AICleanClientConfig, AICleanClientError, AICleanTimeoutError


AI_CLEAN_SYSTEM_PROMPT = """You are an ASR subtitle text correction engine.
Correct only recognition errors, punctuation, spacing, and very minor grammar issues.
Do not rewrite.
Do not summarize.
Do not translate.
Do not expand.
Do not remove spoken content.
Do not add content that is not present in the audio.
Preserve oral style.
Preserve each block independently.
Do not move text between blocks.
Return JSON only.
Return an array of objects with index and clean_text.
Do not output markdown.
Do not output full SRT.
Do not output SRT timestamps."""


class OllamaAICleanClient:
    def __init__(self, config: AICleanClientConfig) -> None:
        self.config = config

    def clean_blocks(self, blocks: list[dict[str, object]], language: str) -> str:
        prompt = build_ai_clean_prompt(blocks, language)
        payload = {
            "model": self.config.model,
            "system": AI_CLEAN_SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
            },
        }
        request = UrlRequest(
            f"{self.config.base_url.rstrip('/')}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "linguaforge-local/ai-clean",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise AICleanTimeoutError("AI clean request timed out") from exc
        except HTTPError as exc:
            raise AICleanClientError(f"AI clean provider returned HTTP {exc.code}") from exc
        except URLError as exc:
            raise AICleanClientError(f"AI clean provider is unavailable: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise AICleanClientError("AI clean provider returned invalid JSON") from exc

        model_text = response_payload.get("response")
        if not isinstance(model_text, str):
            raise AICleanClientError("AI clean provider response did not contain text")
        return model_text


def build_ai_clean_prompt(blocks: list[dict[str, object]], language: str) -> str:
    return "\n".join(
        [
            f"Language: {language or 'auto'}",
            "Input subtitle blocks are JSON objects with index and text only.",
            "Correct each block independently and return JSON only.",
            "Input:",
            json.dumps(blocks, ensure_ascii=False),
        ]
    )
