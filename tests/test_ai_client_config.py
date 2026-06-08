from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.ai_cleaner import AICleanConfig, create_ai_clean_client
from backend.ai_clients import AICleanClientError, OllamaAICleanClient


class FakeOllamaResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return self.body


class AICleanClientConfigTests(unittest.TestCase):
    def test_model_name_is_read_from_environment(self):
        with patch.dict("os.environ", {"AI_CLEAN_MODEL": "qwen3:14b"}, clear=False):
            config = AICleanConfig.from_env()

        self.assertEqual(config.model, "qwen3:14b")

    def test_provider_is_read_from_environment(self):
        with patch.dict("os.environ", {"AI_CLEAN_PROVIDER": "ollama"}, clear=False):
            config = AICleanConfig.from_env()

        self.assertEqual(config.provider, "ollama")

    def test_switching_model_does_not_require_endpoint_logic_changes(self):
        with patch.dict("os.environ", {"AI_CLEAN_MODEL": "qwen3:8b"}, clear=False):
            first = AICleanConfig.from_env()
        with patch.dict("os.environ", {"AI_CLEAN_MODEL": "qwen3:14b"}, clear=False):
            second = AICleanConfig.from_env()

        self.assertEqual(first.model, "qwen3:8b")
        self.assertEqual(second.model, "qwen3:14b")

    def test_generation_options_are_read_from_environment(self):
        with patch.dict(
            "os.environ",
            {
                "AI_CLEAN_NUM_PREDICT": "512",
                "AI_CLEAN_FORMAT_JSON": "false",
                "AI_CLEAN_THINK": "true",
            },
            clear=False,
        ):
            config = AICleanConfig.from_env()

        self.assertEqual(config.num_predict, 512)
        self.assertFalse(config.format_json)
        self.assertTrue(config.think)

    def test_ollama_client_is_selected_by_provider(self):
        config = AICleanConfig(provider="ollama", model="test-model")

        client = create_ai_clean_client(config)

        self.assertIsInstance(client, OllamaAICleanClient)
        self.assertEqual(client.config.model, "test-model")

    def test_unsupported_provider_raises_client_error(self):
        with self.assertRaises(AICleanClientError):
            create_ai_clean_client(AICleanConfig(provider="unknown-provider"))

    def test_main_has_no_ai_provider_specific_http_logic(self):
        main_text = Path("backend/main.py").read_text(encoding="utf-8").lower()

        self.assertNotIn("ollama", main_text)
        self.assertNotIn("ai_clean_model", main_text)
        self.assertNotIn("ai_clean_base_url", main_text)
        self.assertNotIn("/api/generate", main_text)
        self.assertNotIn("qwen3", main_text)

    def test_ollama_payload_sets_generation_controls(self):
        config = AICleanConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="test-model",
            temperature=0.25,
            num_predict=777,
            format_json=True,
            think=False,
        )
        client = OllamaAICleanClient(config.to_client_config())

        with patch(
            "backend.ai_clients.ollama_client.urlopen",
            return_value=FakeOllamaResponse(b'{"response":"[]"}'),
        ) as urlopen_mock:
            client.clean_blocks([{"index": 1, "text": "问答对练"}], "zh")

        request = urlopen_mock.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["options"]["temperature"], 0.25)
        self.assertEqual(payload["options"]["num_predict"], 777)

    def test_ollama_payload_prompt_includes_relevant_advisory_hints(self):
        config = AICleanConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="test-model",
        )
        client = OllamaAICleanClient(config.to_client_config())

        with patch(
            "backend.ai_clients.ollama_client.urlopen",
            return_value=FakeOllamaResponse(b'{"response":"[]"}'),
        ) as urlopen_mock:
            client.clean_blocks([{"index": 1, "text": "今天要做 PatternDrill"}], "en")

        request = urlopen_mock.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        prompt = payload["prompt"]
        self.assertIn("advisory, not mandatory replacements", prompt)
        self.assertIn("PatternDrill", prompt)
        self.assertIn("Pattern Drill", prompt)
        self.assertNotIn("問答對聯", prompt)


if __name__ == "__main__":
    unittest.main()
