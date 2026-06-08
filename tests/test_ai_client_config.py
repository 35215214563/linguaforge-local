from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from backend.ai_cleaner import AICleanConfig, create_ai_clean_client
from backend.ai_clients import AICleanClientError, OllamaAICleanClient


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


if __name__ == "__main__":
    unittest.main()
