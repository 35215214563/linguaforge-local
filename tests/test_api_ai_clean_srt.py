from __future__ import annotations

import json
import unittest

from backend.ai_cleaner import AICleanConfig, AICleaner
from backend.ai_clients import AICleanClientError
from backend.srt_parser import parse_srt
from test_backend_main_helpers import load_main_module

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover - exercised when httpx is missing locally.
    TestClient = None
    TESTCLIENT_IMPORT_ERROR = exc
else:
    TESTCLIENT_IMPORT_ERROR = None


RAW_SRT = """1
00:00:00,000 --> 00:00:02,000
问答对练

2
00:00:02,500 --> 00:00:04,000
过去式还是现在式
"""


class FakeAICleanClient:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def clean_blocks(self, blocks: list[dict[str, object]], language: str) -> str:
        if self.error:
            raise self.error
        return self.response


def ai_response(items: list[dict[str, object]]) -> str:
    return json.dumps(items, ensure_ascii=False)


@unittest.skipIf(TestClient is None, f"FastAPI TestClient unavailable: {TESTCLIENT_IMPORT_ERROR}")
class AICleanSRTAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_module()
        cls.client = TestClient(cls.main.app)
        cls.original_ai_cleaner = cls.main.ai_cleaner

    @classmethod
    def tearDownClass(cls):
        cls.main.ai_cleaner = cls.original_ai_cleaner

    def setUp(self):
        self.main.rate_limit_hits.clear()

    def install_ai_cleaner(
        self,
        client: FakeAICleanClient,
        enabled: bool = True,
    ) -> None:
        self.main.ai_cleaner = AICleaner(
            srt_cleaner=self.main.srt_cleaner,
            config_factory=lambda: AICleanConfig(enabled=enabled, model="test-model"),
            client_factory=lambda _config: client,
        )

    def test_ai_clean_endpoint_returns_200_and_applies_valid_correction(self):
        self.install_ai_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": "问答对练。"},
                        {"index": 2, "clean_text": "过去式还是现在式？"},
                    ]
                )
            )
        )

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "zh", "srt_text": RAW_SRT},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ai_used"])
        self.assertIsNone(payload["fallback_reason"])
        self.assertIn("问答对练。", payload["ai_clean_srt"])
        self.assertIn("rule_based_srt", payload)
        self.assertTrue(any(change["type"] == "ai_text_correction" for change in payload["changes"]))

    def test_ai_clean_endpoint_preserves_indices_and_timing(self):
        self.install_ai_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": "问答对练。"},
                        {"index": 2, "clean_text": "过去式还是现在式？"},
                    ]
                )
            )
        )

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "zh", "srt_text": RAW_SRT},
        )

        self.assertEqual(response.status_code, 200)
        before_blocks = parse_srt(RAW_SRT)
        after_blocks = parse_srt(response.json()["ai_clean_srt"])
        self.assertEqual(len(after_blocks), len(before_blocks))
        self.assertEqual([block.index for block in after_blocks], [block.index for block in before_blocks])
        self.assertEqual([block.start for block in after_blocks], [block.start for block in before_blocks])
        self.assertEqual([block.end for block in after_blocks], [block.end for block in before_blocks])

    def test_invalid_language_returns_400(self):
        self.install_ai_cleaner(FakeAICleanClient("not called"))

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "fr", "srt_text": RAW_SRT},
        )

        self.assertEqual(response.status_code, 400)

    def test_oversized_srt_returns_413(self):
        self.install_ai_cleaner(FakeAICleanClient("not called"))

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "zh", "srt_text": "x" * (self.main.MAX_CLEAN_SRT_CHARS + 1)},
        )

        self.assertEqual(response.status_code, 413)

    def test_malformed_srt_falls_back_safely(self):
        self.install_ai_cleaner(FakeAICleanClient("not called"))

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "zh", "srt_text": "not an srt"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ai_used"])
        self.assertEqual(payload["ai_clean_srt"], "not an srt")
        self.assertEqual(payload["rule_based_srt"], "not an srt")
        self.assertIn("malformed", payload["fallback_reason"])

    def test_ai_disabled_by_request_falls_back(self):
        self.install_ai_cleaner(FakeAICleanClient("not called"))

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "zh", "srt_text": RAW_SRT, "ai_enabled": False},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ai_used"])
        self.assertEqual(payload["ai_clean_srt"], payload["rule_based_srt"])
        self.assertEqual(payload["fallback_reason"], "AI clean disabled by request.")

    def test_ai_disabled_by_env_falls_back(self):
        self.install_ai_cleaner(FakeAICleanClient("not called"), enabled=False)

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "zh", "srt_text": RAW_SRT},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ai_used"])
        self.assertEqual(payload["ai_clean_srt"], payload["rule_based_srt"])
        self.assertEqual(payload["fallback_reason"], "AI clean disabled by environment.")

    def test_ai_unavailable_falls_back(self):
        self.install_ai_cleaner(FakeAICleanClient(error=AICleanClientError("unavailable")))

        response = self.client.post(
            "/srt/ai-clean",
            json={"language": "zh", "srt_text": RAW_SRT},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["ai_used"])
        self.assertEqual(payload["ai_clean_srt"], payload["rule_based_srt"])
        self.assertIn("unavailable", payload["fallback_reason"])


if __name__ == "__main__":
    unittest.main()
