from __future__ import annotations

import unittest

from test_backend_main_helpers import load_main_module

try:
    from fastapi.testclient import TestClient
except Exception as exc:  # pragma: no cover - exercised when httpx is missing locally.
    TestClient = None
    TESTCLIENT_IMPORT_ERROR = exc
else:
    TESTCLIENT_IMPORT_ERROR = None


@unittest.skipIf(TestClient is None, f"FastAPI TestClient unavailable: {TESTCLIENT_IMPORT_ERROR}")
class CleanSRTAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_module()
        cls.client = TestClient(cls.main.app)

    def setUp(self):
        self.main.rate_limit_hits.clear()

    def test_health_endpoint_returns_ok(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_clean_srt_endpoint_cleans_zh_srt(self):
        response = self.client.post(
            "/srt/clean",
            json={
                "language": "zh",
                "srt_text": "1\n00:00:00,000 --> 00:00:01,000\n记忆供电法 PatternDrill\n",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("记忆宫殿法 Pattern Drill", payload["clean_srt"])
        self.assertTrue(payload["changes"])

    def test_clean_srt_endpoint_rejects_invalid_language(self):
        response = self.client.post(
            "/srt/clean",
            json={
                "language": "fr",
                "srt_text": "1\n00:00:00,000 --> 00:00:01,000\ntest\n",
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_clean_srt_endpoint_rejects_oversized_srt(self):
        response = self.client.post(
            "/srt/clean",
            json={
                "language": "zh",
                "srt_text": "x" * (self.main.MAX_CLEAN_SRT_CHARS + 1),
            },
        )

        self.assertEqual(response.status_code, 413)

    def test_clean_srt_endpoint_falls_back_for_malformed_srt(self):
        response = self.client.post(
            "/srt/clean",
            json={
                "language": "zh",
                "srt_text": "not an srt",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["clean_srt"], "not an srt")
        self.assertEqual(payload["changes"][0]["type"], "validation_fallback")


if __name__ == "__main__":
    unittest.main()
