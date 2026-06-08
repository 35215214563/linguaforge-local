from __future__ import annotations

import importlib
import sys
import tempfile
import time
import types
import unittest
from concurrent.futures import Future
from pathlib import Path

from backend.ai_cleaner import AICleanConfig, AICleaner
from backend.srt_parser import parse_srt


def load_main_module():
    fake_multipart = types.ModuleType("multipart")
    fake_multipart.__version__ = "0.0.20"
    fake_multipart_parser = types.ModuleType("multipart.multipart")
    fake_multipart_parser.parse_options_header = lambda value: (value, {})
    sys.modules["multipart"] = fake_multipart
    sys.modules["multipart.multipart"] = fake_multipart_parser

    fake_transcriber_module = types.ModuleType("backend.transcriber")

    class FakeSRTTranscriber:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe_to_srt(self, *args, **kwargs):
            return "1\n00:00:00,000 --> 00:00:01,000\ntest\n"

    fake_transcriber_module.SRTTranscriber = FakeSRTTranscriber
    sys.modules["backend.transcriber"] = fake_transcriber_module
    sys.modules.pop("backend.main", None)
    return importlib.import_module("backend.main")


class FakeAICleanClient:
    def __init__(self, response: str):
        self.response = response

    def clean_blocks(self, blocks: list[dict[str, object]], language: str) -> str:
        return self.response


class BackendMainHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = load_main_module()

    def setUp(self):
        self.main.rate_limit_hits.clear()
        with self.main.jobs_lock:
            self.main.jobs.clear()

    def test_cleanup_old_jobs_expires_finished_jobs_and_marks_stale_running_jobs_error(self):
        now = time.time()
        stale_running_age = self.main.TRANSCRIPTION_TIMEOUT_SECONDS + self.main.STUCK_JOB_GRACE_SECONDS + 1

        with self.main.jobs_lock:
            self.main.jobs.update(
                {
                    "running-old": {
                        "status": "running",
                        "created_at": now - stale_running_age,
                        "updated_at": now - stale_running_age,
                    },
                    "running-active": {
                        "status": "running",
                        "created_at": now,
                        "updated_at": now,
                    },
                    "done-old": {
                        "status": "done",
                        "created_at": now - self.main.JOB_TTL_SECONDS - 1,
                        "updated_at": now - self.main.JOB_TTL_SECONDS - 1,
                    },
                    "error-old": {
                        "status": "error",
                        "created_at": now - self.main.JOB_TTL_SECONDS - 1,
                        "updated_at": now - self.main.JOB_TTL_SECONDS - 1,
                    },
                    "done-fresh": {
                        "status": "done",
                        "created_at": now,
                        "updated_at": now,
                    },
                }
            )

        self.main.cleanup_old_jobs()

        with self.main.jobs_lock:
            self.assertNotIn("done-old", self.main.jobs)
            self.assertNotIn("error-old", self.main.jobs)
            self.assertEqual(self.main.jobs["running-old"]["status"], "error")
            self.assertEqual(
                self.main.jobs["running-old"]["error"],
                "Transcription timed out. Try a shorter audio file.",
            )
            self.assertIn("running-active", self.main.jobs)
            self.assertIn("done-fresh", self.main.jobs)

    def test_cleanup_rate_limit_hits_removes_empty_expired_client_entries(self):
        now = time.monotonic()
        self.main.rate_limit_hits["expired"].append(now - self.main.RATE_LIMIT_WINDOW_SECONDS - 1)
        self.main.rate_limit_hits["active"].append(now)

        self.main.cleanup_rate_limit_hits(now)

        self.assertNotIn("expired", self.main.rate_limit_hits)
        self.assertIn("active", self.main.rate_limit_hits)

    def test_check_rate_limit_rejects_after_limit(self):
        now = time.monotonic()
        for _ in range(self.main.RATE_LIMIT_REQUESTS):
            self.main.rate_limit_hits["client"].append(now)

        with self.assertRaises(self.main.HTTPException) as context:
            self.main.check_rate_limit("client")

        self.assertEqual(context.exception.status_code, 429)

    def test_clear_directory_contents_preserves_gitkeep(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            gitkeep = directory / ".gitkeep"
            gitkeep.write_text("", encoding="utf-8")
            (directory / "one.srt").write_text("x", encoding="utf-8")
            nested = directory / "nested"
            nested.mkdir()
            (nested / "two.srt").write_text("x", encoding="utf-8")

            deleted = self.main.clear_directory_contents(directory)

            self.assertEqual(deleted, 2)
            self.assertTrue(gitkeep.exists())
            self.assertFalse((directory / "one.srt").exists())
            self.assertFalse(nested.exists())

    def test_parse_range_value_accepts_cross_format_time_ranges(self):
        self.assertEqual(self.main.parse_range_value("5-01:02"), (5.0, 62.0))
        self.assertEqual(self.main.parse_range_value("01:02-00:01:03"), (62.0, 63.0))
        self.assertEqual(self.main.parse_range_value("00:00:00,000 --> 00:00:03,500"), (0.0, 3.5))
        self.assertEqual(self.main.parse_range_value("00:00:00:250-00:00:01:500"), (0.25, 1.5))

    def test_model_status_uses_generic_message_on_remote_check_failure(self):
        original_get_cached = self.main.get_cached_model_revision
        original_fetch_remote = self.main.fetch_remote_model_revision
        self.main.get_cached_model_revision = lambda *args, **kwargs: "local-sha"
        self.main.fetch_remote_model_revision = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("/private/path/secret-token")
        )

        try:
            with self.assertLogs(self.main.logger, level="WARNING"):
                result = self.main.model_status()
        finally:
            self.main.get_cached_model_revision = original_get_cached
            self.main.fetch_remote_model_revision = original_fetch_remote

        self.assertEqual(result["status"], "unknown")
        self.assertNotIn("secret-token", result["message"])
        self.assertNotIn("/private/path", result["message"])

    def test_finished_future_does_not_overwrite_timed_out_job(self):
        future = Future()
        future.set_result("1\n00:00:00,000 --> 00:00:01,000\ntest\n")

        class FakeLoop:
            def call_soon_threadsafe(self, callback):
                self.callback = callback

        with self.main.jobs_lock:
            self.main.jobs["job"] = {
                "status": "error",
                "created_at": time.time(),
                "updated_at": time.time(),
                "filename": "audio.mp3",
                "srt_text": "",
                "error": "Transcription timed out. Try a shorter audio file.",
            }

        self.main.finish_transcription_job(
            loop=FakeLoop(),
            job_id="job",
            future=future,
            temp_path=None,
            original_filename="audio.mp3",
            save_output=False,
        )

        with self.main.jobs_lock:
            self.assertEqual(self.main.jobs["job"]["status"], "error")
            self.assertEqual(self.main.jobs["job"]["srt_text"], "")

    def test_clean_srt_endpoint_returns_clean_srt_and_changes(self):
        http_request = types.SimpleNamespace(
            headers={},
            client=types.SimpleNamespace(host="clean-client"),
        )
        request = self.main.CleanSRTRequest(
            srt_text="1\n00:00:00,000 --> 00:00:01,000\n背得滚瓜烂薯 PatternDrill\n",
            language="zh",
        )

        result = self.main.clean_srt(http_request, request)

        self.assertIn("背得滚瓜烂熟 Pattern Drill", result["clean_srt"])
        self.assertTrue(result["changes"])

    def test_clean_srt_endpoint_preserves_indices_and_timing(self):
        raw_srt = """1
00:00:00,000 --> 00:00:02,500
背得滚瓜烂薯 PatternDrill

2
00:00:03,000 --> 00:00:05,250
遇到完全没看过的生殖怎么办
"""
        http_request = types.SimpleNamespace(
            headers={},
            client=types.SimpleNamespace(host="clean-invariant-client"),
        )
        request = self.main.CleanSRTRequest(
            srt_text=raw_srt,
            language="zh",
        )

        result = self.main.clean_srt(http_request, request)

        before_blocks = parse_srt(raw_srt)
        after_blocks = parse_srt(str(result["clean_srt"]))
        self.assertEqual(len(after_blocks), len(before_blocks))
        self.assertEqual([block.index for block in after_blocks], [block.index for block in before_blocks])
        self.assertEqual([block.start for block in after_blocks], [block.start for block in before_blocks])
        self.assertEqual([block.end for block in after_blocks], [block.end for block in before_blocks])
        self.assertIn("背得滚瓜烂熟 Pattern Drill", str(result["clean_srt"]))

    def test_clean_srt_endpoint_rejects_invalid_language(self):
        http_request = types.SimpleNamespace(
            headers={},
            client=types.SimpleNamespace(host="invalid-language-client"),
        )
        request = self.main.CleanSRTRequest(
            srt_text="1\n00:00:00,000 --> 00:00:01,000\ntest\n",
            language="fr",
        )

        with self.assertRaises(self.main.HTTPException) as context:
            self.main.clean_srt(http_request, request)

        self.assertEqual(context.exception.status_code, 400)

    def test_clean_srt_endpoint_is_rate_limited(self):
        http_request = types.SimpleNamespace(
            headers={},
            client=types.SimpleNamespace(host="limited-clean-client"),
        )
        request = self.main.CleanSRTRequest(
            srt_text="1\n00:00:00,000 --> 00:00:01,000\ntest\n",
            language="zh",
        )
        now = time.monotonic()
        for _ in range(self.main.RATE_LIMIT_REQUESTS):
            self.main.rate_limit_hits["limited-clean-client"].append(now)

        with self.assertRaises(self.main.HTTPException) as context:
            self.main.clean_srt(http_request, request)

        self.assertEqual(context.exception.status_code, 429)

    def test_ai_clean_srt_endpoint_returns_ai_clean_srt_and_changes(self):
        self.main.ai_cleaner = AICleaner(
            srt_cleaner=self.main.srt_cleaner,
            config_factory=lambda: AICleanConfig(enabled=True, model="test-model"),
            client_factory=lambda _config: FakeAICleanClient(
                '[{"index": 1, "clean_text": "问答对练。"}]'
            ),
        )
        http_request = types.SimpleNamespace(
            headers={},
            client=types.SimpleNamespace(host="ai-clean-client"),
        )
        request = self.main.AICleanSRTRequest(
            srt_text="1\n00:00:00,000 --> 00:00:01,000\n问答对练\n",
            language="zh",
        )

        result = self.main.ai_clean_srt(http_request, request)

        self.assertTrue(result["ai_used"])
        self.assertIsNone(result["fallback_reason"])
        self.assertIn("问答对练。", result["ai_clean_srt"])
        self.assertTrue(any(change["type"] == "ai_text_correction" for change in result["changes"]))

    def test_ai_clean_srt_endpoint_ai_disabled_by_request(self):
        self.main.ai_cleaner = AICleaner(
            srt_cleaner=self.main.srt_cleaner,
            config_factory=lambda: AICleanConfig(enabled=True, model="test-model"),
            client_factory=lambda _config: FakeAICleanClient("not called"),
        )
        http_request = types.SimpleNamespace(
            headers={},
            client=types.SimpleNamespace(host="ai-clean-disabled-client"),
        )
        request = self.main.AICleanSRTRequest(
            srt_text="1\n00:00:00,000 --> 00:00:01,000\n问答对练\n",
            language="zh",
            ai_enabled=False,
        )

        result = self.main.ai_clean_srt(http_request, request)

        self.assertFalse(result["ai_used"])
        self.assertEqual(result["ai_clean_srt"], result["rule_based_srt"])
        self.assertEqual(result["fallback_reason"], "AI clean disabled by request.")


if __name__ == "__main__":
    unittest.main()
