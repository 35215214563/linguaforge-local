from __future__ import annotations

import importlib
import sys
import tempfile
import time
import types
import unittest
from concurrent.futures import Future
from pathlib import Path


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
        request = self.main.CleanSRTRequest(
            srt_text="1\n00:00:00,000 --> 00:00:01,000\n背得滚瓜烂薯 PatternDrill\n",
            language="zh",
        )

        result = self.main.clean_srt(request)

        self.assertIn("背得滚瓜烂熟 Pattern Drill", result["clean_srt"])
        self.assertTrue(result["changes"])

    def test_clean_srt_endpoint_rejects_invalid_language(self):
        request = self.main.CleanSRTRequest(
            srt_text="1\n00:00:00,000 --> 00:00:01,000\ntest\n",
            language="fr",
        )

        with self.assertRaises(self.main.HTTPException) as context:
            self.main.clean_srt(request)

        self.assertEqual(context.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
