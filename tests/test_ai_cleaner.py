from __future__ import annotations

import json
import unittest

from backend.ai_cleaner import AICleanConfig, AICleaner
from backend.ai_clients import AICleanClientError, AICleanTimeoutError
from backend.srt_parser import parse_srt


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
        self.blocks: list[dict[str, object]] = []

    def clean_blocks(self, blocks: list[dict[str, object]], language: str) -> str:
        self.blocks = blocks
        if self.error:
            raise self.error
        return self.response


def ai_response(items: list[dict[str, object]]) -> str:
    return json.dumps(items, ensure_ascii=False)


def make_cleaner(
    client: FakeAICleanClient,
    enabled: bool = True,
    provider: str = "ollama",
    model: str = "test-model",
) -> AICleaner:
    return AICleaner(
        config_factory=lambda: AICleanConfig(
            enabled=enabled,
            provider=provider,
            model=model,
        ),
        client_factory=lambda _config: client,
    )


class AICleanerTests(unittest.TestCase):
    def test_valid_ai_clean_preserves_srt_invariants_and_records_changes(self):
        client = FakeAICleanClient(
            ai_response(
                [
                    {"index": 1, "clean_text": "问答对练。"},
                    {"index": 2, "clean_text": "过去式还是现在式？"},
                ]
            )
        )
        result = make_cleaner(client).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertIsNone(result.fallback_reason)
        self.assertIn("问答对练。", result.ai_clean_srt)
        self.assertTrue(any(change["type"] == "ai_text_correction" for change in result.changes))

        before_blocks = parse_srt(RAW_SRT)
        after_blocks = parse_srt(result.ai_clean_srt)
        self.assertEqual(len(after_blocks), len(before_blocks))
        self.assertEqual([block.index for block in after_blocks], [block.index for block in before_blocks])
        self.assertEqual([block.start for block in after_blocks], [block.start for block in before_blocks])
        self.assertEqual([block.end for block in after_blocks], [block.end for block in before_blocks])

        self.assertEqual(
            client.blocks,
            [
                {"index": 1, "text": "问答对练"},
                {"index": 2, "text": "过去式还是现在式"},
            ],
        )

    def test_request_disabled_falls_back_to_rule_based_srt(self):
        client = FakeAICleanClient("not called")
        result = make_cleaner(client).clean_srt(RAW_SRT, language="zh", ai_enabled=False)

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertEqual(client.blocks, [])
        self.assertEqual(result.fallback_reason, "AI clean disabled by request.")

    def test_env_disabled_falls_back_to_rule_based_srt(self):
        client = FakeAICleanClient("not called")
        result = make_cleaner(client, enabled=False).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertEqual(client.blocks, [])
        self.assertEqual(result.fallback_reason, "AI clean disabled by environment.")

    def test_unavailable_provider_falls_back_to_rule_based_srt(self):
        client = FakeAICleanClient(error=AICleanClientError("provider unavailable"))
        result = make_cleaner(client).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("provider unavailable", result.fallback_reason or "")

    def test_timeout_falls_back_to_rule_based_srt(self):
        client = FakeAICleanClient(error=AICleanTimeoutError("timeout"))
        result = make_cleaner(client).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("timeout", result.fallback_reason or "")

    def test_invalid_json_falls_back_to_rule_based_srt(self):
        result = make_cleaner(FakeAICleanClient("not json")).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("valid JSON", result.fallback_reason or "")

    def test_missing_block_falls_back_to_rule_based_srt(self):
        result = make_cleaner(
            FakeAICleanClient(ai_response([{"index": 1, "clean_text": "问答对练。"}]))
        ).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("block count", result.fallback_reason or "")

    def test_wrong_index_falls_back_to_rule_based_srt(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 2, "clean_text": "问答对练。"},
                        {"index": 2, "clean_text": "过去式还是现在式？"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("indices", result.fallback_reason or "")

    def test_empty_clean_text_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": ""},
                        {"index": 2, "clean_text": ""},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("empty", result.fallback_reason or "")

    def test_full_srt_text_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {
                            "index": 1,
                            "clean_text": "1\n00:00:00,000 --> 00:00:01,000\n问答对练",
                        },
                        {"index": 2, "clean_text": "过去式还是现在式"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertIn("过去式还是现在式", result.ai_clean_srt)
        self.assertTrue(any(change["type"] == "ai_block_fallback" for change in result.changes))

    def test_markdown_code_fence_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": "`json []`"},
                        {"index": 2, "clean_text": "过去式还是现在式"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertTrue(any(change["type"] == "ai_block_fallback" for change in result.changes))

    def test_timestamp_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": "00:00:00,000"},
                        {"index": 2, "clean_text": "过去式还是现在式"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertTrue(any(change["type"] == "ai_block_fallback" for change in result.changes))

    def test_excessively_long_text_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": "问答对练" * 10},
                        {"index": 2, "clean_text": "过去式还是现在式"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertTrue(any(change["type"] == "ai_block_fallback" for change in result.changes))

    def test_excessively_short_text_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": "问答对练"},
                        {"index": 2, "clean_text": "过"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertTrue(any(change["type"] == "ai_block_fallback" for change in result.changes))

    def test_malformed_srt_falls_back_safely(self):
        result = make_cleaner(FakeAICleanClient("not called")).clean_srt("not an srt", language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, "not an srt")
        self.assertIn("malformed", result.fallback_reason or "")


if __name__ == "__main__":
    unittest.main()
