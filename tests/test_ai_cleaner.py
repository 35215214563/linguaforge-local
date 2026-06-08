from __future__ import annotations

import json
import unittest

from backend.ai_cleaner import AICleanConfig, AICleaner, validate_clean_text
from backend.ai_clients import AICleanClientError, AICleanTimeoutError
from backend.srt_parser import parse_srt


RAW_SRT = """1
00:00:00,000 --> 00:00:02,000
问答对练

2
00:00:02,500 --> 00:00:04,000
过去式还是现在式
"""

HINT_SRT = """1
00:00:00,000 --> 00:00:02,000
外教官會對戀練習
"""

NUMERIC_SRT = """1
00:00:00,000 --> 00:00:02,000
第10課ですか?
"""

MIXED_SCRIPT_SRT = """1
00:00:00,000 --> 00:00:03,000
162ページ第20課

2
00:00:03,500 --> 00:00:06,000
자전거를탈수있어요?
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


def ai_items_response(items: list[dict[str, object]]) -> str:
    return json.dumps({"items": items}, ensure_ascii=False)


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
    def test_top_level_list_response_is_accepted(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_response(
                    [
                        {"index": 1, "clean_text": "问答对练。"},
                        {"index": 2, "clean_text": "过去式还是现在式？"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertIn("问答对练。", result.ai_clean_srt)

    def test_object_items_response_is_accepted(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_items_response(
                    [
                        {"index": 1, "clean_text": "问答对练。"},
                        {"index": 2, "clean_text": "过去式还是现在式？"},
                    ]
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertIn("过去式还是现在式？", result.ai_clean_srt)

        before_blocks = parse_srt(RAW_SRT)
        after_blocks = parse_srt(result.ai_clean_srt)
        self.assertEqual(len(after_blocks), len(before_blocks))
        self.assertEqual([block.start for block in after_blocks], [block.start for block in before_blocks])
        self.assertEqual([block.end for block in after_blocks], [block.end for block in before_blocks])

    def test_object_result_response_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                json.dumps(
                    {
                        "result": [
                            {"index": 1, "clean_text": "问答对练。"},
                            {"index": 2, "clean_text": "过去式还是现在式？"},
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        ).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("items list", result.fallback_reason or "")

    def test_object_items_non_list_response_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(json.dumps({"items": "not a list"}, ensure_ascii=False))
        ).clean_srt(RAW_SRT, language="zh")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("items list", result.fallback_reason or "")

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
        self.assertEqual(result.metrics["model"], "test-model")
        self.assertEqual(result.metrics["provider"], "ollama")
        self.assertIn("rule_based_ms", result.metrics)
        self.assertIn("ai_call_ms", result.metrics)
        self.assertIn("validation_ms", result.metrics)
        self.assertIn("total_ms", result.metrics)
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

    def test_ai_output_that_ignores_hint_is_still_accepted(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_items_response(
                    [
                        {"index": 1, "clean_text": "外教官會對戀練習"},
                    ]
                )
            )
        ).clean_srt(HINT_SRT, language="zh")

        self.assertTrue(result.ai_used)
        self.assertIsNone(result.fallback_reason)
        self.assertIn("外教官會對戀練習", result.ai_clean_srt)

    def test_numeric_token_change_is_rejected(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_items_response(
                    [
                        {"index": 1, "clean_text": "第十課ですか?"},
                    ]
                )
            )
        ).clean_srt(NUMERIC_SRT, language="ja")

        self.assertFalse(result.ai_used)
        self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
        self.assertIn("numeric tokens", result.fallback_reason or "")

    def test_full_width_numeric_change_is_rejected(self):
        error = validate_clean_text(
            "第１０課ですか?",
            "第10課ですか?",
        )

        self.assertEqual(error, "AI clean_text changed numeric tokens.")

    def test_numeric_punctuation_token_preserved_is_accepted(self):
        error = validate_clean_text(
            "10.5 分鐘練習",
            "10.5分鐘練習",
        )

        self.assertIsNone(error)

    def test_japanese_block_translated_to_korean_is_rejected(self):
        error = validate_clean_text(
            "자전거를 탈 수 있습니다.",
            "自転車に乗ることができます。",
        )

        self.assertEqual(
            error,
            "AI clean_text removed Japanese kana, likely translating or changing script.",
        )

    def test_japanese_title_translated_to_korean_is_rejected(self):
        error = validate_clean_text(
            "162페이지 제20과",
            "162ページ第20課",
        )

        self.assertEqual(
            error,
            "AI clean_text removed Japanese kana, likely translating or changing script.",
        )

    def test_korean_spacing_fix_is_accepted(self):
        error = validate_clean_text(
            "자전거를 탈 수 있어요?",
            "자전거를탈수있어요?",
        )

        self.assertIsNone(error)

    def test_korean_block_translated_to_japanese_is_rejected(self):
        error = validate_clean_text(
            "自転車に乗れますか？",
            "자전거를탈수있어요?",
        )

        self.assertEqual(
            error,
            "AI clean_text removed Hangul, likely translating or changing script.",
        )

    def test_mixed_srt_falls_back_only_translated_blocks(self):
        result = make_cleaner(
            FakeAICleanClient(
                ai_items_response(
                    [
                        {"index": 1, "clean_text": "162페이지 제20과"},
                        {"index": 2, "clean_text": "자전거를 탈 수 있어요?"},
                    ]
                )
            )
        ).clean_srt(MIXED_SCRIPT_SRT, language="auto")

        self.assertTrue(result.ai_used)
        self.assertEqual(
            result.fallback_reason,
            "Some AI block corrections were rejected; rule-based text was used for those blocks.",
        )

        self.assertIn("162ページ第20課", result.ai_clean_srt)
        self.assertNotIn("162페이지 제20과", result.ai_clean_srt)
        self.assertIn("자전거를 탈 수 있어요?", result.ai_clean_srt)

        fallback_changes = [
            change for change in result.changes
            if change.get("type") == "ai_block_fallback"
        ]
        self.assertEqual(len(fallback_changes), 1)
        self.assertEqual(fallback_changes[0]["index"], 1)
        self.assertIn("Japanese kana", fallback_changes[0]["reason"])

    def test_japanese_spacing_or_punctuation_fix_is_accepted(self):
        error = validate_clean_text(
            "162ページ 第20課",
            "162ページ第20課",
        )

        self.assertIsNone(error)

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

    def test_thinking_output_markers_fall_back_to_rule_based_srt(self):
        for marker in (
            "Thought for",
            "Thinking Process",
            "<think>",
            "</think>",
            "Analyze the Request",
            "Detailed Correction Plan",
        ):
            with self.subTest(marker=marker):
                result = make_cleaner(
                    FakeAICleanClient(
                        marker
                        + "\n"
                        + ai_response(
                            [
                                {"index": 1, "clean_text": "问答对练。"},
                                {"index": 2, "clean_text": "过去式还是现在式？"},
                            ]
                        )
                    )
                ).clean_srt(RAW_SRT, language="zh")

                self.assertFalse(result.ai_used)
                self.assertEqual(result.ai_clean_srt, result.rule_based_srt)
                self.assertIn("thinking output", result.fallback_reason or "")

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
