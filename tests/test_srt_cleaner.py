from __future__ import annotations

import unittest

from backend.srt_cleaner import SRTCleaner, clean_subtitle_text
from backend.srt_parser import SRTValidationError, parse_srt


RAW_SRT = """1
00:00:00,000 --> 00:00:02,000
背得滚瓜烂薯，PatternDrill

2
00:00:02,100 --> 00:00:04,000
whatImeanis 遇到完全没看过的生殖怎么办
"""


class SRTCleanerTests(unittest.TestCase):
    def setUp(self):
        self.cleaner = SRTCleaner()

    def test_rule_based_clean_applies_safe_and_term_replacements(self):
        result = self.cleaner.clean_rule_based(RAW_SRT, language="zh")

        self.assertIn("背得滚瓜烂熟，Pattern Drill", result.clean_srt)
        self.assertIn("what I mean is 遇到完全没看过的生词怎么办", result.clean_srt)
        self.assertTrue(any(change["type"] == "safe_replacement" for change in result.changes))
        self.assertTrue(any(change["type"] == "term_replacement" for change in result.changes))
        self.assertEqual(len(parse_srt(result.clean_srt)), 2)

    def test_contextual_replacements_are_disabled_by_default(self):
        raw = """1
00:00:00,000 --> 00:00:02,000
這真的很有人眼
"""

        default_result = self.cleaner.clean_rule_based(raw, language="zh")
        contextual_result = self.cleaner.clean_rule_based(
            raw,
            language="zh",
            enable_contextual_corrections=True,
        )

        self.assertIn("很有人眼", default_result.clean_srt)
        self.assertIn("很有吸引力", contextual_result.clean_srt)

    def test_custom_terms_generate_compact_term_replacements(self):
        raw = """1
00:00:00,000 --> 00:00:02,000
LanguageReactor 可以用
"""

        result = self.cleaner.clean_rule_based(raw, language="en", custom_terms=["Language Reactor"])

        self.assertIn("Language Reactor 可以用", result.clean_srt)
        self.assertTrue(any(change["type"] == "term_replacement" for change in result.changes))

    def test_invalid_srt_falls_back_to_raw_text(self):
        raw = """1
00:00:02,000 --> 00:00:01,000
bad timing
"""

        result = self.cleaner.clean_rule_based(raw, language="zh")

        self.assertEqual(result.clean_srt, raw)
        self.assertEqual(result.changes[0]["type"], "validation_fallback")

    def test_parse_srt_rejects_non_consecutive_indices(self):
        raw = """2
00:00:00,000 --> 00:00:01,000
bad index
"""

        with self.assertRaises(SRTValidationError):
            parse_srt(raw)

    def test_transcriber_shared_text_cleanup_uses_external_dictionary(self):
        self.assertEqual(clean_subtitle_text("记忆供电法和Chat GPT"), "记忆宫殿法和ChatGPT")


if __name__ == "__main__":
    unittest.main()
