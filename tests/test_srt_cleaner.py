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
外教官會對戀練習
"""

        default_result = self.cleaner.clean_rule_based(raw, language="zh")
        contextual_result = self.cleaner.clean_rule_based(
            raw,
            language="zh",
            enable_contextual_corrections=True,
        )

        self.assertIn("外教官會對戀練習", default_result.clean_srt)
        self.assertIn("外交官會對練練習", contextual_result.clean_srt)

    def test_custom_terms_generate_compact_term_replacements(self):
        raw = """1
00:00:00,000 --> 00:00:02,000
LanguageReactor 可以用
"""

        result = self.cleaner.clean_rule_based(raw, language="en", custom_terms=["Language Reactor"])

        self.assertIn("Language Reactor 可以用", result.clean_srt)
        self.assertTrue(any(change["type"] == "term_replacement" for change in result.changes))

    def test_short_ascii_replacements_use_word_boundaries(self):
        updated = self.cleaner.apply_replacement_map(
            "en sentence en",
            {"en": "EN"},
            "term_replacement",
            1,
            [],
        )

        self.assertEqual(updated, "EN sentence EN")

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
        self.assertEqual(
            clean_subtitle_text("记忆供电法和Chat GPT", language="zh"),
            "记忆宫殿法和ChatGPT",
        )
        self.assertEqual(
            clean_subtitle_text("记忆供电法和Chat GPT"),
            "记忆供电法和ChatGPT",
        )

    def test_cleaner_repairs_visual_line_wraps_before_replacements(self):
        raw = """1
00:00:00,000 --> 00:00:03,000
把那个换气
、

2
00:00:03,100 --> 00:00:06,000
记
忆供电法和Ho
nestly,
I
think

3
00:00:06,100 --> 00:00:09,000
遇到老外点餐或者开会
的时候,对吧?
"""

        result = self.cleaner.clean_rule_based(raw, language="zh")

        self.assertIn("把那个换气、", result.clean_srt)
        self.assertNotIn("换气\n、", result.clean_srt)
        self.assertIn("记忆宫殿法和Honestly, I think", result.clean_srt)
        self.assertIn("遇到老外点餐或者开会的时候，对吧？", result.clean_srt)
        self.assertNotIn("记 忆供电法", result.clean_srt)

    def test_non_chinese_languages_do_not_apply_chinese_safe_replacements(self):
        raw = """1
00:00:00,000 --> 00:00:02,000
背得滚瓜烂薯 記憶供電法 很有人眼 PatternDrill
"""

        for language in ("ja", "ko", "en", "vi"):
            with self.subTest(language=language):
                result = self.cleaner.clean_rule_based(raw, language=language)

                self.assertIn("背得滚瓜烂薯", result.clean_srt)
                self.assertIn("記憶供電法", result.clean_srt)
                self.assertIn("很有人眼", result.clean_srt)
                self.assertIn("Pattern Drill", result.clean_srt)
                self.assertNotIn("滚瓜烂熟", result.clean_srt)
                self.assertNotIn("記憶宮殿法", result.clean_srt)
                self.assertNotIn("很有吸引力", result.clean_srt)

    def test_japanese_and_korean_text_keep_native_punctuation_and_terms(self):
        raw = """1
00:00:00,000 --> 00:00:02,000
第10課ですか? PatternDrill

2
00:00:02,100 --> 00:00:04,000
몇 시부터 해요? Chat GPT
"""

        ja_result = self.cleaner.clean_rule_based(raw, language="ja")
        ko_result = self.cleaner.clean_rule_based(raw, language="ko")

        self.assertIn("第10課ですか?", ja_result.clean_srt)
        self.assertIn("몇 시부터 해요?", ko_result.clean_srt)
        self.assertIn("Pattern Drill", ja_result.clean_srt)
        self.assertIn("ChatGPT", ko_result.clean_srt)
        self.assertNotIn("第10課ですか？", ja_result.clean_srt)
        self.assertNotIn("몇 시부터 해요？", ko_result.clean_srt)

    def test_auto_and_mixed_do_not_apply_chinese_dictionary(self):
        raw = """1
00:00:00,000 --> 00:00:02,000
背得滚瓜烂薯 PatternDrill
"""

        for language in ("auto", "mixed"):
            with self.subTest(language=language):
                result = self.cleaner.clean_rule_based(raw, language=language)

                self.assertIn("背得滚瓜烂薯", result.clean_srt)
                self.assertIn("Pattern Drill", result.clean_srt)
                self.assertNotIn("背得滚瓜烂熟", result.clean_srt)

        zh_result = self.cleaner.clean_rule_based(raw, language="zh")
        self.assertIn("背得滚瓜烂熟", zh_result.clean_srt)

    def test_long_non_chinese_clean_srt_does_not_apply_chinese_punctuation_or_vocabulary(self):
        raw = """1
00:00:00,000 --> 00:00:06,000
今日は第10課の会話を確認してから、PatternDrillで長い文を自然に言えるように練習します?

2
00:00:06,100 --> 00:00:12,000
오늘은 긴 문장을 천천히 읽고 Chat GPT 대신 ChatGPT라는 용어도 그대로 확인해요?

3
00:00:12,100 --> 00:00:18,000
This is a long English subtitle with whatImeanis and Ithink, but it should not receive Chinese vocabulary fixes.
"""

        ja_result = self.cleaner.clean_rule_based(raw, language="ja")
        ko_result = self.cleaner.clean_rule_based(raw, language="ko")
        en_result = self.cleaner.clean_rule_based(raw, language="en")

        self.assertIn("今日は第10課の会話を確認してから、Pattern Drill", ja_result.clean_srt)
        self.assertIn("練習します?", ja_result.clean_srt)
        self.assertNotIn("練習します？", ja_result.clean_srt)
        self.assertIn("오늘은 긴 문장을 천천히 읽고 ChatGPT", ko_result.clean_srt)
        self.assertIn("확인해요?", ko_result.clean_srt)
        self.assertNotIn("확인해요？", ko_result.clean_srt)
        self.assertIn("what I mean is and I think", en_result.clean_srt)
        self.assertNotIn("滚瓜烂熟", ja_result.clean_srt + ko_result.clean_srt + en_result.clean_srt)

    def test_auto_and_mixed_do_not_apply_chinese_punctuation_normalization(self):
        raw = """1
00:00:00,000 --> 00:00:02,000
今天要練習嗎?
"""

        auto_result = self.cleaner.clean_rule_based(raw, language="auto")
        mixed_result = self.cleaner.clean_rule_based(raw, language="mixed")
        zh_result = self.cleaner.clean_rule_based(raw, language="zh")

        self.assertIn("今天要練習嗎?", auto_result.clean_srt)
        self.assertIn("今天要練習嗎?", mixed_result.clean_srt)
        self.assertIn("今天要練習嗎？", zh_result.clean_srt)

    def test_cleaner_repairs_language_learning_sample_errors(self):
        raw = """1
00:00:00,000 --> 00:00:04,000
叫做70%懂,30%考材

2
00:00:04,100 --> 00:00:08,000
放弃那种死机硬背吧, 单负数有没有加S了?

3
00:00:08,100 --> 00:00:12,000
核心训练叫做巨型替换训练, 这种七手式很有人眼

4
00:00:12,100 --> 00:00:16,000
李柔不断替换whatImeanis后面的词汇
"""

        result = self.cleaner.clean_rule_based(raw, language="zh")

        self.assertIn("30%挑战", result.clean_srt)
        self.assertIn("死记硬背", result.clean_srt)
        self.assertIn("单复数", result.clean_srt)
        self.assertIn("句型替换训练", result.clean_srt)
        self.assertIn("起手式", result.clean_srt)
        self.assertIn("很有吸引力", result.clean_srt)
        self.assertIn("例如不断替换what I mean is后面的词汇", result.clean_srt)


if __name__ == "__main__":
    unittest.main()
