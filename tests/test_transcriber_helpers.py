from __future__ import annotations

import importlib
import sys
import types
import unittest


def load_transcriber_module():
    fake_faster_whisper = types.ModuleType("faster_whisper")

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    fake_faster_whisper.WhisperModel = FakeWhisperModel

    fake_audio = types.ModuleType("faster_whisper.audio")
    fake_audio.decode_audio = lambda *args, **kwargs: []

    fake_vad = types.ModuleType("faster_whisper.vad")

    class FakeVadOptions:
        def __init__(self, *args, **kwargs):
            pass

    fake_vad.VadOptions = FakeVadOptions
    fake_vad.get_speech_timestamps = lambda *args, **kwargs: []

    sys.modules["faster_whisper"] = fake_faster_whisper
    sys.modules["faster_whisper.audio"] = fake_audio
    sys.modules["faster_whisper.vad"] = fake_vad
    sys.modules.pop("backend.transcriber", None)
    return importlib.import_module("backend.transcriber")


class FakeWord:
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class FakeSegment:
    def __init__(self, text: str, start: float, end: float, words: list[FakeWord]):
        self.text = text
        self.start = start
        self.end = end
        self.words = words


class TranscriberHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transcriber = load_transcriber_module()

    def test_word_timestamps_merge_short_phrases_until_sentence_boundary(self):
        words = [
            self.transcriber.WordTiming(0.0, 0.5, "对，"),
            self.transcriber.WordTiming(0.5, 1.5, "要跳进水池"),
            self.transcriber.WordTiming(1.5, 3.0, "首先得学会换气"),
            self.transcriber.WordTiming(3.0, 4.3, "然后复述一次。"),
        ]

        blocks = self.transcriber.segment_words_into_blocks(words)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "对，要跳进水池首先得学会换气然后复述一次。")
        self.assertGreaterEqual(blocks[0].end - blocks[0].start, self.transcriber.PRO_MIN_BLOCK_SECONDS)

    def test_professional_append_uses_word_timestamps_when_available(self):
        segments = [
            FakeSegment(
                text="对，要跳进水池首先得学会换气然后复述一次。",
                start=0.0,
                end=4.3,
                words=[
                    FakeWord("对，", 0.0, 0.5),
                    FakeWord("要跳进水池", 0.5, 1.5),
                    FakeWord("首先得学会换气", 1.5, 3.0),
                    FakeWord("然后复述一次。", 3.0, 4.3),
                ],
            )
        ]
        output_blocks: list[str] = []

        self.transcriber.append_segments_to_blocks(
            output_blocks,
            segments,
            professional_optimization=True,
        )

        self.assertEqual(len(output_blocks), 1)
        self.assertIn("对，要跳进水池首先得学会换气然后复述一次。", output_blocks[0])

    def test_word_timestamps_do_not_merge_across_long_silence(self):
        words = [
            self.transcriber.WordTiming(0.0, 0.5, "第一句"),
            self.transcriber.WordTiming(2.0, 2.5, "第二句"),
        ]

        blocks = self.transcriber.segment_words_into_blocks(words)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, "第一句")
        self.assertEqual(blocks[1].text, "第二句")

    def test_normalize_subtitle_text_applies_high_confidence_vocabulary_fixes(self):
        self.assertEqual(
            self.transcriber.normalize_subtitle_text("背得滚瓜烂薯"),
            "背得滚瓜烂熟",
        )
        self.assertEqual(
            self.transcriber.normalize_subtitle_text("遇到完全没看过的生殖怎么办"),
            "遇到完全没看过的生词怎么办",
        )
        self.assertEqual(
            self.transcriber.normalize_subtitle_text("有一条铁砾让我印象深刻"),
            "有一条铁律让我印象深刻",
        )

    def test_normalize_subtitle_text_uses_requested_language_dictionary(self):
        self.assertEqual(
            self.transcriber.normalize_subtitle_text("背得滚瓜烂薯", language="zh"),
            "背得滚瓜烂熟",
        )
        self.assertEqual(
            self.transcriber.normalize_subtitle_text("背得滚瓜烂薯", language="ko"),
            "背得滚瓜烂薯",
        )

    def test_wrap_subtitle_text_splits_cjk_and_preserves_english_terms(self):
        text = "这是一个很长的中文句子，里面包含 Pattern Drill 这个英文术语"

        wrapped = self.transcriber.wrap_subtitle_text(text)

        self.assertIn("\n", wrapped)
        self.assertIn("Pattern Drill", wrapped)

    def test_wrap_subtitle_text_does_not_start_line_with_punctuation(self):
        text = "想象一想,你读了十年的游泳手册,把那个换气、"

        wrapped = self.transcriber.wrap_subtitle_text(text)

        self.assertNotIn("\n、", wrapped)
        self.assertNotIn("\n，", wrapped)

    def test_wrap_subtitle_text_does_not_split_embedded_ascii_word(self):
        text = "比如说你想记住procrastinate这个字也就是拖延的意思"

        wrapped = self.transcriber.wrap_subtitle_text(text)

        self.assertNotIn("procrastin\nate", wrapped)
        self.assertIn("procrastinate", wrapped.replace("\n", ""))

    def test_wrap_subtitle_text_handles_long_japanese_sentence(self):
        text = "今日は第10課の会話を確認してから、Pattern Drillで長い文を自然に言えるように練習します。"

        wrapped = self.transcriber.wrap_subtitle_text(text)

        self.assertNotIn("\n、", wrapped)
        self.assertNotIn("\n。", wrapped)
        self.assertIn("Pattern Drill", wrapped)

    def test_wrap_subtitle_text_handles_long_korean_sentence(self):
        text = "오늘은 긴 문장을 천천히 읽고 ChatGPT라는 용어를 유지하면서 자연스럽게 말하는 연습을 합니다."

        wrapped = self.transcriber.wrap_subtitle_text(text)

        self.assertNotIn("Chat\nGPT", wrapped)
        self.assertIn("ChatGPT", wrapped.replace("\n", ""))

    def test_wrap_subtitle_text_handles_long_chinese_sentence_with_terms(self):
        text = "这是一个很长的中文句子，里面包含 procrastinate 和 Pattern Drill 这两个英文术语，所以不能把英文切坏。"

        wrapped = self.transcriber.wrap_subtitle_text(text)

        self.assertNotIn("\n，", wrapped)
        self.assertNotIn("\n。", wrapped)
        self.assertIn("procrastinate", wrapped.replace("\n", ""))
        self.assertIn("Pattern Drill", wrapped)

    def test_word_segmentation_splits_long_japanese_sentence(self):
        words = [
            self.transcriber.WordTiming(0.0, 1.0, "今日は第10課の会話を確認してから、"),
            self.transcriber.WordTiming(1.0, 2.0, "Pattern Drillで長い文を自然に言えるように"),
            self.transcriber.WordTiming(2.0, 3.0, "練習します。"),
            self.transcriber.WordTiming(3.0, 4.0, "次の例文も"),
            self.transcriber.WordTiming(4.0, 5.0, "同じリズムで読みます。"),
        ]

        blocks = self.transcriber.segment_words_into_blocks(words, clean_language="ja")

        self.assertGreaterEqual(len(blocks), 2)
        self.assertTrue(all("Pattern\nDrill" not in block.text for block in blocks))
        self.assertTrue(all(not block.text.startswith(("、", "。")) for block in blocks))

    def test_word_segmentation_splits_long_korean_sentence(self):
        words = [
            self.transcriber.WordTiming(0.0, 1.0, "오늘은 긴 문장을 천천히 읽고 "),
            self.transcriber.WordTiming(1.0, 2.0, "ChatGPT라는 용어를 유지하면서 "),
            self.transcriber.WordTiming(2.0, 3.0, "자연스럽게 말하는 연습을 합니다."),
            self.transcriber.WordTiming(3.0, 4.0, "다음 문장도 "),
            self.transcriber.WordTiming(4.0, 5.0, "같은 속도로 확인합니다."),
        ]

        blocks = self.transcriber.segment_words_into_blocks(words, clean_language="ko")

        self.assertGreaterEqual(len(blocks), 2)
        self.assertTrue(all("Chat\nGPT" not in block.text for block in blocks))
        self.assertIn("ChatGPT", "".join(block.text for block in blocks))

    def test_word_segmentation_splits_long_chinese_sentence_with_terms(self):
        words = [
            self.transcriber.WordTiming(0.0, 1.0, "这是一个很长的中文句子，"),
            self.transcriber.WordTiming(1.0, 2.0, "里面包含procrastinate和Pattern Drill这两个英文术语，"),
            self.transcriber.WordTiming(2.0, 3.0, "所以不能把英文切坏。"),
            self.transcriber.WordTiming(3.0, 4.0, "接下来继续说明"),
            self.transcriber.WordTiming(4.0, 5.0, "字幕应该怎样分段。"),
        ]

        blocks = self.transcriber.segment_words_into_blocks(words, clean_language="zh")

        self.assertGreaterEqual(len(blocks), 2)
        combined = "".join(block.text for block in blocks)
        self.assertIn("procrastinate", combined)
        self.assertIn("Pattern Drill", combined)
        self.assertTrue(all(not block.text.startswith(("，", "。")) for block in blocks))

    def test_word_segmentation_prefers_natural_cjk_boundary_before_hard_cut(self):
        words = [
            self.transcriber.WordTiming(0.0, 0.8, "突然被点名"),
            self.transcriber.WordTiming(0.8, 2.2, "大脑就瞬间当机了这种当机也结束了"),
            self.transcriber.WordTiming(2.2, 3.0, "这是因为"),
            self.transcriber.WordTiming(3.0, 3.6, "神经科"),
            self.transcriber.WordTiming(3.6, 5.0, "学层面的问题并不是单纯努力不足"),
        ]

        blocks = self.transcriber.segment_words_into_blocks(words)

        self.assertGreaterEqual(len(blocks), 2)
        self.assertTrue(blocks[0].text.endswith("了"))
        self.assertNotIn("神经科", blocks[0].text)

    def test_format_srt_time_uses_shared_parser_rounding(self):
        self.assertEqual(
            self.transcriber.format_srt_time(3599.9995),
            self.transcriber.format_srt_time(3600.0),
        )


if __name__ == "__main__":
    unittest.main()
