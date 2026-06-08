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


if __name__ == "__main__":
    unittest.main()
