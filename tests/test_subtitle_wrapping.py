from __future__ import annotations

import unittest

from test_transcriber_helpers import load_transcriber_module


class SubtitleWrappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transcriber = load_transcriber_module()

    def assert_phrase_not_split(self, wrapped: str, phrase: str) -> None:
        self.assertIn(phrase, wrapped.replace("\n", ""))
        for split_at in range(1, len(phrase)):
            self.assertNotIn(f"{phrase[:split_at]}\n{phrase[split_at:]}", wrapped)

    def test_wrap_does_not_put_punctuation_on_new_line(self):
        wrapped = self.transcriber.wrap_subtitle_text(
            "想象一想，你读了十年的游泳手册，把那个换气、"
        )

        for punctuation in "、。，．！？!?…；;：:,，":
            self.assertNotIn(f"\n{punctuation}", wrapped)

    def test_wrap_does_not_split_common_english_words(self):
        samples = [
            "比如说你想记住procrastinate这个字也就是拖延的意思",
            "现在我们可以把Gemini或者ChatGPT当作严格的教官",
            "就算你脑袋一片空白也要先丢出Honestly, I think这种起手式",
        ]

        for text in samples:
            with self.subTest(text=text):
                wrapped = self.transcriber.wrap_subtitle_text(text)
                collapsed = wrapped.replace("\n", "")
                self.assertIn("procrastinate", collapsed) if "procrastinate" in text else None
                self.assertIn("Gemini", collapsed) if "Gemini" in text else None
                self.assertIn("Honestly, I think", collapsed) if "Honestly" in text else None
                self.assertNotRegex(wrapped, r"[A-Za-z]{2,}\n[A-Za-z]{2,}")

    def test_wrap_does_not_split_protected_cjk_phrases(self):
        samples = {
            "深入探讨": "这是一个很长的中文句子里面包含深入探讨这个固定词。",
            "空间路径": "我们会把抽象语言画面绑定在非常熟悉的空间路径上。",
            "自动反击": "真正对战的时候身体会形成自动反击不用停下来想。",
            "十分钟对话": "每天打开AI语音强制自己进行十分钟对话训练。",
            "记忆宫殿法": "这时候可以使用记忆宫殿法来处理抽象词汇。",
            "可理解性输入": "黄金比例在语言学上叫做可理解性输入。",
        }

        for phrase, text in samples.items():
            with self.subTest(phrase=phrase):
                wrapped = self.transcriber.wrap_subtitle_text(text)
                self.assert_phrase_not_split(wrapped, phrase)

    def test_wrap_preserves_mixed_phrase_spacing(self):
        wrapped = self.transcriber.wrap_subtitle_text(
            "Honestly, I think this is Pattern Drill for language practice."
        )

        self.assertIn("Honestly, I think", wrapped)
        self.assertIn("Pattern Drill", wrapped)
        self.assertNotIn("thinkPattern", wrapped)


if __name__ == "__main__":
    unittest.main()
