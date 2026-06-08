from __future__ import annotations

import unittest

from backend.srt_parser import SRTValidationError, parse_srt, validate_srt_quality


class SRTParserTests(unittest.TestCase):
    def test_parse_valid_srt(self):
        blocks = parse_srt(
            """1
00:00:00,000 --> 00:00:01,000
Hello

2
00:00:01,200 --> 00:00:02,000
World
"""
        )

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].index, 1)
        self.assertEqual(blocks[0].start, 0.0)
        self.assertEqual(blocks[0].end, 1.0)
        self.assertEqual(blocks[0].text, "Hello")
        self.assertEqual(blocks[1].index, 2)
        self.assertEqual(blocks[1].text, "World")

    def test_parse_empty_srt_raises(self):
        with self.assertRaises(SRTValidationError):
            parse_srt("")

    def test_parse_rejects_non_consecutive_indices(self):
        with self.assertRaises(SRTValidationError):
            parse_srt(
                """1
00:00:00,000 --> 00:00:01,000
A

3
00:00:01,200 --> 00:00:02,000
B
"""
            )

    def test_parse_rejects_start_after_end(self):
        with self.assertRaises(SRTValidationError):
            parse_srt(
                """1
00:00:02,000 --> 00:00:01,000
Bad timing
"""
            )

    def test_parse_rejects_duplicate_indices(self):
        with self.assertRaises(SRTValidationError):
            parse_srt(
                """1
00:00:00,000 --> 00:00:01,000
A

1
00:00:01,200 --> 00:00:02,000
B
"""
            )

    def test_quality_validation_rejects_overlapping_blocks(self):
        blocks = parse_srt(
            """1
00:00:00,000 --> 00:00:02,000
A

2
00:00:01,500 --> 00:00:03,000
B
"""
        )

        with self.assertRaisesRegex(SRTValidationError, "overlaps"):
            validate_srt_quality(blocks)

    def test_quality_validation_rejects_chronological_regression(self):
        blocks = parse_srt(
            """1
00:00:05,000 --> 00:00:06,000
A

2
00:00:03,000 --> 00:00:04,000
B
"""
        )

        with self.assertRaisesRegex(SRTValidationError, "starts before previous"):
            validate_srt_quality(blocks)

    def test_quality_validation_rejects_line_start_punctuation(self):
        blocks = parse_srt(
            """1
00:00:00,000 --> 00:00:03,000
換氣
、
"""
        )

        with self.assertRaisesRegex(SRTValidationError, "punctuation"):
            validate_srt_quality(blocks)

    def test_quality_validation_rejects_split_ascii_word(self):
        blocks = parse_srt(
            """1
00:00:00,000 --> 00:00:03,000
procrastin
ate
"""
        )

        with self.assertRaisesRegex(SRTValidationError, "ASCII word"):
            validate_srt_quality(blocks)


if __name__ == "__main__":
    unittest.main()
