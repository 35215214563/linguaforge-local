from __future__ import annotations

import unittest
from unittest.mock import mock_open, patch

from backend.ai_clean_hints import load_ai_clean_hints, select_relevant_ai_clean_hints
from backend.ai_clients.ollama_client import build_ai_clean_prompt


class AICleanHintTests(unittest.TestCase):
    def test_relevant_hint_selected_when_wrong_text_exists(self):
        selected = select_relevant_ai_clean_hints(
            [{"index": 1, "text": "今天做問答對聯練習"}],
            "zh",
        )

        wrong_terms = {hint["wrong"] for hint in selected.get("correction_hints", [])}
        self.assertIn("問答對聯", wrong_terms)

    def test_irrelevant_hint_not_selected(self):
        selected = select_relevant_ai_clean_hints(
            [{"index": 1, "text": "今天做普通口說練習"}],
            "zh",
        )

        self.assertNotIn("correction_hints", selected)

    def test_language_specific_hint_selected_only_for_matching_language_or_any(self):
        blocks = [{"index": 1, "text": "對戀 和 PatternDrill 都出現了"}]

        zh_selected = select_relevant_ai_clean_hints(blocks, "zh")
        en_selected = select_relevant_ai_clean_hints(blocks, "en")

        zh_wrong_terms = {hint["wrong"] for hint in zh_selected.get("correction_hints", [])}
        en_wrong_terms = {hint["wrong"] for hint in en_selected.get("correction_hints", [])}
        self.assertIn("對戀", zh_wrong_terms)
        self.assertIn("PatternDrill", zh_wrong_terms)
        self.assertNotIn("對戀", en_wrong_terms)
        self.assertIn("PatternDrill", en_wrong_terms)

    def test_protected_term_included_when_present(self):
        selected = select_relevant_ai_clean_hints(
            [{"index": 1, "text": "ChatGPT 和 FSI 都是字幕裡的術語"}],
            "zh",
        )

        protected_terms = selected.get("protected_terms", [])
        self.assertIn("ChatGPT", protected_terms)
        self.assertIn("FSI", protected_terms)

    def test_numeric_rules_included(self):
        selected = select_relevant_ai_clean_hints(
            [{"index": 1, "text": "第10課練習"}],
            "zh",
        )

        numeric_rules = selected.get("numeric_rules", {})
        self.assertTrue(numeric_rules.get("preserve_arabic_digits"))
        self.assertTrue(numeric_rules.get("do_not_convert_digits_to_kanji"))
        self.assertTrue(numeric_rules.get("do_not_convert_half_width_to_full_width"))

    def test_prompt_states_hints_are_advisory_not_mandatory(self):
        prompt = build_ai_clean_prompt(
            [{"index": 1, "text": "今天做問答對聯 10 次"}],
            "zh",
        )

        self.assertIn("advisory, not mandatory replacements", prompt)
        self.assertIn("Use them only when the current context strongly supports the correction.", prompt)
        self.assertIn("If unsure, keep the original text.", prompt)
        self.assertIn("Hard rules such as numeric preservation must always be followed.", prompt)
        self.assertIn("問答對聯", prompt)

    def test_prompt_does_not_include_unrelated_correction_hints(self):
        prompt = build_ai_clean_prompt(
            [{"index": 1, "text": "今天做問答對聯 10 次"}],
            "zh",
        )

        self.assertIn("問答對聯", prompt)
        self.assertNotIn("whatImeanis", prompt)
        self.assertNotIn("HonestlyIthink", prompt)

    def test_suggested_protected_term_is_included_when_hint_is_selected(self):
        selected = select_relevant_ai_clean_hints(
            [{"index": 1, "text": "今天要做 PatternDrill"}],
            "en",
        )

        self.assertIn("Pattern Drill", selected.get("protected_terms", []))

    def test_load_ai_clean_hints_returns_empty_for_non_object_json(self):
        load_ai_clean_hints.cache_clear()
        try:
            with patch("pathlib.Path.open", mock_open(read_data='["not", "an", "object"]')):
                self.assertEqual(load_ai_clean_hints(), {})
        finally:
            load_ai_clean_hints.cache_clear()

    def test_load_ai_clean_hints_returns_empty_for_invalid_json(self):
        load_ai_clean_hints.cache_clear()
        try:
            with patch("pathlib.Path.open", mock_open(read_data="{not json")):
                self.assertEqual(load_ai_clean_hints(), {})
        finally:
            load_ai_clean_hints.cache_clear()


if __name__ == "__main__":
    unittest.main()
