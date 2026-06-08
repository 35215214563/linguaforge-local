from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .srt_parser import SRTBlock, SRTValidationError, parse_srt, serialize_srt, validate_srt_blocks


logger = logging.getLogger(__name__)
CORRECTIONS_DIR = Path(__file__).resolve().parent / "subtitle_corrections"
CJK_CHAR_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
CJK_PUNCTUATION = "、。，．！？!?；;：:"
PUNCTUATION_WITHOUT_LEADING_SPACE = "、。，．！？!?…；;：:,，"


@dataclass
class CleanSRTResult:
    clean_srt: str
    changes: list[dict[str, object]] = field(default_factory=list)


@dataclass
class CorrectionSet:
    safe_replacements: dict[str, str] = field(default_factory=dict)
    contextual_replacements: dict[str, str] = field(default_factory=dict)
    term_replacements: dict[str, str] = field(default_factory=dict)
    terms: list[str] = field(default_factory=list)


class SRTCleaner:
    def __init__(self, corrections_dir: Optional[Path] = None) -> None:
        self.corrections_dir = corrections_dir or CORRECTIONS_DIR

    def clean_rule_based(
        self,
        srt_text: str,
        language: str = "auto",
        script: str = "",
        enable_contextual_corrections: bool = False,
        custom_terms: Optional[list[str]] = None,
    ) -> CleanSRTResult:
        try:
            blocks = parse_srt(srt_text, strict=True)
            corrections = self.load_corrections(language, custom_terms or [])
            cleaned_blocks: list[SRTBlock] = []
            changes: list[dict[str, object]] = []

            for block in blocks:
                cleaned_text, text_changes = self.clean_text(
                    block.text,
                    corrections=corrections,
                    index=block.index,
                    language=language,
                    enable_contextual_corrections=enable_contextual_corrections,
                )
                changes.extend(text_changes)
                cleaned_blocks.append(
                    SRTBlock(index=block.index, start=block.start, end=block.end, text=cleaned_text)
                )

            validate_srt_blocks(cleaned_blocks)
            clean_srt = serialize_srt(cleaned_blocks)
            reparsed_blocks = parse_srt(clean_srt, strict=True)
            if len(reparsed_blocks) != len(blocks):
                raise SRTValidationError("Clean SRT block count changed")

            return CleanSRTResult(clean_srt=clean_srt, changes=changes)
        except Exception as exc:
            logger.warning("Rule-based SRT cleanup fell back to raw SRT: %s", exc)
            return CleanSRTResult(
                clean_srt=srt_text,
                changes=[
                    {
                        "index": None,
                        "before": "",
                        "after": "",
                        "type": "validation_fallback",
                        "message": "Clean SRT validation failed; returned Raw SRT unchanged.",
                    }
                ],
            )

    def clean_with_llm(self, *_args: object, **_kwargs: object) -> CleanSRTResult:
        raise NotImplementedError("LLM cleanup is reserved for a future version.")

    def clean_text(
        self,
        text: str,
        corrections: Optional[CorrectionSet] = None,
        index: Optional[int] = None,
        language: str = "auto",
        enable_contextual_corrections: bool = False,
        custom_terms: Optional[list[str]] = None,
    ) -> tuple[str, list[dict[str, object]]]:
        correction_set = corrections or self.load_corrections(language, custom_terms or [])
        changes: list[dict[str, object]] = []
        current = flatten_subtitle_text(text)

        current = self.apply_replacement_map(
            current,
            correction_set.safe_replacements,
            "safe_replacement",
            index,
            changes,
        )
        current = self.apply_replacement_map(
            current,
            correction_set.term_replacements,
            "term_replacement",
            index,
            changes,
        )
        if enable_contextual_corrections:
            current = self.apply_replacement_map(
                current,
                correction_set.contextual_replacements,
                "contextual_replacement",
                index,
                changes,
            )

        current = apply_format_normalization(current, index=index, changes=changes, language=language)
        return current.strip(), changes

    def load_corrections(self, language: str = "auto", custom_terms: Optional[list[str]] = None) -> CorrectionSet:
        normalized_language = (language or "auto").strip().lower()
        correction_files = ["language_learning_terms.json"]

        if normalized_language == "zh":
            correction_files.insert(0, "common_zh.json")
        elif normalized_language in {"ja", "ko"}:
            correction_files.insert(0, f"common_{normalized_language}.json")

        merged = CorrectionSet()
        for filename in correction_files:
            self.merge_correction_file(merged, self.corrections_dir / filename)

        for term in custom_terms or []:
            normalized_term = flatten_subtitle_text(str(term))
            if normalized_term and normalized_term not in merged.terms:
                merged.terms.append(normalized_term)

        add_generated_term_replacements(merged)
        return merged

    def merge_correction_file(self, merged: CorrectionSet, path: Path) -> None:
        if not path.is_file():
            return

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Failed to load subtitle correction file: %s", path, exc_info=True)
            return

        merged.safe_replacements.update(get_string_map(payload, "safe_replacements"))
        merged.contextual_replacements.update(get_string_map(payload, "contextual_replacements"))
        merged.term_replacements.update(get_string_map(payload, "term_replacements"))

        for term in payload.get("terms", []):
            if isinstance(term, str) and term and term not in merged.terms:
                merged.terms.append(term)

    def apply_replacement_map(
        self,
        text: str,
        replacements: dict[str, str],
        change_type: str,
        index: Optional[int],
        changes: list[dict[str, object]],
    ) -> str:
        current = text
        for before, after in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            if not before or before == after or before not in current:
                continue

            updated = replace_text_safely(current, before, after)
            if updated == current:
                continue

            current = updated
            changes.append(
                {
                    "index": index,
                    "before": before,
                    "after": after,
                    "type": change_type,
                }
            )

        return current


def get_string_map(payload: dict[str, Any], key: str) -> dict[str, str]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        return {}

    result: dict[str, str] = {}
    for before, after in value.items():
        if isinstance(before, str) and isinstance(after, str) and before:
            result[before] = after
    return result


def add_generated_term_replacements(corrections: CorrectionSet) -> None:
    for term in corrections.terms:
        normalized_term = flatten_subtitle_text(term)
        compact_term = re.sub(r"\s+", "", normalized_term)
        if compact_term and compact_term != normalized_term:
            corrections.term_replacements.setdefault(compact_term, normalized_term)


def flatten_subtitle_text(text: str) -> str:
    current = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not current:
        return ""

    current = re.sub(r"[ \t\f\v]+", " ", current)
    current = re.sub(
        r" *\n+ *",
        lambda match: subtitle_line_break_replacement(current, match.start(), match.end()),
        current,
    )
    current = re.sub(r" {2,}", " ", current)
    current = re.sub(rf"\s+([{re.escape(PUNCTUATION_WITHOUT_LEADING_SPACE)}])", r"\1", current)
    return current.strip()


def subtitle_line_break_replacement(source: str, start: int, end: int) -> str:
    left = previous_non_space_char(source, start)
    right = next_non_space_char(source, end)
    if not left or not right:
        return ""

    if should_join_across_subtitle_line_break(left, right):
        return ""

    return " "


def previous_non_space_char(source: str, index: int) -> str:
    for position in range(index - 1, -1, -1):
        if not source[position].isspace():
            return source[position]
    return ""


def next_non_space_char(source: str, index: int) -> str:
    for position in range(index, len(source)):
        if not source[position].isspace():
            return source[position]
    return ""


def should_join_across_subtitle_line_break(left: str, right: str) -> bool:
    if right in PUNCTUATION_WITHOUT_LEADING_SPACE:
        return True
    if left in "（([「『【" or right in "）」』】)]":
        return True
    if is_cjk_char(left) or is_cjk_char(right):
        return True
    if left.isascii() and right.isascii() and left.isalnum() and right.isalnum():
        return True
    return False


def is_cjk_char(value: str) -> bool:
    return bool(CJK_CHAR_PATTERN.match(value))


def replace_text_safely(text: str, before: str, after: str) -> str:
    if needs_word_boundary(before):
        return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(before)}(?![A-Za-z0-9_])", after, text)

    return text.replace(before, after)


def needs_word_boundary(value: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Za-z0-9_+-]+", value)
        and len(value) <= 6
    )


def apply_format_normalization(
    text: str,
    index: Optional[int],
    changes: list[dict[str, object]],
    language: str = "auto",
) -> str:
    current = text
    patterns = [
        (r"(\d+)\s*[%％]\s*(ページ|頁)", r"\1\2"),
        (r"第\s*(\d+)\s*課", r"第\1課"),
        (r"第\s+(\d+)", r"第\1"),
        (r"(\d+)\s+(ページ|頁|課|番)", r"\1\2"),
        (r"\s+([、。，．！？!?])", r"\1"),
    ]

    for pattern, replacement in patterns:
        updated = re.sub(pattern, replacement, current)
        if updated != current:
            changes.append(
                {
                    "index": index,
                    "before": current,
                    "after": updated,
                    "type": "format_normalization",
                }
            )
            current = updated

    if should_normalize_chinese_punctuation(language):
        updated = normalize_cjk_punctuation(current)
        if updated != current:
            changes.append(
                {
                    "index": index,
                    "before": current,
                    "after": updated,
                    "type": "punctuation_normalization",
                }
            )
            current = updated

    return current


def should_normalize_chinese_punctuation(language: str) -> bool:
    return (language or "auto").strip().lower() == "zh"


def normalize_cjk_punctuation(text: str) -> str:
    current = re.sub(r"(\S)\s*,\s*(\S)", normalize_comma_match, text)
    current = re.sub(r"(\S)\s*,\s*$", normalize_trailing_comma_match, current)
    current = re.sub(rf"(?<={CJK_CHAR_PATTERN.pattern})\?", "？", current)
    current = re.sub(rf"(?<={CJK_CHAR_PATTERN.pattern})!", "！", current)
    return current


def normalize_comma_match(match: re.Match[str]) -> str:
    left = match.group(1)
    right = match.group(2)
    if is_cjk_char(left) or is_cjk_char(right):
        return f"{left}，{right}"
    return match.group(0)


def normalize_trailing_comma_match(match: re.Match[str]) -> str:
    left = match.group(1)
    if is_cjk_char(left):
        return f"{left}，"
    return match.group(0)


DEFAULT_CLEANER = SRTCleaner()


def clean_subtitle_text(
    text: str,
    language: str = "auto",
    enable_contextual_corrections: bool = False,
    custom_terms: Optional[list[str]] = None,
) -> str:
    cleaned_text, _changes = DEFAULT_CLEANER.clean_text(
        text,
        language=language,
        enable_contextual_corrections=enable_contextual_corrections,
        custom_terms=custom_terms,
    )
    return cleaned_text
