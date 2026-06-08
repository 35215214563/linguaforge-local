from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)
AI_CLEAN_HINTS_PATH = Path(__file__).resolve().parent / "subtitle_corrections" / "ai_clean_hints.json"


@lru_cache(maxsize=1)
def load_ai_clean_hints() -> dict[str, object]:
    try:
        with AI_CLEAN_HINTS_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("AI clean hints could not be loaded: %s", exc)
        return {}

    if not isinstance(payload, dict):
        logger.warning("AI clean hints file did not contain a JSON object")
        return {}
    return payload


def select_relevant_ai_clean_hints(
        blocks: list[dict[str, object]],
        language: str,
        hints: Mapping[str, object] | None = None,
) -> dict[str, object]:
    hint_source = hints if hints is not None else load_ai_clean_hints()
    block_text = "\n".join(str(block.get("text", "")) for block in blocks)
    normalized_language = (language or "auto").strip().lower()

    correction_hints = select_correction_hints(hint_source, block_text, normalized_language)
    protected_terms = select_protected_terms(hint_source, block_text, correction_hints)
    numeric_rules = select_numeric_rules(hint_source)

    selected: dict[str, object] = {}
    if protected_terms:
        selected["protected_terms"] = protected_terms
    if numeric_rules:
        selected["numeric_rules"] = numeric_rules
    if correction_hints:
        selected["correction_hints"] = correction_hints
    return selected


def select_correction_hints(
        hint_source: Mapping[str, object],
        block_text: str,
        language: str,
) -> list[dict[str, object]]:
    raw_hints = hint_source.get("correction_hints", [])
    if not isinstance(raw_hints, list):
        return []

    selected: list[dict[str, object]] = []
    for raw_hint in raw_hints:
        if not isinstance(raw_hint, dict):
            continue

        wrong = raw_hint.get("wrong")
        if not isinstance(wrong, str) or not wrong or wrong not in block_text:
            continue

        hint_language = str(raw_hint.get("language", "any")).strip().lower() or "any"
        if hint_language not in {"any", language}:
            continue

        selected.append(dict(raw_hint))
    return selected


def select_protected_terms(
        hint_source: Mapping[str, object],
        block_text: str,
        correction_hints: list[dict[str, object]],
) -> list[str]:
    raw_terms = hint_source.get("protected_terms", [])
    if not isinstance(raw_terms, list):
        return []

    suggested_terms = {
        suggest
        for hint in correction_hints
        for suggest in [hint.get("suggest")]
        if isinstance(suggest, str)
    }

    selected: list[str] = []
    for term in raw_terms:
        if not isinstance(term, str) or not term:
            continue
        if term in block_text or term in suggested_terms:
            selected.append(term)
    return selected


def select_numeric_rules(hint_source: Mapping[str, object]) -> dict[str, bool]:
    raw_rules = hint_source.get("numeric_rules", {})
    if not isinstance(raw_rules, dict):
        return {}

    selected = {
        str(rule_name): rule_value
        for rule_name, rule_value in raw_rules.items()
        if isinstance(rule_value, bool)
    }
    if not any(selected.values()):
        return {}
    return selected
