from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from .ai_clients import AICleanClient, AICleanClientConfig, AICleanClientError, OllamaAICleanClient
from .srt_cleaner import CleanSRTResult, SRTCleaner
from .srt_parser import SRTBlock, SRTValidationError, parse_srt, serialize_srt

logger = logging.getLogger(__name__)
DEFAULT_AI_CLEAN_ENABLED = True
DEFAULT_AI_CLEAN_PROVIDER = "ollama"
DEFAULT_AI_CLEAN_BASE_URL = "http://localhost:11434"
DEFAULT_AI_CLEAN_MODEL = "qwen3:8b"
DEFAULT_AI_CLEAN_TIMEOUT_SECONDS = 120.0
DEFAULT_AI_CLEAN_TEMPERATURE = 0.0
DEFAULT_AI_CLEAN_NUM_PREDICT = 2048
DEFAULT_AI_CLEAN_FORMAT_JSON = True
DEFAULT_AI_CLEAN_THINK = False
SHORT_TEXT_LENGTH = 20
VERY_SHORT_TEXT_LENGTH = 4
SHORT_TEXT_LENGTH_MULTIPLIER = 1.6
SHORT_TEXT_LENGTH_EXTRA = 8
LONG_TEXT_LENGTH_MULTIPLIER = 1.35
MIN_TEXT_LENGTH_RATIO = 0.5

MARKDOWN_CODE_FENCE_PATTERN = re.compile(r"```|`+\s*json", re.IGNORECASE)
SRT_TIMESTAMP_PATTERN = re.compile(r"\d{2,}:[0-5]\d:[0-5]\d,\d{3}")
STANDALONE_SRT_INDEX_PATTERN = re.compile(r"(?m)^\s*\d+\s*$")
MULTIPLE_SRT_BLOCKS_PATTERN = re.compile(r"\n\s*\n")
NUMERIC_TOKEN_PATTERN = re.compile(r"[0-9]+(?:[.,:/-][0-9]+)*")
THINKING_OUTPUT_PATTERNS = (
    "Thought for",
    "Thinking Process",
    "<think>",
    "</think>",
    "Analyze the Request",
    "Detailed Correction Plan",
)


@dataclass(frozen=True)
class AICleanConfig:
    enabled: bool = DEFAULT_AI_CLEAN_ENABLED
    provider: str = DEFAULT_AI_CLEAN_PROVIDER
    base_url: str = DEFAULT_AI_CLEAN_BASE_URL
    model: str = DEFAULT_AI_CLEAN_MODEL
    timeout_seconds: float = DEFAULT_AI_CLEAN_TIMEOUT_SECONDS
    temperature: float = DEFAULT_AI_CLEAN_TEMPERATURE
    num_predict: int = DEFAULT_AI_CLEAN_NUM_PREDICT
    format_json: bool = DEFAULT_AI_CLEAN_FORMAT_JSON
    think: bool = DEFAULT_AI_CLEAN_THINK

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None) -> "AICleanConfig":
        source = os.environ if environ is None else environ
        provider = (source.get("AI_CLEAN_PROVIDER") or DEFAULT_AI_CLEAN_PROVIDER).strip().lower()
        base_url = (source.get("AI_CLEAN_BASE_URL") or DEFAULT_AI_CLEAN_BASE_URL).strip().rstrip("/")
        model = (source.get("AI_CLEAN_MODEL") or DEFAULT_AI_CLEAN_MODEL).strip()
        return cls(
            enabled=parse_bool(source.get("AI_CLEAN_ENABLED"), DEFAULT_AI_CLEAN_ENABLED),
            provider=provider or DEFAULT_AI_CLEAN_PROVIDER,
            base_url=base_url or DEFAULT_AI_CLEAN_BASE_URL,
            model=model or DEFAULT_AI_CLEAN_MODEL,
            timeout_seconds=parse_float(
                source.get("AI_CLEAN_TIMEOUT_SECONDS"),
                DEFAULT_AI_CLEAN_TIMEOUT_SECONDS,
            ),
            temperature=parse_float(
                source.get("AI_CLEAN_TEMPERATURE"),
                DEFAULT_AI_CLEAN_TEMPERATURE,
            ),
            num_predict=parse_int(
                source.get("AI_CLEAN_NUM_PREDICT"),
                DEFAULT_AI_CLEAN_NUM_PREDICT,
                minimum=1,
            ),
            format_json=parse_bool(source.get("AI_CLEAN_FORMAT_JSON"), DEFAULT_AI_CLEAN_FORMAT_JSON),
            think=parse_bool(source.get("AI_CLEAN_THINK"), DEFAULT_AI_CLEAN_THINK),
        )

    def to_client_config(self) -> AICleanClientConfig:
        return AICleanClientConfig(
            provider=self.provider,
            base_url=self.base_url,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            temperature=self.temperature,
            num_predict=self.num_predict,
            format_json=self.format_json,
            think=self.think,
        )


@dataclass
class AICleanSRTResult:
    ai_clean_srt: str
    rule_based_srt: str
    changes: list[dict[str, object]] = field(default_factory=list)
    ai_used: bool = False
    fallback_reason: Optional[str] = None
    metrics: dict[str, object] = field(default_factory=dict)


@dataclass
class AIResponseValidation:
    accepted_text_by_index: dict[int, str] = field(default_factory=dict)
    block_errors: dict[int, str] = field(default_factory=dict)
    full_error: Optional[str] = None


ConfigFactory = Callable[[], AICleanConfig]
ClientFactory = Callable[[AICleanConfig], AICleanClient]


class AICleaner:
    def __init__(
            self,
            srt_cleaner: Optional[SRTCleaner] = None,
            config_factory: ConfigFactory = AICleanConfig.from_env,
            client_factory: Optional[ClientFactory] = None,
    ) -> None:
        self.srt_cleaner = srt_cleaner or SRTCleaner()
        self.config_factory = config_factory
        self.client_factory = client_factory or create_ai_clean_client

    def clean_srt(
            self,
            srt_text: str,
            language: str = "auto",
            script: str = "",
            enable_contextual_corrections: bool = False,
            custom_terms: Optional[list[str]] = None,
            ai_enabled: bool = True,
    ) -> AICleanSRTResult:
        total_start = time.perf_counter()
        metrics: dict[str, object] = {
            "ai_call_ms": 0.0,
            "validation_ms": 0.0,
            "model": None,
            "provider": None,
        }
        rule_start = time.perf_counter()
        rule_result = self.srt_cleaner.clean_rule_based(
            srt_text,
            language=language,
            script=script,
            enable_contextual_corrections=enable_contextual_corrections,
            custom_terms=custom_terms or [],
        )
        metrics["rule_based_ms"] = elapsed_ms(rule_start)
        changes = list(rule_result.changes)

        original_blocks, rule_blocks = parse_pipeline_blocks(srt_text, rule_result)
        if original_blocks is None or rule_blocks is None:
            return fallback_result(
                rule_result,
                changes,
                "Input SRT is malformed; AI clean skipped.",
                metrics,
                total_start,
            )

        invariant_error = validate_rule_based_invariants(original_blocks, rule_blocks)
        if invariant_error:
            return fallback_result(rule_result, changes, invariant_error, metrics, total_start)

        config = self.config_factory()
        metrics["model"] = config.model
        metrics["provider"] = config.provider

        if not ai_enabled:
            return fallback_result(
                rule_result,
                changes,
                "AI clean disabled by request.",
                metrics,
                total_start,
            )
        if not config.enabled:
            return fallback_result(
                rule_result,
                changes,
                "AI clean disabled by environment.",
                metrics,
                total_start,
            )

        try:
            client = self.client_factory(config)
            ai_call_start = time.perf_counter()
            try:
                ai_response = client.clean_blocks(blocks_for_ai(rule_blocks), language)
            finally:
                metrics["ai_call_ms"] = elapsed_ms(ai_call_start)
        except Exception as exc:
            logger.warning("AI clean fell back to rule-based SRT: %s", exc)
            return fallback_result(
                rule_result,
                changes,
                str(exc) or "AI clean provider failed.",
                metrics,
                total_start,
            )

        validation_start = time.perf_counter()
        validation = validate_ai_response(ai_response, rule_blocks)
        metrics["validation_ms"] = elapsed_ms(validation_start)
        if validation.full_error:
            return fallback_result(rule_result, changes, validation.full_error, metrics, total_start)

        final_blocks: list[SRTBlock] = []
        ai_change_count = 0
        for original_block, rule_block in zip(original_blocks, rule_blocks):
            ai_text = validation.accepted_text_by_index.get(rule_block.index)
            if ai_text is None:
                final_text = rule_block.text
                if rule_block.index in validation.block_errors:
                    changes.append(
                        {
                            "index": rule_block.index,
                            "type": "ai_block_fallback",
                            "reason": validation.block_errors[rule_block.index],
                        }
                    )
            else:
                final_text = ai_text
                if final_text != rule_block.text:
                    ai_change_count += 1
                    changes.append(
                        {
                            "index": rule_block.index,
                            "before": rule_block.text,
                            "after": final_text,
                            "type": "ai_text_correction",
                        }
                    )

            final_blocks.append(
                SRTBlock(
                    index=original_block.index,
                    start=original_block.start,
                    end=original_block.end,
                    text=final_text,
                )
            )

        try:
            ai_clean_srt = serialize_srt(final_blocks)
        except SRTValidationError as exc:
            return fallback_result(
                rule_result,
                changes,
                f"AI clean SRT validation failed: {exc}",
                metrics,
                total_start,
            )

        final_validation_start = time.perf_counter()
        final_error = validate_final_srt(ai_clean_srt, original_blocks)
        metrics["validation_ms"] = round(
            float(metrics["validation_ms"]) + elapsed_ms(final_validation_start),
            3,
        )
        if final_error:
            return fallback_result(rule_result, changes, final_error, metrics, total_start)

        if validation.accepted_text_by_index:
            fallback_reason = None
            if validation.block_errors:
                fallback_reason = "Some AI block corrections were rejected; rule-based text was used for those blocks."
            return AICleanSRTResult(
                ai_clean_srt=ai_clean_srt,
                rule_based_srt=rule_result.clean_srt,
                changes=changes,
                ai_used=True,
                fallback_reason=fallback_reason,
                metrics=finalize_metrics(metrics, total_start),
            )

        reason = "All AI block corrections were rejected; used rule-based clean SRT."
        if not ai_change_count and validation.block_errors:
            reason = next(iter(validation.block_errors.values()))
        return fallback_result(rule_result, changes, reason, metrics, total_start)


def parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def parse_float(value: Optional[str], default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def parse_int(value: Optional[str], default: int, minimum: int = 0) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed >= minimum else default


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def finalize_metrics(metrics: dict[str, object], total_start: float) -> dict[str, object]:
    finished = dict(metrics)
    finished.setdefault("rule_based_ms", 0.0)
    finished.setdefault("ai_call_ms", 0.0)
    finished.setdefault("validation_ms", 0.0)
    finished.setdefault("model", None)
    finished.setdefault("provider", None)
    finished["total_ms"] = elapsed_ms(total_start)
    return finished


def create_ai_clean_client(config: AICleanConfig) -> AICleanClient:
    if config.provider == "ollama":
        return OllamaAICleanClient(config.to_client_config())
    raise AICleanClientError(f"Unsupported AI clean provider: {config.provider}")


def parse_pipeline_blocks(
        original_srt: str,
        rule_result: CleanSRTResult,
) -> tuple[Optional[list[SRTBlock]], Optional[list[SRTBlock]]]:
    try:
        original_blocks = parse_srt(original_srt, strict=True)
        rule_blocks = parse_srt(rule_result.clean_srt, strict=True)
    except SRTValidationError:
        return None, None
    return original_blocks, rule_blocks


def validate_rule_based_invariants(original_blocks: list[SRTBlock], rule_blocks: list[SRTBlock]) -> Optional[str]:
    if len(original_blocks) != len(rule_blocks):
        return "Rule-based clean SRT changed block count; AI clean skipped."
    if [block.index for block in original_blocks] != [block.index for block in rule_blocks]:
        return "Rule-based clean SRT changed indices; AI clean skipped."
    if [block.start for block in original_blocks] != [block.start for block in rule_blocks]:
        return "Rule-based clean SRT changed start times; AI clean skipped."
    if [block.end for block in original_blocks] != [block.end for block in rule_blocks]:
        return "Rule-based clean SRT changed end times; AI clean skipped."
    return None


def fallback_result(
        rule_result: CleanSRTResult,
        changes: list[dict[str, object]],
        reason: str,
        metrics: Optional[dict[str, object]] = None,
        total_start: Optional[float] = None,
) -> AICleanSRTResult:
    final_metrics = metrics or {}
    if total_start is not None:
        final_metrics = finalize_metrics(final_metrics, total_start)

    return AICleanSRTResult(
        ai_clean_srt=rule_result.clean_srt,
        rule_based_srt=rule_result.clean_srt,
        changes=changes,
        ai_used=False,
        fallback_reason=reason,
        metrics=final_metrics,
    )


def blocks_for_ai(blocks: list[SRTBlock]) -> list[dict[str, object]]:
    return [{"index": block.index, "text": block.text} for block in blocks]


def validate_ai_response(raw_response: str, blocks: list[SRTBlock]) -> AIResponseValidation:
    thinking_error = detect_thinking_output(raw_response)
    if thinking_error:
        return AIResponseValidation(full_error=thinking_error)

    try:
        payload = json.loads(raw_response.strip())
    except json.JSONDecodeError:
        return AIResponseValidation(full_error="AI response was not valid JSON.")

    items, item_error = extract_ai_response_items(payload)
    if item_error:
        return AIResponseValidation(full_error=item_error)
    if items is None:
        return AIResponseValidation(full_error="AI response did not contain subtitle correction items.")

    if len(items) != len(blocks):
        return AIResponseValidation(full_error="AI response block count did not match SRT block count.")

    validation = AIResponseValidation()
    for item, block in zip(items, blocks):
        if not isinstance(item, dict):
            return AIResponseValidation(full_error="AI response item was not an object.")

        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            return AIResponseValidation(full_error="AI response item index was not an integer.")
        if index != block.index:
            return AIResponseValidation(full_error="AI response indices did not match SRT block indices.")

        clean_text = item.get("clean_text")
        if not isinstance(clean_text, str):
            validation.block_errors[block.index] = "AI clean_text was not a string."
            continue

        candidate = clean_text.strip()
        error = validate_clean_text(candidate, block.text)
        if error:
            validation.block_errors[block.index] = error
            continue

        validation.accepted_text_by_index[block.index] = candidate

    return validation


def extract_ai_response_items(payload: object) -> tuple[Optional[list[object]], Optional[str]]:
    if isinstance(payload, list):
        return payload, None

    if isinstance(payload, dict):
        items = payload.get("items")
        if isinstance(items, list):
            return items, None
        return None, "AI response object did not contain a valid items list."

    return None, "AI response top-level value was not a list or an object with an items list."


def validate_clean_text(clean_text: str, reference_text: str) -> Optional[str]:
    if not clean_text:
        return "AI clean_text was empty."
    thinking_error = detect_thinking_output(clean_text)
    if thinking_error:
        return thinking_error
    if MARKDOWN_CODE_FENCE_PATTERN.search(clean_text):
        return "AI clean_text contained markdown code fences."
    if "-->" in clean_text:
        return "AI clean_text contained an SRT timestamp separator."
    if SRT_TIMESTAMP_PATTERN.search(clean_text):
        return "AI clean_text contained an SRT timestamp."
    if MULTIPLE_SRT_BLOCKS_PATTERN.search(clean_text):
        return "AI clean_text appeared to contain multiple SRT blocks."
    if STANDALONE_SRT_INDEX_PATTERN.search(clean_text):
        return "AI clean_text contained a standalone SRT index."
    numeric_error = validate_numeric_tokens(clean_text, reference_text)
    if numeric_error:
        return numeric_error

    script_error = validate_script_preservation(clean_text, reference_text)
    if script_error:
        return script_error

    reference_length = len(reference_text.strip())
    candidate_length = len(clean_text)
    max_length = max_allowed_clean_text_length(reference_length)
    if candidate_length > max_length:
        return "AI clean_text was excessively longer than the rule-based text."
    if reference_length > VERY_SHORT_TEXT_LENGTH and candidate_length < (reference_length * MIN_TEXT_LENGTH_RATIO):
        return "AI clean_text was excessively shorter than the rule-based text."

    return None


def detect_thinking_output(text: str) -> Optional[str]:
    for pattern in THINKING_OUTPUT_PATTERNS:
        if pattern in text:
            return f"AI response contained thinking output marker: {pattern}."
    return None


def validate_numeric_tokens(clean_text: str, reference_text: str) -> Optional[str]:
    reference_tokens = extract_numeric_tokens(reference_text)
    clean_tokens = extract_numeric_tokens(clean_text)
    if clean_tokens != reference_tokens:
        return "AI clean_text changed numeric tokens."
    return None


def validate_script_preservation(clean_text: str, reference_text: str) -> Optional[str]:
    reference_kana = count_kana(reference_text)
    clean_kana = count_kana(clean_text)

    reference_hangul = count_hangul(reference_text)
    clean_hangul = count_hangul(clean_text)

    if reference_kana >= 2 and clean_kana == 0:
        return "AI clean_text removed Japanese kana, likely translating or changing script."

    if reference_hangul == 0 and clean_hangul >= 3:
        return "AI clean_text introduced Hangul into a non-Korean block."

    if reference_hangul >= 3 and clean_hangul == 0:
        return "AI clean_text removed Hangul, likely translating or changing script."

    return None


def count_hangul(text: str) -> int:
    return sum(1 for char in text if "\uAC00" <= char <= "\uD7AF")


def count_hiragana(text: str) -> int:
    return sum(1 for char in text if "\u3040" <= char <= "\u309F")


def count_katakana(text: str) -> int:
    return sum(1 for char in text if "\u30A0" <= char <= "\u30FF")


def count_kana(text: str) -> int:
    return count_hiragana(text) + count_katakana(text)


def count_cjk(text: str) -> int:
    return sum(1 for char in text if "\u4E00" <= char <= "\u9FFF")


def extract_numeric_tokens(text: str) -> list[str]:
    return NUMERIC_TOKEN_PATTERN.findall(text)


def max_allowed_clean_text_length(reference_length: int) -> int:
    if reference_length <= SHORT_TEXT_LENGTH:
        return math.ceil(max(
            reference_length * SHORT_TEXT_LENGTH_MULTIPLIER,
            reference_length + SHORT_TEXT_LENGTH_EXTRA,
        ))
    return math.ceil(reference_length * LONG_TEXT_LENGTH_MULTIPLIER)


def validate_final_srt(final_srt: str, original_blocks: list[SRTBlock]) -> Optional[str]:
    try:
        final_blocks = parse_srt(final_srt, strict=True)
    except SRTValidationError as exc:
        return f"Final AI clean SRT did not parse: {exc}"

    if len(final_blocks) != len(original_blocks):
        return "Final AI clean SRT changed block count."
    if [block.index for block in final_blocks] != [block.index for block in original_blocks]:
        return "Final AI clean SRT changed indices."
    if [block.start for block in final_blocks] != [block.start for block in original_blocks]:
        return "Final AI clean SRT changed start times."
    if [block.end for block in final_blocks] != [block.end for block in original_blocks]:
        return "Final AI clean SRT changed end times."
    return None
