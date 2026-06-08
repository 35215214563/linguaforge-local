from __future__ import annotations

import re
from dataclasses import dataclass


SRT_TIME_PATTERN = re.compile(r"^(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})$")
CJK_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
LINE_START_PUNCTUATION = "、。，．！？!?…；;：:,，"
ASCII_WORD_SPLIT_PATTERN = re.compile(r"[A-Za-z]{2,}\n[A-Za-z]{2,}")
DEFAULT_MIN_GAP_SECONDS = 0.084
DEFAULT_MIN_DURATION_SECONDS = 0.5
DEFAULT_MAX_DURATION_SECONDS = 10.0
DEFAULT_MAX_CJK_LINE_CHARS = 24
DEFAULT_MAX_LATIN_LINE_CHARS = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MAX_CJK_CPS = 12.0
DEFAULT_MAX_LATIN_CPS = 20.0


class SRTValidationError(ValueError):
    pass


@dataclass
class SRTBlock:
    index: int
    start: float
    end: float
    text: str


def parse_srt(srt_text: str, strict: bool = True) -> list[SRTBlock]:
    normalized = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise SRTValidationError("SRT text is empty")

    blocks: list[SRTBlock] = []
    raw_blocks = re.split(r"\n\s*\n", normalized)
    for block_position, raw_block in enumerate(raw_blocks, start=1):
        lines = [line.strip("\ufeff").strip() for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue

        time_line_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if time_line_index is None:
            if strict:
                raise SRTValidationError(f"Block {block_position} has no time line")
            continue

        index = block_position
        if time_line_index > 0 and lines[0].isdigit():
            index = int(lines[0])
        elif strict:
            raise SRTValidationError(f"Block {block_position} has no numeric index")

        time_parts = [part.strip() for part in lines[time_line_index].split("-->", 1)]
        if len(time_parts) != 2:
            raise SRTValidationError(f"Block {block_position} has invalid time separator")

        try:
            start = parse_srt_time(time_parts[0])
            end = parse_srt_time(time_parts[1])
        except ValueError as exc:
            raise SRTValidationError(str(exc)) from exc

        text = "\n".join(lines[time_line_index + 1:]).strip()
        blocks.append(SRTBlock(index=index, start=start, end=end, text=text))

    validate_srt_blocks(blocks)
    return blocks


def validate_srt_blocks(blocks: list[SRTBlock]) -> None:
    if not blocks:
        raise SRTValidationError("SRT has no subtitle blocks")

    expected_indices = list(range(1, len(blocks) + 1))
    actual_indices = [block.index for block in blocks]
    if actual_indices != expected_indices:
        raise SRTValidationError("SRT indices are not consecutive")

    seen_indices: set[int] = set()
    for block in blocks:
        if block.index in seen_indices:
            raise SRTValidationError(f"Duplicate SRT index: {block.index}")
        seen_indices.add(block.index)

        if block.start < 0 or block.end < 0:
            raise SRTValidationError(f"Block {block.index} has negative time")
        if block.start >= block.end:
            raise SRTValidationError(f"Block {block.index} start time must be before end time")
        if not block.text.strip():
            raise SRTValidationError(f"Block {block.index} is empty")


def validate_srt_quality(
    blocks: list[SRTBlock],
    *,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    max_cjk_line_chars: int = DEFAULT_MAX_CJK_LINE_CHARS,
    max_latin_line_chars: int = DEFAULT_MAX_LATIN_LINE_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
    max_cjk_cps: float = DEFAULT_MAX_CJK_CPS,
    max_latin_cps: float = DEFAULT_MAX_LATIN_CPS,
) -> None:
    validate_srt_blocks(blocks)
    issues = collect_srt_quality_issues(
        blocks,
        min_gap_seconds=min_gap_seconds,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
        max_cjk_line_chars=max_cjk_line_chars,
        max_latin_line_chars=max_latin_line_chars,
        max_lines=max_lines,
        max_cjk_cps=max_cjk_cps,
        max_latin_cps=max_latin_cps,
    )
    if issues:
        raise SRTValidationError(issues[0])


def collect_srt_quality_issues(
    blocks: list[SRTBlock],
    *,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    min_duration_seconds: float = DEFAULT_MIN_DURATION_SECONDS,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    max_cjk_line_chars: int = DEFAULT_MAX_CJK_LINE_CHARS,
    max_latin_line_chars: int = DEFAULT_MAX_LATIN_LINE_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
    max_cjk_cps: float = DEFAULT_MAX_CJK_CPS,
    max_latin_cps: float = DEFAULT_MAX_LATIN_CPS,
) -> list[str]:
    issues: list[str] = []
    previous: SRTBlock | None = None

    for block in blocks:
        duration = block.end - block.start
        if previous:
            if block.start < previous.start:
                issues.append(f"Block {block.index} starts before previous block")
            if block.start < previous.end:
                issues.append(f"Block {block.index} overlaps previous block")
            elif block.start - previous.end < min_gap_seconds:
                issues.append(f"Block {block.index} gap is shorter than {min_gap_seconds:.3f}s")

        if duration < min_duration_seconds:
            issues.append(f"Block {block.index} duration is shorter than {min_duration_seconds:.3f}s")
        if duration > max_duration_seconds:
            issues.append(f"Block {block.index} duration is longer than {max_duration_seconds:.3f}s")

        text = block.text.strip()
        reading_chars = count_reading_chars(text)
        cps_limit = max_cjk_cps if contains_cjk(text) else max_latin_cps
        if duration > 0 and reading_chars / duration > cps_limit:
            issues.append(f"Block {block.index} reading speed is above {cps_limit:.1f} CPS")

        lines = text.splitlines()
        if len(lines) > max_lines:
            issues.append(f"Block {block.index} has more than {max_lines} subtitle lines")
        if ASCII_WORD_SPLIT_PATTERN.search(text):
            issues.append(f"Block {block.index} splits an ASCII word across lines")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0] in LINE_START_PUNCTUATION:
                issues.append(f"Block {block.index} has punctuation at the start of a line")
            line_limit = max_cjk_line_chars if contains_cjk(stripped) else max_latin_line_chars
            if count_reading_chars(stripped) > line_limit:
                issues.append(f"Block {block.index} line is longer than {line_limit} chars")

        previous = block

    return issues


def contains_cjk(text: str) -> bool:
    return bool(CJK_PATTERN.search(text))


def count_reading_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def serialize_srt(blocks: list[SRTBlock]) -> str:
    output_blocks: list[str] = []
    for index, block in enumerate(blocks, start=1):
        text = block.text.strip()
        if not text:
            raise SRTValidationError(f"Block {index} is empty")
        if block.start >= block.end:
            raise SRTValidationError(f"Block {index} start time must be before end time")

        output_blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(block.start)} --> {format_srt_time(block.end)}",
                    text,
                ]
            )
        )

    return "\n\n".join(output_blocks) + ("\n" if output_blocks else "")


def parse_srt_time(value: str) -> float:
    match = SRT_TIME_PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"Invalid SRT time: {value}")

    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    return (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)


def format_srt_time(seconds: float) -> str:
    milliseconds = round(max(seconds, 0.0) * 1000)
    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000
    minutes = milliseconds // 60_000
    milliseconds %= 60_000
    secs = milliseconds // 1000
    millis = milliseconds % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
