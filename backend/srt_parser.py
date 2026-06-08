from __future__ import annotations

import re
from dataclasses import dataclass


SRT_TIME_PATTERN = re.compile(r"^(\d{2,}):([0-5]\d):([0-5]\d),(\d{3})$")


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
