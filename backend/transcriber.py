from __future__ import annotations

import re
from dataclasses import dataclass

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps


SAMPLE_RATE = 16000
MIN_MIXED_SEGMENT_SECONDS = 0.35
PRO_CONTEXT_PADDING_SECONDS = 0.4
PRO_MERGE_GAP_SECONDS = 0.65
PRO_MIN_VAD_SEGMENT_SECONDS = 1.2
PRO_MAX_MERGED_VAD_SECONDS = 30.0
PRO_MIN_SUBTITLE_SECONDS = 0.7
PRO_SHORT_SUBTITLE_SECONDS = 0.9
PRO_MAX_MERGED_TEXT_CHARS = 50
PRO_MAX_LINE_CHARS = 42
PRO_SPLIT_SUBTITLE_SECONDS = 4.5
PRO_SPLIT_MIN_PART_CHARS = 4


@dataclass
class SubtitleBlock:
    start: float
    end: float
    text: str


class SRTTranscriber:
    def __init__(
        self,
        model_name: str = "large-v3",
        device: str = "cpu",
        compute_type: str = "int8",
        vad_filter: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.vad_filter = vad_filter
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def transcribe_to_srt(
        self,
        audio_path: str,
        language: str = "auto",
        mixed_ranges: list[tuple[float, float, str]] | None = None,
        professional_optimization: bool = False,
    ) -> str:
        if language == "mixed" and not mixed_ranges:
            return self._transcribe_mixed_to_srt(audio_path, professional_optimization)

        if mixed_ranges:
            return self._transcribe_with_mixed_ranges_to_srt(
                audio_path,
                language,
                mixed_ranges,
                professional_optimization,
            )

        transcribe_kwargs = {
            "task": "transcribe",
            "beam_size": 5,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": False,
            "temperature": 0.0,
        }

        if language and language != "auto":
            transcribe_kwargs["language"] = language

        segments, _info = self.model.transcribe(audio_path, **transcribe_kwargs)
        srt_text = segments_to_srt(segments)
        return optimize_srt_text(srt_text) if professional_optimization else srt_text

    def _transcribe_mixed_to_srt(self, audio_path: str, professional_optimization: bool) -> str:
        audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        blocks: list[str] = []
        self._append_mixed_audio_to_blocks(
            blocks,
            audio,
            offset=0.0,
            professional_optimization=professional_optimization,
        )
        srt_text = "\n\n".join(blocks) + ("\n" if blocks else "")
        return optimize_srt_text(srt_text) if professional_optimization else srt_text

    def _transcribe_with_mixed_ranges_to_srt(
        self,
        audio_path: str,
        language: str,
        mixed_ranges: list[tuple[float, float, str]],
        professional_optimization: bool,
    ) -> str:
        audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        total_duration = len(audio) / SAMPLE_RATE
        ranges = normalize_ranges(mixed_ranges, total_duration)
        if not ranges:
            return self.transcribe_to_srt(
                audio_path,
                language,
                professional_optimization=professional_optimization,
            )

        blocks: list[str] = []
        cursor = 0.0

        for start, end, range_language in ranges:
            if start > cursor:
                self._append_language_audio_to_blocks(
                    blocks,
                    audio,
                    start=cursor,
                    end=start,
                    language=language,
                    professional_optimization=professional_optimization,
                )

            self._append_language_audio_to_blocks(
                blocks,
                audio,
                start=start,
                end=end,
                language=range_language,
                professional_optimization=professional_optimization,
            )
            cursor = end

        if cursor < total_duration:
            self._append_language_audio_to_blocks(
                blocks,
                audio,
                start=cursor,
                end=total_duration,
                language=language,
                professional_optimization=professional_optimization,
            )

        srt_text = "\n\n".join(blocks) + ("\n" if blocks else "")
        return optimize_srt_text(srt_text) if professional_optimization else srt_text

    def _append_language_audio_to_blocks(
        self,
        blocks: list[str],
        audio,
        start: float,
        end: float,
        language: str,
        professional_optimization: bool,
    ) -> None:
        if language == "mixed":
            self._append_mixed_audio_to_blocks(
                blocks,
                slice_audio(audio, start, end),
                offset=start,
                professional_optimization=professional_optimization,
            )
            return

        self._append_target_audio_to_blocks(
            blocks,
            audio,
            start=start,
            end=end,
            language=language,
            professional_optimization=professional_optimization,
        )

    def _append_target_audio_to_blocks(
        self,
        blocks: list[str],
        audio,
        start: float,
        end: float,
        language: str,
        professional_optimization: bool,
    ) -> None:
        chunk = slice_audio(audio, start, end)
        chunk_duration = len(chunk) / SAMPLE_RATE
        if chunk_duration < MIN_MIXED_SEGMENT_SECONDS:
            return

        if professional_optimization:
            vad_options = VadOptions(
                threshold=0.5,
                min_silence_duration_ms=700,
                speech_pad_ms=300,
                max_speech_duration_s=30,
            )
            speech_timestamps = get_speech_timestamps(
                chunk,
                vad_options=vad_options,
                sampling_rate=SAMPLE_RATE,
            )
            if speech_timestamps:
                for speech in merge_speech_timestamps(speech_timestamps):
                    self._append_target_window_to_blocks(
                        blocks,
                        chunk,
                        base_offset=start,
                        speech_start_sample=max(0, int(speech["start"])),
                        speech_end_sample=min(len(chunk), int(speech["end"])),
                        language=language,
                        professional_optimization=True,
                    )
                return

        self._append_target_window_to_blocks(
            blocks,
            chunk,
            base_offset=start,
            speech_start_sample=0,
            speech_end_sample=len(chunk),
            language=language,
            professional_optimization=False,
        )

    def _append_target_window_to_blocks(
        self,
        blocks: list[str],
        audio,
        base_offset: float,
        speech_start_sample: int,
        speech_end_sample: int,
        language: str,
        professional_optimization: bool,
    ) -> None:
        if speech_end_sample <= speech_start_sample:
            return

        start_sample = speech_start_sample
        end_sample = speech_end_sample
        if professional_optimization:
            padding_samples = int(PRO_CONTEXT_PADDING_SECONDS * SAMPLE_RATE)
            start_sample = max(0, speech_start_sample - padding_samples)
            end_sample = min(len(audio), speech_end_sample + padding_samples)

        chunk = audio[start_sample:end_sample]
        chunk_duration = len(chunk) / SAMPLE_RATE
        if chunk_duration < MIN_MIXED_SEGMENT_SECONDS:
            return

        transcribe_kwargs = {
            "task": "transcribe",
            "beam_size": 5,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": False,
            "temperature": 0.0,
        }
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        segments, _info = self.model.transcribe(chunk, **transcribe_kwargs)
        append_segments_to_blocks(
            blocks,
            segments,
            offset=base_offset + (start_sample / SAMPLE_RATE),
            min_start=base_offset + (speech_start_sample / SAMPLE_RATE) if professional_optimization else None,
            max_end=base_offset + (speech_end_sample / SAMPLE_RATE) if professional_optimization else base_offset + (len(audio) / SAMPLE_RATE),
        )

    def _append_mixed_audio_to_blocks(
        self,
        blocks: list[str],
        audio,
        offset: float,
        professional_optimization: bool,
    ) -> None:
        vad_options = VadOptions(
            threshold=0.5,
            min_silence_duration_ms=700 if professional_optimization else 500,
            speech_pad_ms=300,
            max_speech_duration_s=30,
        )
        speech_timestamps = get_speech_timestamps(
            audio,
            vad_options=vad_options,
            sampling_rate=SAMPLE_RATE,
        )

        if not speech_timestamps:
            segments, _info = self.model.transcribe(
                audio,
                task="transcribe",
                beam_size=10,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4],
                multilingual=True,
                language_detection_segments=1,
            )
            append_segments_to_blocks(blocks, segments, offset=offset, max_end=offset + (len(audio) / SAMPLE_RATE))
            return

        if professional_optimization:
            speech_timestamps = merge_speech_timestamps(speech_timestamps)

        base_offset = offset
        for speech in speech_timestamps:
            raw_start_sample = max(0, int(speech["start"]))
            raw_end_sample = min(len(audio), int(speech["end"]))
            start_sample = raw_start_sample
            end_sample = raw_end_sample
            if professional_optimization:
                padding_samples = int(PRO_CONTEXT_PADDING_SECONDS * SAMPLE_RATE)
                start_sample = max(0, raw_start_sample - padding_samples)
                end_sample = min(len(audio), raw_end_sample + padding_samples)
            if end_sample <= start_sample:
                continue

            chunk = audio[start_sample:end_sample]
            chunk_duration = len(chunk) / SAMPLE_RATE
            if chunk_duration < MIN_MIXED_SEGMENT_SECONDS:
                continue

            speech_start = start_sample / SAMPLE_RATE
            speech_end = end_sample / SAMPLE_RATE
            output_start = raw_start_sample / SAMPLE_RATE
            output_end = raw_end_sample / SAMPLE_RATE
            segments, _info = self.model.transcribe(
                chunk,
                task="transcribe",
                beam_size=10,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4],
                multilingual=True,
                language_detection_segments=1,
            )
            append_segments_to_blocks(
                blocks,
                segments,
                offset=base_offset + speech_start,
                min_start=base_offset + output_start if professional_optimization else None,
                max_end=base_offset + (output_end if professional_optimization else speech_end),
            )


def segments_to_srt(segments) -> str:
    blocks: list[str] = []
    append_segments_to_blocks(blocks, segments)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def append_segments_to_blocks(
    blocks: list[str],
    segments,
    offset: float = 0.0,
    min_start: float | None = None,
    max_end: float | None = None,
) -> None:
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue

        index = len(blocks) + 1
        start = offset + max(float(segment.start or 0), 0.0)
        end = offset + max(float(segment.end or 0), 0.0)
        if min_start is not None:
            start = max(start, min_start)
            end = max(end, min_start)
        if max_end is not None:
            start = min(start, max_end)
            end = min(end, max_end)
        if end <= start:
            if min_start is not None and max_end is not None:
                if max_end <= start:
                    continue
                end = min(start + 0.2, max_end)
            else:
                end = start + 0.2

        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(start)} --> {format_srt_time(end)}",
                    text,
                ]
            )
        )


def merge_speech_timestamps(speech_timestamps) -> list[dict[str, int]]:
    merged: list[dict[str, int]] = []
    for speech in speech_timestamps:
        start = int(speech["start"])
        end = int(speech["end"])
        if end <= start:
            continue

        if not merged:
            merged.append({"start": start, "end": end})
            continue

        previous = merged[-1]
        gap = (start - previous["end"]) / SAMPLE_RATE
        previous_duration = (previous["end"] - previous["start"]) / SAMPLE_RATE
        current_duration = (end - start) / SAMPLE_RATE
        merged_duration = (end - previous["start"]) / SAMPLE_RATE
        should_merge = (
            merged_duration <= PRO_MAX_MERGED_VAD_SECONDS
            and (
                gap <= PRO_MERGE_GAP_SECONDS
                or (
                    gap <= 1.2
                    and (
                        previous_duration < PRO_MIN_VAD_SEGMENT_SECONDS
                        or current_duration < PRO_MIN_VAD_SEGMENT_SECONDS
                    )
                )
            )
        )

        if should_merge:
            previous["end"] = max(previous["end"], end)
        else:
            merged.append({"start": start, "end": end})

    return merged


def optimize_srt_text(srt_text: str) -> str:
    blocks = parse_srt_blocks(srt_text)
    if not blocks:
        return srt_text

    blocks = [
        SubtitleBlock(block.start, block.end, normalize_subtitle_text(block.text))
        for block in blocks
    ]
    blocks = merge_short_subtitle_blocks(blocks)
    blocks = split_long_sentence_blocks(blocks)
    blocks = [
        SubtitleBlock(block.start, block.end, wrap_subtitle_text(normalize_subtitle_text(block.text)))
        for block in blocks
    ]
    blocks = fix_subtitle_timings(blocks)
    return subtitle_blocks_to_srt(blocks)


def parse_srt_blocks(srt_text: str) -> list[SubtitleBlock]:
    parsed: list[SubtitleBlock] = []
    for raw_block in re.split(r"\n\s*\n", srt_text.strip()):
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        time_line_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if time_line_index is None or time_line_index + 1 >= len(lines):
            continue

        time_line = lines[time_line_index]
        start_text, end_text = [part.strip() for part in time_line.split("-->", 1)]
        try:
            start = parse_srt_time(start_text)
            end = parse_srt_time(end_text)
        except ValueError:
            continue

        text = "\n".join(lines[time_line_index + 1:]).strip()
        if text and end > start:
            parsed.append(SubtitleBlock(start=start, end=end, text=text))

    return parsed


def subtitle_blocks_to_srt(blocks: list[SubtitleBlock]) -> str:
    output_blocks: list[str] = []
    for index, block in enumerate(blocks, start=1):
        text = block.text.strip()
        if not text:
            continue

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


def merge_short_subtitle_blocks(blocks: list[SubtitleBlock]) -> list[SubtitleBlock]:
    merged: list[SubtitleBlock] = []
    index = 0
    while index < len(blocks):
        current = blocks[index]
        while index + 1 < len(blocks) and should_merge_subtitle_blocks(current, blocks[index + 1]):
            next_block = blocks[index + 1]
            current = SubtitleBlock(
                start=current.start,
                end=max(current.end, next_block.end),
                text=join_subtitle_text(current.text, next_block.text),
            )
            index += 1

        merged.append(current)
        index += 1

    return merged


def should_merge_subtitle_blocks(current: SubtitleBlock, next_block: SubtitleBlock) -> bool:
    gap = next_block.start - current.end
    if gap < 0:
        gap = 0

    combined_text = join_subtitle_text(current.text, next_block.text)
    combined_chars = len(re.sub(r"\s+", "", combined_text))
    if combined_chars > PRO_MAX_MERGED_TEXT_CHARS:
        return False

    if is_page_fragment(current.text, next_block.text) and gap <= 1.2:
        return True

    current_duration = current.end - current.start
    next_duration = next_block.end - next_block.start
    current_chars = len(re.sub(r"\s+", "", current.text))
    next_chars = len(re.sub(r"\s+", "", next_block.text))

    if gap <= 0.35 and (
        current_duration < PRO_SHORT_SUBTITLE_SECONDS
        or next_duration < PRO_SHORT_SUBTITLE_SECONDS
        or current_chars <= 4
        or next_chars <= 4
    ):
        return True

    return current_duration < 0.6 and gap <= 0.8


def split_long_sentence_blocks(blocks: list[SubtitleBlock]) -> list[SubtitleBlock]:
    split_blocks: list[SubtitleBlock] = []
    for block in blocks:
        split_blocks.extend(split_long_sentence_block(block))
    return split_blocks


def split_long_sentence_block(block: SubtitleBlock) -> list[SubtitleBlock]:
    text = flatten_subtitle_text(block.text)
    duration = block.end - block.start
    if duration < PRO_SPLIT_SUBTITLE_SECONDS:
        return [block]

    parts = split_text_on_sentence_boundaries(text)
    if len(parts) < 2:
        return [block]

    part_lengths = [len(re.sub(r"\s+", "", part)) for part in parts]
    if any(length < PRO_SPLIT_MIN_PART_CHARS for length in part_lengths):
        return [block]

    total_length = sum(part_lengths)
    if total_length <= 0:
        return [block]

    split_blocks: list[SubtitleBlock] = []
    cursor = block.start
    elapsed_length = 0
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            part_end = block.end
        else:
            elapsed_length += part_lengths[index]
            part_end = block.start + (duration * elapsed_length / total_length)

        if part_end - cursor < PRO_MIN_SUBTITLE_SECONDS:
            return [block]

        split_blocks.append(SubtitleBlock(start=cursor, end=part_end, text=part))
        cursor = part_end

    return split_blocks


def split_text_on_sentence_boundaries(text: str) -> list[str]:
    parts = re.split(r"(?<=[。．.!?！？])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def is_page_fragment(left: str, right: str) -> bool:
    left_text = flatten_subtitle_text(left)
    right_text = flatten_subtitle_text(right)
    return bool(
        re.search(r"\d+\s*[%％]?$", left_text)
        and re.match(r"^(ページ|頁)$", right_text)
    )


def join_subtitle_text(left: str, right: str) -> str:
    left_text = flatten_subtitle_text(left)
    right_text = flatten_subtitle_text(right)
    if not left_text:
        return right_text
    if not right_text:
        return left_text

    if should_join_without_space(left_text, right_text):
        return left_text.rstrip() + right_text.lstrip()
    return left_text.rstrip() + " " + right_text.lstrip()


def flatten_subtitle_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def should_join_without_space(left: str, right: str) -> bool:
    if re.match(r"^[、。，．！？!?）」』】\]）]", right):
        return True
    if re.match(r"^(ページ|頁|課|番)", right):
        return True
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]$", left):
        return True
    if re.match(r"^[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", right):
        return True
    return False


def normalize_subtitle_text(text: str) -> str:
    normalized = flatten_subtitle_text(text)
    normalized = re.sub(r"(\d+)\s*[%％]\s*(ページ|頁)", r"\1\2", normalized)
    normalized = re.sub(r"第\s*(\d+)\s*課", r"第\1課", normalized)
    normalized = re.sub(r"第\s+(\d+)", r"第\1", normalized)
    normalized = re.sub(r"(\d+)\s+(ページ|頁|課|番)", r"\1\2", normalized)
    normalized = re.sub(r"\s+([、。，．！？!?])", r"\1", normalized)
    return normalized.strip()


def wrap_subtitle_text(text: str) -> str:
    if "\n" in text or len(text) <= PRO_MAX_LINE_CHARS or " " not in text:
        return text

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > PRO_MAX_LINE_CHARS and len(lines) < 1:
            lines.append(current)
            current = word
        else:
            current = candidate

    if current:
        lines.append(current)

    return "\n".join(lines) if len(lines) > 1 else text


def fix_subtitle_timings(blocks: list[SubtitleBlock]) -> list[SubtitleBlock]:
    fixed = sorted(blocks, key=lambda block: block.start)
    for index, block in enumerate(fixed):
        if block.end <= block.start:
            block.end = block.start + 0.2

        next_block = fixed[index + 1] if index + 1 < len(fixed) else None
        if next_block and block.end > next_block.start:
            if next_block.start > block.start:
                block.end = next_block.start
            else:
                next_block.start = block.end + 0.02

        if block.end - block.start < PRO_MIN_SUBTITLE_SECONDS:
            target_end = block.start + PRO_MIN_SUBTITLE_SECONDS
            if not next_block or target_end <= next_block.start:
                block.end = target_end

    return [block for block in fixed if block.text.strip() and block.end > block.start]


def parse_srt_time(value: str) -> float:
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT time: {value}")

    hours, minutes, seconds, milliseconds = [int(part) for part in match.groups()]
    return (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)


def slice_audio(audio, start: float, end: float):
    start_sample = max(0, int(start * SAMPLE_RATE))
    end_sample = min(len(audio), int(end * SAMPLE_RATE))
    return audio[start_sample:end_sample]


def normalize_ranges(ranges: list[tuple[float, float, str]], total_duration: float) -> list[tuple[float, float, str]]:
    normalized: list[tuple[float, float, str]] = []
    for start, end, language in ranges:
        start = max(0.0, min(float(start), total_duration))
        end = max(0.0, min(float(end), total_duration))
        if end - start >= MIN_MIXED_SEGMENT_SECONDS:
            normalized.append((start, end, language))

    normalized.sort(key=lambda item: item[0])
    merged: list[tuple[float, float, str]] = []
    for start, end, language in normalized:
        if not merged or start > merged[-1][1] or language != merged[-1][2]:
            merged.append((start, end, language))
        else:
            prev_start, prev_end, prev_language = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_language)

    return merged


def format_srt_time(seconds: float) -> str:
    seconds = max(float(seconds or 0), 0.0)

    whole_seconds = int(seconds)
    milliseconds = round((seconds - whole_seconds) * 1000)

    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0

    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
