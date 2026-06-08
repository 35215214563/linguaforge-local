from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from .srt_cleaner import clean_subtitle_text
from .srt_cleaner import flatten_subtitle_text as clean_flatten_subtitle_text
from .srt_parser import format_srt_time


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
PRO_MAX_LINES_PER_BLOCK = 2
PRO_MAX_BLOCK_SECONDS = 7.0
PRO_MIN_BLOCK_SECONDS = 1.0
PRO_MIN_GAP_SECONDS = 0.084
PRO_CJK_MAX_LINE_CHARS = 21
PRO_LATIN_MAX_LINE_CHARS = 42
PRO_CJK_MAX_CPS = 9.0
PRO_LATIN_MAX_CPS = 17.0
PRO_SOFT_CUT_SECONDS = 2.4
PRO_TARGET_SPLIT_SECONDS = 5.5
PRO_WORD_GAP_CUT_SECONDS = 1.2
PRO_HARD_CUT_GRACE_CHARS = 8
PROTECTED_PHRASES_FILE = Path(__file__).resolve().parent / "subtitle_corrections" / "protected_phrases.json"
CJK_WRAP_BOUNDARY_CHARS = "、。，．！？!?；;：:,，"
CJK_WRAP_PROHIBITED_START_CHARS = "、。，．！？!?…；;：:,，）」』】)]"
CJK_WRAP_PROHIBITED_PREFIX_CHARS = "这那哪每各第"
CJK_WRAP_PROHIBITED_SUFFIX_START_CHARS = "个些种样位条只本张件次天年月日点分秒字词课页人们的了着过学"
DEFAULT_PROTECTED_PHRASES = (
    "深入探讨",
    "深入探討",
    "神经科学",
    "神經科學",
    "核心机制",
    "核心機制",
    "空间路径",
    "空間路徑",
    "自动反击",
    "自動反擊",
    "十分钟对话",
    "十分鐘對話",
    "可理解性输入",
    "可理解性輸入",
    "记忆宫殿法",
    "記憶宮殿法",
    "句型替换训练",
    "句型替換訓練",
    "严格的教官",
    "嚴格的教官",
    "严格",
    "嚴格",
    "Honestly, I think",
    "what I mean is",
    "Pattern Drill",
)


def load_protected_phrases(path: Path, fallback: tuple[str, ...]) -> tuple[str, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback

    raw_phrases = payload.get("phrases", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_phrases, list):
        return fallback

    phrases: list[str] = []
    for phrase in raw_phrases:
        if isinstance(phrase, str):
            normalized = phrase.strip()
            if normalized and normalized not in phrases:
                phrases.append(normalized)

    return tuple(phrases) if phrases else fallback


PROTECTED_PHRASES = load_protected_phrases(PROTECTED_PHRASES_FILE, DEFAULT_PROTECTED_PHRASES)


@dataclass
class SubtitleBlock:
    start: float
    end: float
    text: str


@dataclass
class WordTiming:
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
        mixed_ranges: Optional[list[tuple[float, float, str]]] = None,
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
        if professional_optimization:
            transcribe_kwargs["word_timestamps"] = True

        if language and language != "auto":
            transcribe_kwargs["language"] = language

        segments, _info = self.model.transcribe(audio_path, **transcribe_kwargs)
        srt_text = segments_to_srt(
            segments,
            professional_optimization=professional_optimization,
            clean_language=language,
        )
        return optimize_srt_text(srt_text, language=language) if professional_optimization else srt_text

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
        return optimize_srt_text(srt_text, language="mixed") if professional_optimization else srt_text

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
        return optimize_srt_text(srt_text, language=language) if professional_optimization else srt_text

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
        if professional_optimization:
            transcribe_kwargs["word_timestamps"] = True
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        segments, _info = self.model.transcribe(chunk, **transcribe_kwargs)
        append_segments_to_blocks(
            blocks,
            segments,
            offset=base_offset + (start_sample / SAMPLE_RATE),
            min_start=base_offset + (speech_start_sample / SAMPLE_RATE) if professional_optimization else None,
            max_end=base_offset + (speech_end_sample / SAMPLE_RATE) if professional_optimization else base_offset + (len(audio) / SAMPLE_RATE),
            professional_optimization=professional_optimization,
            clean_language=language,
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
            transcribe_kwargs = {
                "task": "transcribe",
                "beam_size": 10,
                "vad_filter": False,
                "condition_on_previous_text": False,
                "temperature": [0.0, 0.2, 0.4],
                "multilingual": True,
                "language_detection_segments": 1,
            }
            if professional_optimization:
                transcribe_kwargs["word_timestamps"] = True

            segments, _info = self.model.transcribe(audio, **transcribe_kwargs)
            append_segments_to_blocks(
                blocks,
                segments,
                offset=offset,
                max_end=offset + (len(audio) / SAMPLE_RATE),
                professional_optimization=professional_optimization,
                clean_language="mixed",
            )
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
            transcribe_kwargs = {
                "task": "transcribe",
                "beam_size": 10,
                "vad_filter": False,
                "condition_on_previous_text": False,
                "temperature": [0.0, 0.2, 0.4],
                "multilingual": True,
                "language_detection_segments": 1,
            }
            if professional_optimization:
                transcribe_kwargs["word_timestamps"] = True

            segments, _info = self.model.transcribe(chunk, **transcribe_kwargs)
            append_segments_to_blocks(
                blocks,
                segments,
                offset=base_offset + speech_start,
                min_start=base_offset + output_start if professional_optimization else None,
                max_end=base_offset + (output_end if professional_optimization else speech_end),
                professional_optimization=professional_optimization,
                clean_language="mixed",
            )


def segments_to_srt(
    segments,
    professional_optimization: bool = False,
    clean_language: str = "auto",
) -> str:
    blocks: list[str] = []
    append_segments_to_blocks(
        blocks,
        segments,
        professional_optimization=professional_optimization,
        clean_language=clean_language,
    )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def append_segments_to_blocks(
    blocks: list[str],
    segments,
    offset: float = 0.0,
    min_start: Optional[float] = None,
    max_end: Optional[float] = None,
    professional_optimization: bool = False,
    clean_language: str = "auto",
) -> None:
    if professional_optimization:
        subtitle_blocks = collect_professional_subtitle_blocks(
            segments,
            offset=offset,
            min_start=min_start,
            max_end=max_end,
            clean_language=clean_language,
        )
        append_subtitle_blocks_to_blocks(blocks, subtitle_blocks)
        return

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


def collect_professional_subtitle_blocks(
    segments,
    offset: float = 0.0,
    min_start: Optional[float] = None,
    max_end: Optional[float] = None,
    clean_language: str = "auto",
) -> list[SubtitleBlock]:
    words: list[WordTiming] = []
    fallback_blocks: list[SubtitleBlock] = []

    for segment in segments:
        text = (getattr(segment, "text", "") or "").strip()
        start = offset + max(float(getattr(segment, "start", 0) or 0), 0.0)
        end = offset + max(float(getattr(segment, "end", 0) or 0), 0.0)
        start, end = clamp_time_range(start, end, min_start=min_start, max_end=max_end)
        if text and end > start:
            fallback_blocks.append(SubtitleBlock(start=start, end=end, text=text))

        for word in getattr(segment, "words", None) or []:
            word_text = (getattr(word, "word", "") or "").strip()
            if not word_text:
                continue

            word_start = offset + max(float(getattr(word, "start", 0) or 0), 0.0)
            word_end = offset + max(float(getattr(word, "end", 0) or 0), 0.0)
            word_start, word_end = clamp_time_range(
                word_start,
                word_end,
                min_start=min_start,
                max_end=max_end,
            )
            if word_end <= word_start:
                continue

            words.append(WordTiming(start=word_start, end=word_end, text=word_text))

    if not words:
        return fallback_blocks

    return segment_words_into_blocks(words, clean_language=clean_language)


def clamp_time_range(
    start: float,
    end: float,
    min_start: Optional[float] = None,
    max_end: Optional[float] = None,
) -> tuple[float, float]:
    if min_start is not None:
        start = max(start, min_start)
        end = max(end, min_start)
    if max_end is not None:
        start = min(start, max_end)
        end = min(end, max_end)
    return start, end


def segment_words_into_blocks(
    words: list[WordTiming],
    clean_language: str = "auto",
) -> list[SubtitleBlock]:
    blocks: list[SubtitleBlock] = []
    buffer_words: list[WordTiming] = []

    for word in words:
        if buffer_words and word.start - buffer_words[-1].end >= PRO_WORD_GAP_CUT_SECONDS:
            blocks.append(
                SubtitleBlock(
                    start=buffer_words[0].start,
                    end=buffer_words[-1].end,
                    text=words_to_text(buffer_words),
                )
            )
            buffer_words = []

        buffer_words.append(word)
        text = words_to_text(buffer_words)
        duration = buffer_words[-1].end - buffer_words[0].start
        compact_chars = count_reading_chars(text)
        hard_char_limit = max_chars_per_line(text) * PRO_MAX_LINES_PER_BLOCK
        soft_char_limit = max_chars_per_line(text)

        should_cut = is_sentence_end(word.text) or (
            is_soft_sentence_boundary(word.text)
            and duration >= PRO_SOFT_CUT_SECONDS
            and compact_chars >= soft_char_limit
        )
        cut_count = len(buffer_words) if should_cut else 0

        if not cut_count and (duration >= PRO_MAX_BLOCK_SECONDS or compact_chars >= hard_char_limit):
            cut_count = choose_word_buffer_cut_count(
                buffer_words,
                hard_char_limit=hard_char_limit,
                force_cut=duration >= PRO_MAX_BLOCK_SECONDS
                or compact_chars >= hard_char_limit + PRO_HARD_CUT_GRACE_CHARS,
            )

        if cut_count:
            block_words = buffer_words[:cut_count]
            blocks.append(
                SubtitleBlock(
                    start=block_words[0].start,
                    end=block_words[-1].end,
                    text=words_to_text(block_words),
                )
            )
            buffer_words = buffer_words[cut_count:]

    if buffer_words:
        blocks.append(
            SubtitleBlock(
                start=buffer_words[0].start,
                end=buffer_words[-1].end,
                text=words_to_text(buffer_words),
            )
        )

    return enforce_industry_standards(blocks, language=clean_language)


def words_to_text(words: list[WordTiming]) -> str:
    text = "".join(word.text for word in words).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([、。，．！？!?…])", r"\1", text)
    return text.strip()


def choose_word_buffer_cut_count(
    buffer_words: list[WordTiming],
    hard_char_limit: int,
    force_cut: bool,
) -> int:
    if not buffer_words:
        return 0

    best_cut = find_natural_word_cut_count(buffer_words, hard_char_limit)
    if best_cut:
        return best_cut

    if not force_cut:
        return 0

    fallback_cut = len(buffer_words)
    for index in range(len(buffer_words) - 1, 0, -1):
        prefix = words_to_text(buffer_words[:index])
        if count_reading_chars(prefix) <= hard_char_limit:
            fallback_cut = index
            break

    return max(1, fallback_cut)


def find_natural_word_cut_count(buffer_words: list[WordTiming], hard_char_limit: int) -> int:
    min_chars = max(8, int(hard_char_limit * 0.45))
    max_chars = hard_char_limit + PRO_HARD_CUT_GRACE_CHARS

    for index in range(len(buffer_words) - 1, 0, -1):
        prefix = words_to_text(buffer_words[:index])
        prefix_chars = count_reading_chars(prefix)
        if prefix_chars < min_chars or prefix_chars > max_chars:
            continue
        if is_natural_cjk_cut_text(prefix):
            return index

    return 0


def is_natural_cjk_cut_text(text: str) -> bool:
    stripped = text.strip()
    if is_sentence_end(stripped) or is_soft_sentence_boundary(stripped):
        return True
    if not contains_cjk(stripped):
        return False
    return bool(re.search(r"(了|嘛|吗|呢|吧|啊|哦|啦|咯|对|没错)$", stripped))


def is_sentence_end(text: str) -> bool:
    return bool(re.search(r"[。．！？!?…]+[\"'」』）\])]*$", text.strip()))


def is_soft_sentence_boundary(text: str) -> bool:
    return bool(re.search(r"[、，,;；:：]+[\"'」』）\])]*$", text.strip()))


def count_reading_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text))


def max_chars_per_line(text: str) -> int:
    return PRO_CJK_MAX_LINE_CHARS if contains_cjk(text) else PRO_LATIN_MAX_LINE_CHARS


def max_cps(text: str) -> float:
    return PRO_CJK_MAX_CPS if contains_cjk(text) else PRO_LATIN_MAX_CPS


def enforce_industry_standards(
    blocks: list[SubtitleBlock],
    language: str = "auto",
) -> list[SubtitleBlock]:
    if not blocks:
        return []

    adjusted: list[SubtitleBlock] = []
    sorted_blocks = sorted(blocks, key=lambda block: block.start)

    for index, block in enumerate(sorted_blocks):
        text = normalize_subtitle_text(block.text, language=language)
        if not text:
            continue

        start = max(block.start, 0.0)
        end = max(block.end, start + 0.2)
        char_count = count_reading_chars(text)
        target_duration = max(
            end - start,
            PRO_MIN_BLOCK_SECONDS,
            char_count / max_cps(text) if char_count else PRO_MIN_BLOCK_SECONDS,
        )
        target_duration = min(target_duration, PRO_MAX_BLOCK_SECONDS)

        next_block = sorted_blocks[index + 1] if index + 1 < len(sorted_blocks) else None
        max_allowed_end = next_block.start - PRO_MIN_GAP_SECONDS if next_block else None
        target_end = start + target_duration
        if max_allowed_end is not None and target_end > max_allowed_end:
            if max_allowed_end > start + 0.5:
                target_end = max_allowed_end
            else:
                target_end = max(end, start + 0.2)

        adjusted.append(SubtitleBlock(start=start, end=target_end, text=text))

    return fix_subtitle_timings(adjusted)


def append_subtitle_blocks_to_blocks(blocks: list[str], subtitle_blocks: list[SubtitleBlock]) -> None:
    for subtitle_block in subtitle_blocks:
        text = subtitle_block.text.strip()
        if not text or subtitle_block.end <= subtitle_block.start:
            continue

        index = len(blocks) + 1
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{format_srt_time(subtitle_block.start)} --> {format_srt_time(subtitle_block.end)}",
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


def optimize_srt_text(srt_text: str, language: str = "auto") -> str:
    blocks = parse_srt_blocks(srt_text)
    if not blocks:
        return srt_text

    blocks = [
        SubtitleBlock(block.start, block.end, normalize_subtitle_text(block.text, language=language))
        for block in blocks
    ]
    blocks = merge_short_subtitle_blocks(blocks)
    blocks = split_long_sentence_blocks(blocks)
    blocks = [
        SubtitleBlock(
            block.start,
            block.end,
            wrap_subtitle_text(normalize_subtitle_text(block.text, language=language)),
        )
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
    combined_duration = max(current.end, next_block.end) - current.start
    current_chars = len(re.sub(r"\s+", "", current.text))
    next_chars = len(re.sub(r"\s+", "", next_block.text))

    if (
        gap <= 0.25
        and combined_duration <= PRO_MAX_BLOCK_SECONDS
        and combined_chars <= max_chars_per_line(combined_text) * PRO_MAX_LINES_PER_BLOCK
        and not is_sentence_end(current.text)
    ):
        return True

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
    char_count = count_reading_chars(text)
    hard_char_limit = max_chars_per_line(text) * PRO_MAX_LINES_PER_BLOCK
    if duration < PRO_SPLIT_SUBTITLE_SECONDS and char_count <= hard_char_limit:
        return [block]

    target_parts = max(
        2,
        math.ceil(duration / PRO_TARGET_SPLIT_SECONDS),
        math.ceil(char_count / hard_char_limit) if hard_char_limit else 2,
    )
    parts = split_text_for_subtitle_blocks(text, target_parts)
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


def split_text_for_subtitle_blocks(text: str, target_parts: int) -> list[str]:
    normalized = flatten_subtitle_text(text)
    if target_parts <= 1:
        return [normalized] if normalized else []

    split_indices = choose_natural_text_split_indices(normalized, target_parts)
    if not split_indices:
        return split_text_on_sentence_boundaries(normalized)

    parts: list[str] = []
    previous = 0
    for split_index in split_indices:
        part = normalized[previous:split_index].strip()
        if part:
            parts.append(part)
        previous = split_index

    tail = normalized[previous:].strip()
    if tail:
        parts.append(tail)

    return parts


def choose_natural_text_split_indices(text: str, target_parts: int) -> list[int]:
    total_chars = count_reading_chars(text)
    if total_chars <= PRO_SPLIT_MIN_PART_CHARS * 2:
        return []

    candidates = natural_text_split_candidates(text)
    split_indices: list[int] = []
    previous_index = 0
    previous_reading_count = 0

    for part_number in range(1, target_parts):
        desired_count = round(total_chars * part_number / target_parts)
        remaining_parts = target_parts - part_number
        candidate = choose_text_split_candidate(
            text,
            candidates,
            desired_count,
            previous_index,
            previous_reading_count,
            remaining_parts,
            total_chars,
        )
        if candidate is None:
            fallback_index = string_index_at_reading_count(text, desired_count)
            candidate = find_fallback_text_split_index(text, fallback_index, previous_index)

        if candidate is None or candidate <= previous_index or candidate >= len(text):
            return split_indices

        split_indices.append(candidate)
        previous_index = candidate
        previous_reading_count = count_reading_chars(text[:candidate])

    return split_indices


def natural_text_split_candidates(text: str) -> list[int]:
    candidates: list[int] = []
    for index, char in enumerate(text[:-1]):
        if char not in CJK_WRAP_BOUNDARY_CHARS:
            continue
        split_index = adjust_protected_phrase_wrap_index(text, index + 1, index + 1)
        if split_index <= 0 or split_index >= len(text):
            continue
        if split_index not in candidates:
            candidates.append(split_index)

    return candidates


def choose_text_split_candidate(
    text: str,
    candidates: list[int],
    desired_count: int,
    previous_index: int,
    previous_reading_count: int,
    remaining_parts: int,
    total_chars: int,
) -> Optional[int]:
    min_part_chars = max(PRO_SPLIT_MIN_PART_CHARS, 6 if contains_cjk(text) else 8)
    usable_candidates: list[tuple[int, int]] = []
    for candidate in candidates:
        if candidate <= previous_index or candidate >= len(text):
            continue
        candidate_count = count_reading_chars(text[:candidate])
        part_chars = candidate_count - previous_reading_count
        remaining_chars = total_chars - candidate_count
        if part_chars < min_part_chars:
            continue
        if remaining_parts > 0 and remaining_chars < min_part_chars * remaining_parts:
            continue
        usable_candidates.append((candidate, candidate_count))

    if not usable_candidates:
        return None

    return min(
        usable_candidates,
        key=lambda item: (
            abs(item[1] - desired_count) + text_split_candidate_penalty(text, item[0]),
            abs(item[0] - previous_index),
        ),
    )[0]


def text_split_candidate_penalty(text: str, candidate: int) -> int:
    if candidate <= 0 or candidate > len(text):
        return 0

    previous_char = text[candidate - 1]
    if previous_char == "、":
        return 8
    if previous_char in "。．！？!?":
        return -3
    if previous_char in "；;：:":
        return -1
    return 0


def string_index_at_reading_count(text: str, target_count: int) -> int:
    count = 0
    for index, char in enumerate(text):
        if char.isspace():
            continue
        count += 1
        if count >= target_count:
            return index + 1
    return len(text)


def find_fallback_text_split_index(text: str, target_index: int, previous_index: int) -> Optional[int]:
    search_start = max(previous_index + PRO_SPLIT_MIN_PART_CHARS, target_index - 10)
    search_end = min(len(text) - 1, target_index + 10)
    candidates: list[int] = []

    if contains_cjk(text):
        adjusted = adjust_cjk_wrap_index(text, target_index)
        if previous_index < adjusted < len(text):
            candidates.append(adjusted)

    left_space = text.rfind(" ", previous_index + 1, target_index)
    right_space = text.find(" ", target_index, search_end)
    for candidate in (left_space, right_space):
        if candidate != -1 and search_start <= candidate <= search_end:
            candidates.append(candidate)

    candidates = [candidate for candidate in candidates if previous_index < candidate < len(text)]
    if not candidates:
        return None

    return min(candidates, key=lambda candidate: abs(candidate - target_index))


def split_text_on_sentence_boundaries(text: str) -> list[str]:
    parts = re.split(r"(?<=[。．.!?！？])\s*", text)
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
    return clean_flatten_subtitle_text(text)


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


def normalize_subtitle_text(text: str, language: str = "auto") -> str:
    return clean_subtitle_text(text, language=language)


def wrap_subtitle_text(text: str) -> str:
    if "\n" in text:
        return text

    if contains_cjk(text):
        normalized = flatten_subtitle_text(text)
        limit = PRO_CJK_MAX_LINE_CHARS
        reading_chars = count_reading_chars(normalized)
        if reading_chars <= limit:
            return normalized
        if reading_chars <= limit * PRO_MAX_LINES_PER_BLOCK:
            split_index = find_cjk_wrap_index(normalized, limit)
            if split_index <= 0 or split_index >= len(normalized):
                return normalized
            return normalized[:split_index].rstrip() + "\n" + normalized[split_index:].lstrip()
        return normalized

    if len(text) <= PRO_LATIN_MAX_LINE_CHARS or " " not in text:
        return text

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > PRO_LATIN_MAX_LINE_CHARS and len(lines) < 1:
            lines.append(current)
            current = word
        else:
            current = candidate

    if current:
        lines.append(current)

    return "\n".join(lines) if len(lines) > 1 else text


def find_cjk_wrap_index(text: str, target: int) -> int:
    character_positions = [index for index, char in enumerate(text) if not char.isspace()]
    if len(character_positions) <= 1:
        return len(text)

    target_position = character_positions[min(target, len(character_positions) - 1)]
    search_start = max(1, target_position - 8)
    search_end = min(len(text) - 1, target_position + 8)
    boundary_chars = CJK_WRAP_BOUNDARY_CHARS
    for index in range(search_end, search_start - 1, -1):
        if text[index - 1] in boundary_chars:
            return adjust_protected_phrase_wrap_index(text, index, target_position)

    return adjust_cjk_wrap_index(text, target_position)


def adjust_cjk_wrap_index(text: str, index: int) -> int:
    split_index = max(1, min(index, len(text) - 1))

    while split_index < len(text) and text[split_index] in CJK_WRAP_PROHIBITED_START_CHARS:
        split_index += 1

    while (
        split_index < len(text)
        and text[split_index - 1] in CJK_WRAP_PROHIBITED_PREFIX_CHARS
        and text[split_index] in CJK_WRAP_PROHIBITED_SUFFIX_START_CHARS
    ):
        split_index += 1

    while (
        split_index < len(text)
        and text[split_index] in CJK_WRAP_PROHIBITED_SUFFIX_START_CHARS
        and contains_cjk(text[split_index - 1])
    ):
        split_index += 1

    if split_index >= len(text) - 1:
        return len(text)

    if would_split_ascii_phrase(text, split_index):
        phrase_bounds = ascii_phrase_bounds(text, split_index)
        if phrase_bounds:
            phrase_start, phrase_end = phrase_bounds
            candidates = [
                candidate
                for candidate in (phrase_start, phrase_end)
                if candidate >= 1 and candidate < len(text) - 1
            ]
            if candidates:
                return min(candidates, key=lambda candidate: abs(candidate - index))

        left_space = text.rfind(" ", 0, split_index)
        right_space = text.find(" ", split_index)
        candidates = [
            candidate
            for candidate in (left_space, right_space)
            if candidate >= 1 and candidate < len(text) - 1
        ]
        if candidates:
            split_index = min(candidates, key=lambda candidate: abs(candidate - index))
        else:
            split_index = len(text)

    split_index = adjust_protected_phrase_wrap_index(text, split_index, index)

    return split_index


def adjust_protected_phrase_wrap_index(text: str, split_index: int, target_index: int) -> int:
    protected_bounds = protected_phrase_bounds(text, split_index)
    if protected_bounds:
        phrase_start, phrase_end = protected_bounds
        candidates = [
            candidate
            for candidate in (phrase_start, phrase_end)
            if candidate >= 1 and candidate < len(text) - 1
        ]
        if candidates:
            return min(candidates, key=lambda candidate: abs(candidate - target_index))

    return split_index


def protected_phrase_bounds(text: str, index: int) -> Optional[tuple[int, int]]:
    for phrase in PROTECTED_PHRASES:
        start = text.find(phrase)
        while start != -1:
            end = start + len(phrase)
            if start < index < end:
                return start, end
            start = text.find(phrase, start + 1)
    return None


def ascii_phrase_bounds(text: str, index: int) -> Optional[tuple[int, int]]:
    start = index
    while start > 0 and is_ascii_phrase_char(text[start - 1]):
        start -= 1

    end = index
    while end < len(text) and is_ascii_phrase_char(text[end]):
        end += 1

    phrase = text[start:end].strip()
    if " " not in phrase or len(phrase) > 48:
        return None

    return start, end


def is_ascii_phrase_char(value: str) -> bool:
    return value.isascii() and (value.isalnum() or value in " _+-'.,")


def would_split_ascii_phrase(text: str, index: int) -> bool:
    left = text[index - 1] if index > 0 else ""
    right = text[index] if index < len(text) else ""
    if left.isascii() and left.isalnum() and right.isascii() and right.isalnum():
        return True
    if left.isascii() and left.isalnum() and right == " ":
        return True
    if left == " " and right.isascii() and right.isalnum():
        return True
    return False


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
