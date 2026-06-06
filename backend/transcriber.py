from __future__ import annotations

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps


SAMPLE_RATE = 16000
MIN_MIXED_SEGMENT_SECONDS = 0.35


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
        mixed_ranges: list[tuple[float, float]] | None = None,
    ) -> str:
        if language == "mixed":
            return self._transcribe_mixed_to_srt(audio_path)

        if mixed_ranges:
            return self._transcribe_with_mixed_ranges_to_srt(audio_path, language, mixed_ranges)

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
        return segments_to_srt(segments)

    def _transcribe_mixed_to_srt(self, audio_path: str) -> str:
        audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        blocks: list[str] = []
        self._append_mixed_audio_to_blocks(blocks, audio, offset=0.0)
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _transcribe_with_mixed_ranges_to_srt(
        self,
        audio_path: str,
        language: str,
        mixed_ranges: list[tuple[float, float]],
    ) -> str:
        audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)
        total_duration = len(audio) / SAMPLE_RATE
        ranges = normalize_ranges(mixed_ranges, total_duration)
        if not ranges:
            return self.transcribe_to_srt(audio_path, language)

        blocks: list[str] = []
        cursor = 0.0

        for start, end in ranges:
            if start > cursor:
                self._append_target_audio_to_blocks(
                    blocks,
                    audio,
                    start=cursor,
                    end=start,
                    language=language,
                )

            self._append_mixed_audio_to_blocks(
                blocks,
                slice_audio(audio, start, end),
                offset=start,
            )
            cursor = end

        if cursor < total_duration:
            self._append_target_audio_to_blocks(
                blocks,
                audio,
                start=cursor,
                end=total_duration,
                language=language,
            )

        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _append_target_audio_to_blocks(
        self,
        blocks: list[str],
        audio,
        start: float,
        end: float,
        language: str,
    ) -> None:
        chunk = slice_audio(audio, start, end)
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
        append_segments_to_blocks(blocks, segments, offset=start, max_end=end)

    def _append_mixed_audio_to_blocks(self, blocks: list[str], audio, offset: float) -> None:
        vad_options = VadOptions(
            threshold=0.5,
            min_silence_duration_ms=500,
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

        base_offset = offset
        for speech in speech_timestamps:
            start_sample = max(0, int(speech["start"]))
            end_sample = min(len(audio), int(speech["end"]))
            if end_sample <= start_sample:
                continue

            chunk = audio[start_sample:end_sample]
            chunk_duration = len(chunk) / SAMPLE_RATE
            if chunk_duration < MIN_MIXED_SEGMENT_SECONDS:
                continue

            speech_start = start_sample / SAMPLE_RATE
            speech_end = end_sample / SAMPLE_RATE
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
                max_end=base_offset + speech_end,
            )


def segments_to_srt(segments) -> str:
    blocks: list[str] = []
    append_segments_to_blocks(blocks, segments)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def append_segments_to_blocks(blocks: list[str], segments, offset: float = 0.0, max_end: float | None = None) -> None:
    for segment in segments:
        text = (segment.text or "").strip()
        if not text:
            continue

        index = len(blocks) + 1
        start = offset + max(float(segment.start or 0), 0.0)
        end = offset + max(float(segment.end or 0), 0.0)
        if max_end is not None:
            start = min(start, max_end)
            end = min(end, max_end)
        if end <= start:
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


def slice_audio(audio, start: float, end: float):
    start_sample = max(0, int(start * SAMPLE_RATE))
    end_sample = min(len(audio), int(end * SAMPLE_RATE))
    return audio[start_sample:end_sample]


def normalize_ranges(ranges: list[tuple[float, float]], total_duration: float) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for start, end in ranges:
        start = max(0.0, min(float(start), total_duration))
        end = max(0.0, min(float(end), total_duration))
        if end - start >= MIN_MIXED_SEGMENT_SECONDS:
            normalized.append((start, end))

    normalized.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))

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
