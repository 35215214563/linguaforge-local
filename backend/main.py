from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import tempfile
import unicodedata
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Optional, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from .transcriber import SRTTranscriber

ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT_DIR / "audio"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_NAME = "large-v3"
MODEL_REPO_ID = "Systran/faster-whisper-large-v3"
MODEL_REVISION = "main"
ALLOWED_LANGUAGES = {"auto", "mixed", "ko", "ja", "vi", "zh", "en"}
ALLOWED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".oga", ".ogg", ".opus", ".wav", ".webm"}
CHUNK_SIZE = 1024 * 1024
HEADER_BYTES = 4096
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_CONCURRENT_TRANSCRIPTIONS = 1
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
TRANSCRIPTION_TIMEOUT_SECONDS = 30 * 60
STUCK_JOB_GRACE_SECONDS = 5 * 60
JOB_TTL_SECONDS = 60 * 60
MODEL_STATUS_TIMEOUT_SECONDS = 10

logger = logging.getLogger(__name__)
rate_limit_hits: defaultdict[str, deque[float]] = defaultdict(deque)
active_transcriptions = 0
active_transcriptions_lock = asyncio.Lock()
transcription_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS)
jobs: dict[str, dict[str, object]] = {}
jobs_lock = Lock()

AUDIO_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="LinguaForge Local API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=(
        r"^http://("
        r"localhost|"
        r"127\.0\.0\.1|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|"
        r"100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}|"
        r"[^/:]+\.local"
        r"):8000$"
    ),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

transcriber = SRTTranscriber(
    model_name=MODEL_NAME,
    device="cpu",
    compute_type="int8",
)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            upload_size = int(content_length)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length header"})

        if upload_size > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"detail": "File too large (max 500 MB)"})

    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-status")
def model_status() -> dict[str, object]:
    local_revision = get_cached_model_revision(MODEL_REPO_ID, MODEL_REVISION)
    try:
        remote_revision = fetch_remote_model_revision(MODEL_REPO_ID, MODEL_REVISION)
    except Exception:
        logger.warning("Failed to check remote Hugging Face model revision", exc_info=True)
        return {
            "status": "unknown",
            "model_name": MODEL_NAME,
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "local_revision": local_revision,
            "remote_revision": None,
            "is_latest": None,
            "message": "無法連線到 Hugging Face 檢查遠端版本。請確認網路連線或稍後再試。",
        }

    if local_revision is None:
        return {
            "status": "not_cached",
            "model_name": MODEL_NAME,
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "local_revision": None,
            "remote_revision": remote_revision,
            "is_latest": None,
            "message": "找不到本地模型 cache 版本資訊。模型可能尚未下載，或 cache metadata 不完整。",
        }

    is_latest = local_revision == remote_revision
    return {
        "status": "latest" if is_latest else "outdated",
        "model_name": MODEL_NAME,
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "local_revision": local_revision,
        "remote_revision": remote_revision,
        "is_latest": is_latest,
        "message": "本地模型已是遠端 main 最新版本。" if is_latest else "本地模型不是遠端 main 最新版本。",
    }


@app.delete("/output")
def clear_output() -> dict[str, Union[int, str]]:
    deleted = clear_directory_contents(OUTPUT_DIR)
    return {"status": "ok", "deleted": deleted}


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    save_output: bool = Form(True),
    mixed_ranges: str = Form(""),
    mixed_ranges_custom: bool = Form(False),
    mixed_ranges_default_language: str = Form(""),
    professional_optimization: bool = Form(False),
) -> Response:
    check_rate_limit(get_client_key(request))

    normalized_language = language.strip().lower()
    if normalized_language not in ALLOWED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="language must be one of: auto, mixed, ko, ja, vi, zh, en",
        )
    parsed_mixed_ranges = parse_mixed_ranges(
        mixed_ranges,
        mixed_ranges_custom,
        mixed_ranges_default_language,
    )

    temp_path: Optional[Path] = None
    transcription_slot_acquired = False
    future: Optional[Future[str]] = None

    try:
        suffix = get_allowed_suffix(file.filename)
        header = await file.read(HEADER_BYTES)
        validate_audio_header(header)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=AUDIO_DIR,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            bytes_written = len(header)
            temp_file.write(header)

            while chunk := await file.read(CHUNK_SIZE):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
                temp_file.write(chunk)

            temp_path = Path(temp_file.name)

        transcription_slot_acquired = await try_acquire_transcription_slot()
        if not transcription_slot_acquired:
            raise HTTPException(
                status_code=429,
                detail="Transcription server is busy. Try again after the current job finishes.",
            )

        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected before transcription started")

        future = transcription_executor.submit(
            transcriber.transcribe_to_srt,
            str(temp_path),
            normalized_language,
            parsed_mixed_ranges,
            professional_optimization,
        )
        srt_text = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=TRANSCRIPTION_TIMEOUT_SECONDS,
        )

        if save_output:
            output_path = OUTPUT_DIR / make_srt_filename(file.filename)
            await run_in_threadpool(output_path.write_text, srt_text, encoding="utf-8")

        return Response(
            content=srt_text,
            media_type="text/plain; charset=utf-8",
        )
    except HTTPException:
        raise
    except asyncio.TimeoutError as exc:
        logger.exception("Transcription timed out for file %s", file.filename)
        if future is not None:
            defer_future_cleanup(future, temp_path, transcription_slot_acquired)
            temp_path = None
            transcription_slot_acquired = False
        raise HTTPException(
            status_code=504,
            detail="Transcription timed out. Try a shorter audio file.",
        ) from exc
    except asyncio.CancelledError:
        logger.warning("Transcription request was cancelled for file %s", file.filename)
        if future is not None:
            defer_future_cleanup(future, temp_path, transcription_slot_acquired)
            temp_path = None
            transcription_slot_acquired = False
        raise
    except Exception as exc:
        logger.exception("Transcription failed for file %s", file.filename)
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Check server logs.",
        ) from exc
    finally:
        if transcription_slot_acquired:
            await release_transcription_slot()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()


@app.post("/transcribe-job")
async def create_transcription_job(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    save_output: bool = Form(True),
    mixed_ranges: str = Form(""),
    mixed_ranges_custom: bool = Form(False),
    mixed_ranges_default_language: str = Form(""),
    professional_optimization: bool = Form(False),
) -> dict[str, str]:
    cleanup_old_jobs()
    check_rate_limit(get_client_key(request))

    normalized_language = language.strip().lower()
    if normalized_language not in ALLOWED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="language must be one of: auto, mixed, ko, ja, vi, zh, en",
        )
    parsed_mixed_ranges = parse_mixed_ranges(
        mixed_ranges,
        mixed_ranges_custom,
        mixed_ranges_default_language,
    )

    temp_path: Optional[Path] = None
    transcription_slot_acquired = False

    try:
        suffix = get_allowed_suffix(file.filename)
        header = await file.read(HEADER_BYTES)
        validate_audio_header(header)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=AUDIO_DIR,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            bytes_written = len(header)
            temp_file.write(header)

            while chunk := await file.read(CHUNK_SIZE):
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File too large (max 500 MB)")
                temp_file.write(chunk)

        transcription_slot_acquired = await try_acquire_transcription_slot()
        if not transcription_slot_acquired:
            raise HTTPException(
                status_code=429,
                detail="Transcription server is busy. Try again after the current job finishes.",
            )

        job_id = uuid4().hex
        now = time.time()
        with jobs_lock:
            jobs[job_id] = {
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "filename": file.filename or "audio",
                "srt_text": "",
                "error": "",
            }

        loop = asyncio.get_running_loop()
        future = transcription_executor.submit(
            transcriber.transcribe_to_srt,
            str(temp_path),
            normalized_language,
            parsed_mixed_ranges,
            professional_optimization,
        )
        future.add_done_callback(
            lambda done_future: finish_transcription_job(
                loop=loop,
                job_id=job_id,
                future=done_future,
                temp_path=temp_path,
                original_filename=file.filename,
                save_output=save_output,
            )
        )

        return {"job_id": job_id, "status": "running"}
    except HTTPException:
        if transcription_slot_acquired:
            await release_transcription_slot()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        logger.exception("Failed to create transcription job for file %s", file.filename)
        if transcription_slot_acquired:
            await release_transcription_slot()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to create transcription job. Check server logs.",
        ) from exc
    finally:
        await file.close()


@app.get("/jobs/{job_id}")
def get_transcription_job(job_id: str) -> dict[str, object]:
    cleanup_old_jobs()
    job = get_job_or_404(job_id)
    return {
        "job_id": job_id,
        "status": job["status"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
        "filename": job["filename"],
        "error": job["error"],
    }


@app.get("/jobs/{job_id}/srt")
def get_transcription_job_srt(job_id: str) -> Response:
    cleanup_old_jobs()
    job = get_job_or_404(job_id)

    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "Transcription failed")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Transcription job is not finished yet")

    return Response(
        content=str(job["srt_text"] or ""),
        media_type="text/plain; charset=utf-8",
    )


def make_srt_filename(original_filename: Optional[str]) -> str:
    stem = Path(original_filename or "transcription").stem or "transcription"
    normalized_stem = unicodedata.normalize("NFKC", stem)
    safe_stem = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in normalized_stem
    )
    safe_stem = re.sub(r"_+", "_", safe_stem).strip("._-")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_suffix = uuid4().hex[:8]
    return f"{safe_stem or 'transcription'}_{timestamp}_{unique_suffix}.srt"


def clear_directory_contents(directory: Path) -> int:
    deleted = 0
    directory.mkdir(exist_ok=True)

    for child in directory.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
        deleted += 1

    return deleted


def get_cached_model_revision(repo_id: str, revision: str = "main") -> Optional[str]:
    repo_cache_dir = get_hf_hub_cache_dir() / f"models--{repo_id.replace('/', '--')}"
    ref_path = repo_cache_dir / "refs" / revision
    if ref_path.is_file():
        cached_revision = ref_path.read_text(encoding="utf-8").strip()
        if cached_revision:
            return cached_revision

    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None

    latest_snapshot = max(snapshots, key=lambda path: path.stat().st_mtime)
    return latest_snapshot.name


def get_hf_hub_cache_dir() -> Path:
    if hub_cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(hub_cache)
    if hf_home := os.environ.get("HF_HOME"):
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def fetch_remote_model_revision(repo_id: str, revision: str = "main") -> str:
    url = f"https://huggingface.co/api/models/{repo_id}/revision/{revision}"
    request = UrlRequest(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "linguaforge-local/1.0",
        },
    )

    try:
        with urlopen(request, timeout=MODEL_STATUS_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Hugging Face 回傳 HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("連線逾時") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Hugging Face 回傳格式不是 JSON") from exc

    remote_revision = payload.get("sha")
    if not isinstance(remote_revision, str) or not remote_revision:
        raise RuntimeError("Hugging Face 回傳裡沒有 sha")

    return remote_revision


def parse_mixed_ranges(
    raw_ranges: str,
    custom_languages: bool = False,
    default_language: str = "",
) -> list[tuple[float, float, str]]:
    normalized_default_language = normalize_mixed_ranges_default_language(
        default_language,
        custom_languages,
    )
    if not raw_ranges.strip():
        return []

    ranges: list[tuple[float, float, str]] = []
    range_items = split_mixed_range_items(raw_ranges, custom_languages)
    for line_number, raw_range in enumerate(range_items, start=1):
        item = raw_range.strip()
        if not item:
            continue

        if custom_languages:
            if normalized_default_language:
                if re.search(r"\s+(auto|mixed|ko|ja|vi|zh|en)$", item, flags=re.IGNORECASE):
                    raise HTTPException(
                        status_code=400,
                        detail="已選預設例外語言時，每行只需要輸入時間段，例如：0-3",
                    )
                range_text = item
                range_language = normalized_default_language
            else:
                match = re.match(r"^(.+?)\s+(auto|mixed|ko|ja|vi|zh|en)$", item, flags=re.IGNORECASE)
                if not match:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"mixed_ranges line {line_number} must look like: "
                            "0-3 ja, 12-18 mixed, or 00:00:00,000 --> 00:00:03,000 ja"
                        ),
                    )
                range_text, range_language = match.group(1).strip(), match.group(2).lower()
                if range_language not in ALLOWED_LANGUAGES:
                    raise HTTPException(
                        status_code=400,
                        detail="mixed_ranges language must be one of: auto, mixed, ko, ja, vi, zh, en",
                    )
        else:
            range_text = item
            range_language = "mixed"
            if re.search(r"\s+(auto|mixed|ko|ja|vi|zh|en)$", range_text, flags=re.IGNORECASE):
                raise HTTPException(
                    status_code=400,
                    detail="勾選自定義每段語言時才可以輸入語言碼，例如：0-3 ja",
                )

        start, end = parse_range_value(range_text)

        if end <= start:
            raise HTTPException(status_code=400, detail="mixed_ranges end time must be after start time")
        ranges.append((start, end, range_language))

    ensure_ranges_do_not_overlap(ranges)

    return ranges


def split_mixed_range_items(raw_ranges: str, custom_languages: bool) -> list[str]:
    if custom_languages:
        return raw_ranges.splitlines()

    return re.split(r"[\n;]+|,\s+", raw_ranges)


def normalize_mixed_ranges_default_language(default_language: str, custom_languages: bool) -> str:
    normalized_language = default_language.strip().lower()
    if not normalized_language:
        return ""

    if normalized_language not in ALLOWED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="mixed_ranges_default_language must be one of: auto, mixed, ko, ja, vi, zh, en",
        )

    if not custom_languages:
        raise HTTPException(
            status_code=400,
            detail="mixed_ranges_default_language can only be used when mixed_ranges_custom=true",
        )

    return normalized_language


def parse_range_value(raw_range: str) -> tuple[float, float]:
    range_parts = split_range_value(raw_range)
    if range_parts is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "mixed_ranges format must look like 12-18, "
                "00:01:12-00:01:20, or 00:00:00,000 --> 00:00:03,000"
            ),
        )

    start = parse_time_value(range_parts[0])
    end = parse_time_value(range_parts[1])
    return start, end


def split_range_value(raw_range: str) -> Optional[tuple[str, str]]:
    value = raw_range.strip()
    if not value:
        return None

    arrow_parts = re.split(r"\s*-->\s*", value)
    if len(arrow_parts) == 2 and arrow_parts[0].strip() and arrow_parts[1].strip():
        return arrow_parts[0].strip(), arrow_parts[1].strip()
    if len(arrow_parts) > 2:
        return None

    match = re.match(r"^(.+?)\s*(?:~|到|至)\s*(.+)$", value)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    match = re.match(r"^(.+?)\s*-\s*(.+)$", value)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return None


def ensure_ranges_do_not_overlap(ranges: list[tuple[float, float, str]]) -> None:
    sorted_ranges = sorted(ranges, key=lambda item: item[0])
    previous_end: Optional[float] = None
    for start, end, _language in sorted_ranges:
        if previous_end is not None and start < previous_end:
            raise HTTPException(status_code=400, detail="mixed_ranges must not overlap")
        previous_end = end


def parse_time_value(raw_value: str) -> float:
    value = raw_value.strip().replace(",", ".")
    if not value:
        raise HTTPException(status_code=400, detail="mixed_ranges contains an empty time value")

    if ":" not in value:
        try:
            seconds = float(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid time value: {raw_value}") from exc
        if seconds < 0:
            raise HTTPException(status_code=400, detail="mixed_ranges time values must be >= 0")
        return seconds

    parts = value.split(":")
    if len(parts) not in {2, 3, 4}:
        raise HTTPException(status_code=400, detail=f"Invalid time value: {raw_value}")

    if len(parts) == 2:
        minutes = parse_integer_time_part(parts[0], raw_value)
        seconds = parse_seconds_time_part(parts[1], raw_value)
        if seconds >= 60:
            raise HTTPException(status_code=400, detail=f"Invalid time value: {raw_value}")
        return (minutes * 60) + seconds

    if len(parts) == 3:
        hours = parse_integer_time_part(parts[0], raw_value)
        minutes = parse_integer_time_part(parts[1], raw_value)
        seconds = parse_seconds_time_part(parts[2], raw_value)
        if minutes >= 60 or seconds >= 60:
            raise HTTPException(status_code=400, detail=f"Invalid time value: {raw_value}")
        return (hours * 3600) + (minutes * 60) + seconds

    hours = parse_integer_time_part(parts[0], raw_value)
    minutes = parse_integer_time_part(parts[1], raw_value)
    seconds = parse_integer_time_part(parts[2], raw_value)
    milliseconds = parse_integer_time_part(parts[3], raw_value)
    if minutes >= 60 or seconds >= 60 or not re.match(r"^\d{1,3}$", parts[3]) or milliseconds > 999:
        raise HTTPException(status_code=400, detail=f"Invalid time value: {raw_value}")

    return (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000)


def parse_integer_time_part(value: str, raw_value: str) -> int:
    if not re.match(r"^\d+$", value):
        raise HTTPException(status_code=400, detail=f"Invalid time value: {raw_value}")
    return int(value)


def parse_seconds_time_part(value: str, raw_value: str) -> float:
    if not re.match(r"^\d+(?:\.\d{1,3})?$", value):
        raise HTTPException(status_code=400, detail=f"Invalid time value: {raw_value}")
    return float(value)


def get_allowed_suffix(original_filename: Optional[str]) -> str:
    suffix = Path(original_filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Use mp3, m4a, wav, flac, ogg, opus, webm, aac, or mp4.",
        )
    return suffix


def validate_audio_header(header: bytes) -> None:
    if not header:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if is_supported_audio_header(header):
        return

    raise HTTPException(status_code=400, detail="Unsupported or invalid audio file")


def is_supported_audio_header(header: bytes) -> bool:
    if header.startswith(b"ID3"):
        return True
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return True
    if header.startswith(b"fLaC"):
        return True
    if header.startswith(b"OggS"):
        return True
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    if len(header) >= 12 and header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return True
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return True

    return False


def get_client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_rate_limit(client_key: str) -> None:
    now = time.monotonic()
    cleanup_rate_limit_hits(now)
    hits = rate_limit_hits[client_key]

    while hits and now - hits[0] > RATE_LIMIT_WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many transcription requests. Try again later.")

    hits.append(now)


def cleanup_rate_limit_hits(now: float) -> None:
    for key, hits in list(rate_limit_hits.items()):
        while hits and now - hits[0] > RATE_LIMIT_WINDOW_SECONDS:
            hits.popleft()

        if not hits:
            rate_limit_hits.pop(key, None)


def get_job_or_404(job_id: str) -> dict[str, object]:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Transcription job not found")
        return dict(job)


def cleanup_old_jobs() -> None:
    now = time.time()
    stale_running_after = TRANSCRIPTION_TIMEOUT_SECONDS + STUCK_JOB_GRACE_SECONDS
    with jobs_lock:
        expired_job_ids = []
        for job_id, job in jobs.items():
            status = job.get("status")
            if status in {"done", "error"} and now - float(job.get("updated_at", now)) > JOB_TTL_SECONDS:
                expired_job_ids.append(job_id)
            elif status == "running" and now - float(job.get("created_at", now)) > stale_running_after:
                job.update(
                    {
                        "status": "error",
                        "error": "Transcription timed out. Try a shorter audio file.",
                        "updated_at": now,
                    }
                )

        for job_id in expired_job_ids:
            jobs.pop(job_id, None)


async def try_acquire_transcription_slot() -> bool:
    global active_transcriptions

    async with active_transcriptions_lock:
        if active_transcriptions >= MAX_CONCURRENT_TRANSCRIPTIONS:
            return False
        active_transcriptions += 1
        return True


async def release_transcription_slot() -> None:
    global active_transcriptions

    async with active_transcriptions_lock:
        active_transcriptions = max(0, active_transcriptions - 1)


def defer_future_cleanup(
    future: Future[str],
    temp_path: Optional[Path],
    release_slot: bool,
) -> None:
    loop = asyncio.get_running_loop()

    def cleanup(done_future: Future[str]) -> None:
        try:
            done_future.result()
        except Exception:
            logger.exception("Background transcription finished after request ended")

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        if release_slot:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(release_transcription_slot()))

    future.add_done_callback(cleanup)


def finish_transcription_job(
    loop: asyncio.AbstractEventLoop,
    job_id: str,
    future: Future[str],
    temp_path: Optional[Path],
    original_filename: Optional[str],
    save_output: bool,
) -> None:
    try:
        srt_text = future.result()
        with jobs_lock:
            should_update_job = job_id in jobs and jobs[job_id].get("status") == "running"

        if save_output and should_update_job:
            output_path = OUTPUT_DIR / make_srt_filename(original_filename)
            output_path.write_text(srt_text, encoding="utf-8")

        with jobs_lock:
            if should_update_job and job_id in jobs and jobs[job_id].get("status") == "running":
                jobs[job_id].update(
                    {
                        "status": "done",
                        "srt_text": srt_text,
                        "error": "",
                        "updated_at": time.time(),
                    }
                )
    except Exception:
        logger.exception("Background transcription job failed for file %s", original_filename)
        with jobs_lock:
            if job_id in jobs and jobs[job_id].get("status") == "running":
                jobs[job_id].update(
                    {
                        "status": "error",
                        "srt_text": "",
                        "error": "Transcription failed. Check server logs.",
                        "updated_at": time.time(),
                    }
                )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        loop.call_soon_threadsafe(lambda: asyncio.create_task(release_transcription_slot()))
