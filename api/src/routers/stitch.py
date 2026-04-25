"""POST /api/stitch, GET /api/video, GET /api/captions (issue fzm, fw-2it)."""

import asyncio
import functools
import html
import json
import pathlib
import textwrap

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.stitch_service import StitchService

router = APIRouter(prefix="/api")

_stitch_service = StitchService(ui_dir=settings.data_dir)



def _wrap_caption_text(text: str, max_chars: int = 42, max_lines: int = 2) -> list[str]:
    """Wrap one caption into at most two readable subtitle lines."""
    words = text.strip().split()
    if not words:
        return []

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) <= max_chars or not current:
            current.append(word)
            continue

        lines.append(" ".join(current))
        current = [word]

        if len(lines) >= max_lines:
            break

    if current and len(lines) < max_lines:
        lines.append(" ".join(current))

    visible = " ".join(lines)
    if len(visible) < len(text.strip()) and lines:
        lines[-1] = lines[-1].rstrip(".,;:") + "..."

    return lines[:max_lines]


def _segments_to_vtt(segments: list[dict]) -> str:
    """Convert transcript segments to WebVTT.

    Each cue only shows the current subtitle text. Long captions are wrapped
    into at most two lines. End times are clamped to the next cue start time
    to prevent browsers from displaying overlapping subtitles.
    """
    segs = [
        s for s in segments
        if s.get("text", "").strip()
        and "start" in s
        and "end" in s
    ]
    segs = sorted(segs, key=lambda s: float(s["start"]))

    if not segs:
        return "WEBVTT\n"

    lines = ["WEBVTT", ""]

    for i, seg in enumerate(segs):
        start_s = float(seg["start"])
        raw_end_s = float(seg["end"])

        next_start_s = None
        if i + 1 < len(segs):
            next_start_s = float(segs[i + 1]["start"])

        end_s = raw_end_s
        if next_start_s is not None and next_start_s > start_s:
            end_s = min(raw_end_s, next_start_s)

        if end_s <= start_s:
            end_s = start_s + 0.5

        lines.append(str(i + 1))
        lines.append(f"{_format_vtt_time(start_s)} --> {_format_vtt_time(end_s)}")
        lines.extend(_wrap_caption_text(seg.get("text", "").strip()))
        lines.append("")

    return "\n".join(lines)


def _format_vtt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for WebVTT."""
    seconds = max(0.0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))

    if ms == 1000:
        s += 1
        ms = 0
    if s == 60:
        m += 1
        s = 0
    if m == 60:
        h += 1
        m = 0

    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def _serve_captions(vtt_dir: pathlib.Path, json_fallback_dir: pathlib.Path, video_id: str):
    """Serve VTT captions from disk. Falls back to generating from JSON if VTT doesn't exist yet."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    vtt_path = vtt_dir / f"{title}.vtt"

    # Serve existing VTT file directly
    if vtt_path.exists():
        return PlainTextResponse(vtt_path.read_text(), media_type="text/vtt")

    # Fallback: generate VTT from transcript JSON
    json_path = json_fallback_dir / f"{title}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="No captions available")

    data = json.loads(json_path.read_text())
    segments = data.get("segments", [])
    vtt = _segments_to_vtt(segments)

    # Persist so we don't regenerate next time
    vtt_dir.mkdir(parents=True, exist_ok=True)
    vtt_path.write_text(vtt)
    return PlainTextResponse(vtt, media_type="text/vtt")


def _compute_speech_offset(title: str) -> float:
    """Compute the timing offset between YouTube captions and Whisper segments.

    YouTube captions have accurate start times (e.g. 4.8s into the video),
    while Whisper starts at 0.0s. Returns the offset to add to Whisper timestamps.
    """
    yt_path = settings.youtube_captions_dir / f"{title}.txt"
    whisper_path = settings.transcriptions_dir / f"{title}.json"

    if not yt_path.exists() or not whisper_path.exists():
        return 0.0

    # First YouTube caption start time
    first_line = yt_path.read_text().split("\n", 1)[0].strip()
    if not first_line:
        return 0.0
    yt_start = json.loads(first_line).get("start", 0.0)

    # First Whisper segment start time
    whisper_data = json.loads(whisper_path.read_text())
    segments = whisper_data.get("segments", [])
    whisper_start = segments[0]["start"] if segments else 0.0

    return yt_start - whisper_start



def _find_latest_tts_align_report(title: str):
    """Find the newest TTS alignment report for a title."""
    root = settings.data_dir / "tts_audio" / "chatterbox"
    if not root.exists():
        return None

    candidates = []
    for config_dir in root.iterdir():
        if not config_dir.is_dir():
            continue
        candidate = config_dir / f"{title}.align.json"
        if candidate.exists():
            candidates.append(candidate)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _segments_to_vtt_from_tts_schedule(segments: list[dict], align_report: dict) -> str:
    """Convert translated segments to VTT using the TTS assembly timeline."""
    details = align_report.get("segments", [])

    if not segments or not details or len(segments) != len(details):
        return _segments_to_vtt(segments)

    initial_offset_s = align_report.get("initial_offset_s")
    if initial_offset_s is None:
        initial_offset_s = segments[0].get("start", 0.0)

    cursor = float(initial_offset_s)
    lines = ["WEBVTT", ""]

    for cue_no, detail in enumerate(details, start=1):
        idx = int(detail.get("index", cue_no - 1))
        if idx < 0 or idx >= len(segments):
            continue

        duration_s = (
            detail.get("scheduled_duration_s")
            or detail.get("target_sec")
            or (
                float(segments[idx].get("end", 0.0))
                - float(segments[idx].get("start", 0.0))
            )
        )
        duration_s = max(0.1, float(duration_s))

        start_s = cursor
        end_s = cursor + duration_s
        cursor = end_s

        text = segments[idx].get("text", "").strip()
        if not text:
            continue

        lines.append(str(cue_no))
        lines.append(f"{_format_vtt_time(start_s)} --> {_format_vtt_time(end_s)}")
        lines.extend(_wrap_caption_text(text))
        lines.append("")

    return "\n".join(lines)


@router.get("/captions/{video_id}")
async def get_captions(video_id: str):
    """Serve translated target-language captions as WebVTT.

    If a TTS alignment report exists, generate captions from the dubbed TTS
    timeline so subtitles stay synchronized with the assembled audio.
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    vtt_dir = settings.dubbed_captions_dir
    vtt_path = vtt_dir / f"{title}.vtt"

    if vtt_path.exists():
        return PlainTextResponse(vtt_path.read_text(), media_type="text/vtt")

    json_path = settings.translations_dir / f"{title}.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Translated captions not found")

    data = json.loads(json_path.read_text())
    segments = data.get("segments", [])

    align_path = _find_latest_tts_align_report(title)
    if align_path is not None:
        align_report = json.loads(align_path.read_text())
        vtt = _segments_to_vtt_from_tts_schedule(segments, align_report)
    else:
        offset = _compute_speech_offset(title)
        if offset > 0:
            segments = [
                {**seg, "start": seg["start"] + offset, "end": seg["end"] + offset}
                for seg in segments
            ]
        vtt = _segments_to_vtt(segments)

    vtt_dir.mkdir(parents=True, exist_ok=True)
    vtt_path.write_text(vtt)
    return PlainTextResponse(vtt, media_type="text/vtt")


def _youtube_captions_to_vtt(caption_path: pathlib.Path) -> str:
    """Convert YouTube line-delimited JSON captions to WebVTT.

    YouTube format: {"text": "...", "start": float, "duration": float} per line.
    Each cue only shows current text, wrapped to at most two lines. End times
    are clamped to the next cue start time to avoid overlap.
    """
    segs: list[tuple[float, float, str]] = []

    for line in caption_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue

        seg = json.loads(line)
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0))
        duration = float(seg.get("duration", 0))

        if text and duration > 0:
            segs.append((start, start + duration, text))

    segs = sorted(segs, key=lambda x: x[0])

    if not segs:
        return "WEBVTT\n"

    lines_out = ["WEBVTT", ""]

    for i, (start, raw_end, text) in enumerate(segs):
        next_start = segs[i + 1][0] if i + 1 < len(segs) else None

        end = raw_end
        if next_start is not None and next_start > start:
            end = min(raw_end, next_start)

        if end <= start:
            end = start + 0.5

        lines_out.append(str(i + 1))
        lines_out.append(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}")
        lines_out.extend(_wrap_caption_text(text))
        lines_out.append("")

    return "\n".join(lines_out)


@router.get("/captions/{video_id}/original")
async def get_original_captions(video_id: str):
    """Serve original (source-language) captions as WebVTT.

    Prefers: existing VTT on disk > YouTube captions (accurate timestamps) > Whisper transcription.
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    # 1. Generate from YouTube captions (most accurate timestamps)
    yt_caption_path = settings.youtube_captions_dir / f"{title}.txt"
    if yt_caption_path.exists():
        vtt = _youtube_captions_to_vtt(yt_caption_path)
        return PlainTextResponse(vtt, media_type="text/vtt")

    # 2. Fall back to Whisper transcription
    whisper_path = settings.transcriptions_dir / f"{title}.json"
    if not whisper_path.exists():
        raise HTTPException(status_code=404, detail="No captions available")
    data = json.loads(whisper_path.read_text())
    return PlainTextResponse(
        _segments_to_vtt(data.get("segments", [])), media_type="text/vtt",
    )


@router.post("/stitch/{video_id}")
async def stitch_endpoint(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Replace video audio with dubbed TTS audio.

    *config* selects which TTS audio to use (opaque directory name).
    """
    videos_dir = settings.videos_dir
    output_dir = settings.dubbed_videos_dir / config
    output_dir.mkdir(parents=True, exist_ok=True)

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    output_path = output_dir / f"{title}.mp4"

    if output_path.exists():
        return {"video_id": video_id, "video_path": str(output_path), "config": config}

    video_path = str(videos_dir / f"{title}.mp4")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        functools.partial(
            _stitch_service.stitch_audio_only,
            video_path,
            str(audio_path),
            str(output_path),
        ),
    )

    return {"video_id": video_id, "video_path": str(output_path), "config": config}


def _serve_video(file_path: pathlib.Path, request: Request):
    """Serve a video file with HTTP range request support."""
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1
        chunk_size = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    read_size = min(8192, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    return FileResponse(
        str(file_path),
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


@router.get("/video/{video_id}")
async def get_video(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the dubbed MP4."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    video_path = settings.dubbed_videos_dir / config / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Dubbed video not yet generated")

    return _serve_video(video_path, request)


@router.get("/video/{video_id}/original")
async def get_original_video(video_id: str, request: Request):
    """Stream the original downloaded MP4."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Original video not found")

    return _serve_video(video_path, request)
