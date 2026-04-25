"""POST /api/diarize/{video_id} — speaker diarization."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")


def _diarizations_dir() -> Path:
    return getattr(settings, "diarizations_dir", settings.data_dir / "diarizations")


def _extract_audio(video_path: Path, wav_path: Path) -> None:
    """Extract mono 16 kHz WAV audio for diarization."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists() and wav_path.stat().st_size > 0:
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _run_pyannote(wav_path: Path) -> list[dict]:
    """Run pyannote speaker diarization and return serializable segments."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is not installed in the API container."
        ) from exc

    token = getattr(settings, "fw_hf_token", None) or os.environ.get("FW_HF_TOKEN")
    if not token:
        raise RuntimeError("FW_HF_TOKEN is required for pyannote diarization.")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    diarization = pipeline(str(wav_path))

    segments: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start_s": float(turn.start),
                "end_s": float(turn.end),
                "speaker": str(speaker),
            }
        )

    return segments


def _load_diarization_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("segments", [])


def _merge_speakers_into_transcription(title: str, diarization_segments: list[dict]) -> int:
    """Update Whisper transcription JSON with speaker labels when available."""
    transcription_path = settings.transcriptions_dir / f"{title}.json"

    if not transcription_path.exists():
        return 0

    data = json.loads(transcription_path.read_text(encoding="utf-8"))
    segments = data.get("segments", [])

    if not segments:
        return 0

    labeled_segments = assign_speakers(segments, diarization_segments)
    data["segments"] = labeled_segments

    transcription_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return sum("speaker" in segment for segment in labeled_segments)


@router.post("/diarize/{video_id}")
async def diarize_endpoint(video_id: str):
    """Run speaker diarization, cache the result, and merge speaker labels."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Downloaded video not found: {video_path.name}",
        )

    diar_dir = _diarizations_dir()
    diar_dir.mkdir(parents=True, exist_ok=True)

    wav_path = diar_dir / f"{title}.wav"
    json_path = diar_dir / f"{title}.json"

    skipped = json_path.exists()

    try:
        await asyncio.to_thread(_extract_audio, video_path, wav_path)

        if skipped:
            diarization_segments = _load_diarization_segments(json_path)
        else:
            diarization_segments = await asyncio.to_thread(_run_pyannote, wav_path)
            payload = {
                "video_id": video_id,
                "title": title,
                "segments": diarization_segments,
            }
            json_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        merged_count = _merge_speakers_into_transcription(title, diarization_segments)
        speakers = sorted({segment.get("speaker") for segment in diarization_segments if segment.get("speaker")})

        return {
            "video_id": video_id,
            "title": title,
            "status": "ok",
            "speakers": speakers,
            "diarization_segments": len(diarization_segments),
            "merged_transcription_segments": merged_count,
            "skipped": skipped,
            "audio_path": str(wav_path),
            "diarization_path": str(json_path),
        }

    except subprocess.CalledProcessError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract audio for diarization: {exc.stderr.decode(errors='ignore')}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
