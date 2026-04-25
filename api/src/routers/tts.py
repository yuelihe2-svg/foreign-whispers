"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService

router = APIRouter(prefix="/api")


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str | None = Query(None),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = str(trans_dir / f"{title}.json")

    # Task 5: Build speaker-to-voice mapping based on available reference WAVs.
    speaker_mapping = {}
    source_file = Path(source_path)

    if speaker_wav:
        # Explicit speaker_wav query parameter overrides automatic per-speaker mapping.
        if source_file.exists():
            transcript = json.loads(source_file.read_text())
            speakers = {
                seg.get("speaker", "SPEAKER_00")
                for seg in transcript.get("segments", [])
            }
        else:
            speakers = {"SPEAKER_00"}

        for spk in speakers:
            speaker_mapping[spk] = speaker_wav

    elif source_file.exists():

        transcript = json.loads(source_file.read_text())
        lang = transcript.get("language", "es")
        
        # Navigate to pipeline_data/speakers/{lang}

        speakers_dir = settings.speakers_dir / lang
        
        if speakers_dir.exists():
            available_voices = sorted(list(speakers_dir.glob("*.wav")))
            if available_voices:
                # Extract unique speakers from the transcript segments

                speakers = set()
                for seg in transcript.get("segments", []):
                    if "speaker" in seg:
                        speakers.add(seg["speaker"])
                
                # Apply Index-based Mapping strategy

                for spk in sorted(list(speakers)):
                    if spk == "<NO SPEAKER>":
                        speaker_mapping[spk] = str(available_voices[0])
                    else:
                        try:
                            # Parse 'SPEAKER_01' -> 1

                            spk_idx = int(spk.split("_")[-1])
                        except ValueError:
                            spk_idx = 0
                        
                        # Round-robin assignment using modulo operator

                        assigned_voice = available_voices[spk_idx % len(available_voices)]
                        speaker_mapping[spk] = str(assigned_voice)

    # Pass the constructed mapping to the TTS service

    await _run_in_threadpool(
        None, svc.text_file_to_speech, source_path, str(audio_dir), alignment=alignment, speaker_mapping=speaker_mapping
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")