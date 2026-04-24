"""POST /api/diarize/{video_id} — speaker diarization (issue fw-lua)."""

import asyncio
import json
import subprocess

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService

# assignment function finished in Task 1
from foreign_whispers.diarization import assign_speakers

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track.

    Steps:
    1. Extract audio from video via ffmpeg
    2. Run pyannote diarization
    3. Cache and return speaker segments
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    # Return cached result
    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
            skipped=True,
        )

    # ---- YOUR CODE HERE ----
    # Step 1: Extract audio from video via ffmpeg
    # 第一阶段：使用 ffmpeg 从视频中提取 16kHz 单声道音频
    video_path = settings.videos_dir / f"{title}.mp4"
    audio_path = diar_dir / f"{title}.wav"
    
    if not audio_path.exists():
        # -vn: no video, -acodec pcm_s16le: 16-bit PCM, -ar 16000: 16kHz sample rate
        subprocess.run([
            "ffmpeg", "-i", str(video_path), 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", 
            "-y", str(audio_path)
        ], check=True, capture_output=True)

    # Step 2: Run diarization via AlignmentService
    # 第二阶段：调用服务运行声纹识别
    diar_segments = _alignment_service.diarize(str(audio_path))

    # Step 3: Extract unique speakers
    # 第三阶段：提取并排序不重复的说话人列表
    speakers = sorted(list(set(s["speaker"] for s in diar_segments)))

    # Step 4: Cache result to disk
    # 第四阶段：将结果序列化并保存到缓存目录
    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result))

    # Task 3: Merge speaker labels into the transcription JSON.
    # 任务 3：将说话人标签合并到 Whisper 生成的字幕 JSON 中。
    transcript_path = settings.transcriptions_dir / "whisper" / f"{title}.json"
    if transcript_path.exists():
        transcript = json.loads(transcript_path.read_text())
        
        # Assign speakers to the transcription segments based on temporal overlap.
        # 根据时间重叠度，为字幕片段分配对应的说话人。
        labeled_segments = assign_speakers(transcript.get("segments", []), diar_segments)
        transcript["segments"] = labeled_segments
        
        # Write the updated transcription (now containing speaker metadata) back to disk.
        # 将更新后的字幕（现已包含说话人元数据）重新写回磁盘。
        transcript_path.write_text(json.dumps(transcript))

    # Step 5: Return standard response.
    # 第五阶段：返回标准的响应对象。
    return DiarizeResponse(
        video_id=video_id, 
        speakers=speakers, 
        segments=diar_segments,
        skipped=False
    )
    # ---- END YOUR CODE ----

