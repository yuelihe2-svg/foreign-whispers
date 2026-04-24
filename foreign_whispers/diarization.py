"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import logging

logger = logging.getLogger(__name__)


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        from pyannote.audio import Pipeline
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline    = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(audio_path)
        return [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []
    
def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    # Initialize the list for the output segments
    # 初始化用于存放输出片段的列表
    merged_segments = []

    for t_seg in segments:
        # Get start and end times for the text segment
        # 获取字幕片段的起止时间
        t_start = t_seg["start"]
        t_end = t_seg.get("end", t_start + 1.0)
        
        # Default to SPEAKER_00 if no overlap is found or diarization is empty
        # 如果没有找到重叠部分或声纹列表为空，默认分配为 SPEAKER_00
        best_speaker = "SPEAKER_00" 
        max_overlap = 0.0

        # Compare with every diarization segment to find the max overlap
        # 与每一个声纹片段进行对比，寻找重叠时间最长的说话人
        for d_seg in diarization:
            d_start = d_seg["start_s"]
            d_end = d_seg["end_s"]

            # Calculate intersection window (overlap duration)
            # 计算两个时间段的交集（重叠时长）
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap_duration = max(0.0, overlap_end - overlap_start)

            # Update best speaker if this overlap is the largest so far
            # 如果当前重叠时长大于历史最大值，则更新最佳说话人
            if overlap_duration > max_overlap:
                max_overlap = overlap_duration
                best_speaker = d_seg["speaker"]

        # Create a shallow copy of the segment and inject the speaker field
        # 浅拷贝原有的字幕字典并注入 speaker 字段，以避免修改原始输入
        new_seg = dict(t_seg)
        new_seg["speaker"] = best_speaker
        merged_segments.append(new_seg)

    return merged_segments