"""POST /api/download — download YouTube video + captions (issue by5)."""

from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.video_registry import get_video
from api.src.schemas.download import DownloadRequest, DownloadResponse
from api.src.services.download_service import DownloadService

router = APIRouter(prefix="/api")

_download_service = DownloadService(ui_dir=settings.data_dir)


def _extract_youtube_video_id(url: str) -> str | None:
    """Extract an 11-character YouTube video id without calling YouTube.

    This allows the download endpoint to reuse local cached artifacts even when
    yt-dlp cannot access YouTube due to bot checks or missing cookies.
    """
    parsed = urlparse(url)

    if parsed.netloc.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/", 1)[0]
        return video_id if len(video_id) == 11 else None

    if "youtube.com" in parsed.netloc:
        video_ids = parse_qs(parsed.query).get("v", [])
        if video_ids and len(video_ids[0]) == 11:
            return video_ids[0]

    return None


@router.post("/download", response_model=DownloadResponse)
async def download_endpoint(body: DownloadRequest):
    """Download video and captions, returning video_id and caption segments.

    If the requested video already exists in the local pipeline cache, return it
    without contacting YouTube. This keeps the demo usable even when yt-dlp is
    blocked by YouTube bot checks.
    """
    videos_dir = settings.videos_dir
    captions_dir = settings.youtube_captions_dir
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: use registry + local files before calling yt-dlp.
    cached_video_id = _extract_youtube_video_id(body.url)
    cached_entry = get_video(cached_video_id) if cached_video_id else None

    if cached_video_id and cached_entry:
        cached_stem = cached_entry.title
        cached_video_path = videos_dir / f"{cached_stem}.mp4"
        cached_caption_path = captions_dir / f"{cached_stem}.txt"

        if cached_video_path.exists() and cached_caption_path.exists():
            segments = _download_service.read_caption_segments(cached_caption_path)
            return DownloadResponse(
                video_id=cached_video_id,
                title=cached_entry.title,
                caption_segments=segments,
            )

    # Slow path: use yt-dlp only when local cache is missing.
    try:
        video_id, title = _download_service.get_video_info(body.url)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not fetch video metadata from YouTube, and no complete "
                "local cache was found for this URL. If this is the demo video, "
                "make sure the local MP4 and caption TXT exist."
            ),
        ) from exc

    # Use title from registry; fall back to yt-dlp title with colons stripped.
    entry = get_video(video_id)
    stem = entry.title if entry else title.replace(":", "")

    video_path = videos_dir / f"{stem}.mp4"
    caption_path = captions_dir / f"{stem}.txt"

    if not video_path.exists():
        _download_service.download_video(body.url, str(videos_dir), stem)

    if not caption_path.exists():
        _download_service.download_caption(body.url, str(captions_dir), stem)

    segments = _download_service.read_caption_segments(caption_path)

    return DownloadResponse(
        video_id=video_id,
        title=stem,
        caption_segments=segments,
    )
