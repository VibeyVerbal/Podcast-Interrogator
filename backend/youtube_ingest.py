"""
Step 1 of the pipeline: Ingest.
Takes a YouTube URL (or bare video ID) and returns the auto-generated
transcript as a list of timed snippets, using youtube-transcript-api.
"""
import re
from dataclasses import dataclass
from typing import List, Sequence

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

_VIDEO_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)
_BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


class TranscriptError(Exception):
    """Raised whenever we can't get a usable transcript for a video."""


def extract_video_id(url_or_id: str) -> str:
    """Accepts a full YouTube URL or bare 11-character video ID."""
    candidate = url_or_id.strip()
    if _BARE_ID_RE.match(candidate):
        return candidate
    match = _VIDEO_ID_RE.search(candidate)
    if match:
        return match.group(1)
    raise TranscriptError(
        f"Could not find a YouTube video ID in: {url_or_id!r}. "
        "Pass a full YouTube URL or an 11-character video ID."
    )


@dataclass
class TranscriptSnippet:
    text: str
    start: float
    duration: float


def fetch_transcript(video_id: str, languages: Sequence[str] = ("en",)) -> List[TranscriptSnippet]:
    """Fetch the transcript for a video.
    Tries the requested languages first (auto-generated captions included).
    If none of those are available, falls back to whatever transcript
    YouTube does offer for the video, so non-English podcasts still work.
    """
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=list(languages))
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as primary_error:
        try:
            available = api.list(video_id)
            transcript = next(iter(available))
        except (StopIteration, CouldNotRetrieveTranscript) as fallback_error:
            raise TranscriptError(
                f"No transcript is available for video '{video_id}'. "
                f"({primary_error})"
            ) from fallback_error
        fetched = transcript.fetch()
    except CouldNotRetrieveTranscript as e:
        raise TranscriptError(f"Could not retrieve a transcript for '{video_id}': {e}") from e

    snippets = [
        TranscriptSnippet(text=s.text, start=s.start, duration=s.duration) for s in fetched
    ]
    if not snippets:
        raise TranscriptError(f"Transcript for '{video_id}' came back empty.")
    return snippets


def get_video_title(video_id: str) -> str:
    """Best-effort title via YouTube oEmbed. Falls back to video ID."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("title", video_id)
    except Exception:
        return video_id
