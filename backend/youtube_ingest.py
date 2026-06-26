"""
Step 1 of the pipeline: Ingest.

Transcript fetching strategy:
  - Cloud (SUPADATA_API_KEY set): uses Supadata API — bypasses YouTube IP blocks
  - Local (no key): uses youtube-transcript-api directly
"""
import os
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
    pass


def extract_video_id(url_or_id: str) -> str:
    candidate = url_or_id.strip()
    if _BARE_ID_RE.match(candidate):
        return candidate
    match = _VIDEO_ID_RE.search(candidate)
    if match:
        return match.group(1)
    raise TranscriptError(
        f"Could not find a YouTube video ID in: {url_or_id!r}."
    )


@dataclass
class TranscriptSnippet:
    text: str
    start: float
    duration: float


def _fetch_via_supadata(video_id: str) -> List[TranscriptSnippet]:
    """Use Supadata API — works on any cloud provider, bypasses YouTube blocks."""
    api_key = os.environ.get("SUPADATA_API_KEY", "").strip()

    resp = requests.get(
        "https://api.supadata.ai/v1/youtube/transcript",
        params={"videoId": video_id, "text": "false"},
        headers={"x-api-key": api_key},
        timeout=30,
    )

    if resp.status_code == 401:
        raise TranscriptError("Supadata: invalid API key. Check SUPADATA_API_KEY in Render.")
    if resp.status_code == 404:
        raise TranscriptError(f"No transcript found for '{video_id}' via Supadata.")
    resp.raise_for_status()

    content = resp.json().get("content", [])
    if not content:
        raise TranscriptError(f"Supadata returned empty transcript for '{video_id}'.")

    return [
        TranscriptSnippet(
            text=item["text"],
            start=item["offset"] / 1000,
            duration=item["duration"] / 1000,
        )
        for item in content
    ]


def _fetch_via_yta(video_id: str, languages: Sequence[str]) -> List[TranscriptSnippet]:
    """Use youtube-transcript-api directly — for local development only."""
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=list(languages))
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
        try:
            available = api.list(video_id)
            transcript = next(iter(available))
            fetched = transcript.fetch()
        except (StopIteration, CouldNotRetrieveTranscript):
            raise TranscriptError(f"No transcript available for '{video_id}'. ({e})")
    except CouldNotRetrieveTranscript as e:
        raise TranscriptError(f"Could not retrieve transcript for '{video_id}': {e}")

    snippets = [
        TranscriptSnippet(text=s.text, start=s.start, duration=s.duration)
        for s in fetched
    ]
    if not snippets:
        raise TranscriptError(f"Transcript for '{video_id}' came back empty.")
    return snippets


def fetch_transcript(video_id: str, languages: Sequence[str] = ("en",)) -> List[TranscriptSnippet]:
    if os.environ.get("SUPADATA_API_KEY"):
        return _fetch_via_supadata(video_id)
    return _fetch_via_yta(video_id, languages)


def get_video_title(video_id: str) -> str:
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
