"""
Step 1 of the pipeline: Ingest.

Transcript fetching strategy (in order):
  1. Supadata API  — cloud-friendly, bypasses YouTube IP blocks (set SUPADATA_API_KEY)
  2. youtube-transcript-api with proxy  — fallback if proxy creds are set
  3. youtube-transcript-api direct  — local development only

Set SUPADATA_API_KEY in Render environment variables for cloud deployment.
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


# ---------------------------------------------------------------------------
# Strategy 1 — Supadata (cloud-safe, bypasses YouTube IP blocks)
# ---------------------------------------------------------------------------

def _fetch_via_supadata(video_id: str) -> List[TranscriptSnippet]:
    """
    Use Supadata's YouTube transcript API.
    Handles YouTube IP blocks transparently — works on any cloud provider.
    Free tier: 200 requests/month at supadata.ai
    """
    api_key = os.environ.get("SUPADATA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("SUPADATA_API_KEY not set.")

    resp = requests.get(
        "https://api.supadata.ai/v1/youtube/transcript",
        params={"videoId": video_id, "text": "false"},
        headers={"x-api-key": api_key},
        timeout=30,
    )

    if resp.status_code == 404:
        raise TranscriptError(f"Supadata: no transcript found for '{video_id}'.")
    if resp.status_code == 401:
        raise TranscriptError("Supadata: invalid API key. Check SUPADATA_API_KEY in Render.")
    resp.raise_for_status()

    data = resp.json()
    content = data.get("content", [])

    if not content:
        raise TranscriptError(f"Supadata returned an empty transcript for '{video_id}'.")

    return [
        TranscriptSnippet(
            text=item["text"],
            start=item["offset"] / 1000,    # ms → seconds
            duration=item["duration"] / 1000,
        )
        for item in content
    ]


# ---------------------------------------------------------------------------
# Strategy 2 & 3 — youtube-transcript-api (with or without proxy)
# ---------------------------------------------------------------------------

def _build_api() -> YouTubeTranscriptApi:
    username = os.environ.get("PROXY_USERNAME", "").strip()
    password = os.environ.get("PROXY_PASSWORD", "").strip()
    if username and password:
        proxy_url = f"http://{username}:{password}@p.webshare.io:80"
        return YouTubeTranscriptApi(proxies={"http": proxy_url, "https": proxy_url})
    return YouTubeTranscriptApi()


def _fetch_via_yta(video_id: str, languages: Sequence[str]) -> List[TranscriptSnippet]:
    api = _build_api()
    try:
        fetched = api.fetch(video_id, languages=list(languages))
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as primary_error:
        try:
            available = api.list(video_id)
            transcript = next(iter(available))
        except (StopIteration, CouldNotRetrieveTranscript) as fallback_error:
            raise TranscriptError(
                f"No transcript available for '{video_id}'. ({primary_error})"
            ) from fallback_error
        fetched = transcript.fetch()
    except CouldNotRetrieveTranscript as e:
        raise TranscriptError(f"Could not retrieve transcript for '{video_id}': {e}") from e

    snippets = [
        TranscriptSnippet(text=s.text, start=s.start, duration=s.duration)
        for s in fetched
    ]
    if not snippets:
        raise TranscriptError(f"Transcript for '{video_id}' came back empty.")
    return snippets


# ---------------------------------------------------------------------------
# Public entry point — tries all strategies in order
# ---------------------------------------------------------------------------

def fetch_transcript(video_id: str, languages: Sequence[str] = ("en",)) -> List[TranscriptSnippet]:
    """
    Fetch transcript using the best available method.
    Priority: Supadata → youtube-transcript-api (proxy) → youtube-transcript-api (direct)
    """
    # Strategy 1: Supadata (recommended for cloud)
    if os.environ.get("SUPADATA_API_KEY"):
        try:
            return _fetch_via_supadata(video_id)
        except TranscriptError:
            raise   # Surface Supadata-specific errors (no transcript, bad key)
        except Exception:
            pass    # Network issue — fall through to next strategy

    # Strategy 2 & 3: youtube-transcript-api
    return _fetch_via_yta(video_id, languages)


# ---------------------------------------------------------------------------
# Video title
# ---------------------------------------------------------------------------

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
