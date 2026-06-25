"""
Step 2 of the pipeline: Chunk.

Splits a transcript into overlapping text segments, each tagged with the
timestamp it starts at, so retrieved chunks can be cited and linked back
to the exact moment in the video.
"""

from dataclasses import dataclass
from typing import List, Sequence

from youtube_ingest import TranscriptSnippet


@dataclass
class Chunk:
    chunk_id: int
    text: str
    start_time: float  # seconds into the video
    end_time: float  # seconds into the video


def format_timestamp(seconds: float) -> str:
    """12.0 -> '00:12', 3725.0 -> '1:02:05'."""
    total = int(max(seconds, 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def chunk_transcript(
    snippets: Sequence[TranscriptSnippet],
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> List[Chunk]:
    """Build overlapping, timestamped chunks from raw transcript snippets.

    Auto-generated captions arrive as many short snippets (often just a
    few words each), which is too granular to embed well and too narrow
    for an LLM to reason over. We flatten every word into a single
    timeline (each word remembers which snippet/timestamp it came from),
    then slide a window over that timeline.

    Overlap means a sentence that gets cut off at a chunk boundary still
    appears in full inside the *next* chunk too, so retrieval doesn't miss
    answers that straddle a boundary.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")

    timeline: List[tuple] = []  # (word, start_time)
    for snippet in snippets:
        for word in snippet.text.split():
            timeline.append((word, snippet.start))

    if not timeline:
        return []

    step = chunk_size_words - overlap_words
    chunks: List[Chunk] = []
    i = 0
    chunk_id = 0
    n = len(timeline)

    while i < n:
        window = timeline[i : i + chunk_size_words]
        text = " ".join(word for word, _ in window)
        start_time = window[0][1]
        end_time = window[-1][1]
        chunks.append(Chunk(chunk_id=chunk_id, text=text, start_time=start_time, end_time=end_time))

        chunk_id += 1
        if i + chunk_size_words >= n:
            break
        i += step

    return chunks
