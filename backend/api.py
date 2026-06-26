"""
Podcast Interrogator — FastAPI wrapper
--------------------------------------
Wraps the existing RAG pipeline as a REST API so the Streamlit
frontend can call it over HTTP.

Run with:
    uvicorn api:app --reload --port 8000
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chunker import chunk_transcript
from config import SETTINGS
from embedder import EmbeddingError, GeminiEmbedder
from qa_engine import QAEngine
from vector_store import VectorStore
from youtube_ingest import TranscriptError, extract_video_id, fetch_transcript, get_video_title


# ---------------------------------------------------------------------------
# Singletons — built once at startup, reused across all requests
# ---------------------------------------------------------------------------

_embedder: GeminiEmbedder | None = None
_store: VectorStore | None = None
_qa: QAEngine | None = None
_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store, _qa

    if not SETTINGS.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    _embedder = GeminiEmbedder(
        api_key=SETTINGS.gemini_api_key,
        model=SETTINGS.embedding_model,
        output_dimensionality=SETTINGS.embedding_dimensions,
    )
    _store = VectorStore(db_path=SETTINGS.db_path)
    _qa = QAEngine(
        embedder=_embedder,
        store=_store,
        api_key=SETTINGS.gemini_api_key,
        generation_model=SETTINGS.generation_model,
        top_k=SETTINGS.top_k,
    )
    yield
    _executor.shutdown(wait=False)


app = FastAPI(title="Podcast Interrogator API", version="1.0.0", lifespan=lifespan)

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    url: str
    force: bool = False  # set True to re-index an already-indexed video


class IngestResponse(BaseModel):
    video_id: str
    title: str
    status: str  # "indexed" | "already_indexed"


class AskRequest(BaseModel):
    video_id: str
    question: str


class CitationOut(BaseModel):
    timestamp: str
    start_time: float
    url: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]


class VideosResponse(BaseModel):
    videos: list[str]


# ---------------------------------------------------------------------------
# Helper: run blocking (sync) code without freezing the API
# ---------------------------------------------------------------------------

async def _run(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Quick check that the API is alive."""
    return {"status": "ok"}


@app.get("/videos", response_model=VideosResponse)
async def list_videos():
    """Return all video IDs that have been indexed."""
    return {"videos": _store.list_videos()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    Ingest a YouTube podcast URL.
    Fetches transcript → chunks → embeds → stores in ChromaDB.
    If the video is already indexed, skips processing unless force=True.
    """
    try:
        video_id = extract_video_id(req.url)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or video ID.")

    # Already indexed — skip the slow pipeline
    if _store.has_video(video_id) and not req.force:
        try:
            title = get_video_title(video_id)
        except Exception:
            title = video_id
        return IngestResponse(video_id=video_id, title=title, status="already_indexed")

    # Run the slow pipeline in a thread so the API stays responsive
    def _do_ingest():
        snippets = fetch_transcript(video_id, languages=SETTINGS.transcript_languages)
        title = get_video_title(video_id)
        chunks = chunk_transcript(
            snippets,
            chunk_size_words=SETTINGS.chunk_size_words,
            overlap_words=SETTINGS.chunk_overlap_words,
        )
        embeddings = _embedder.embed_documents([c.text for c in chunks])
        if _store.has_video(video_id):
            _store.delete_video(video_id)
        _store.add_chunks(video_id, chunks, embeddings, video_title=title)
        return title

    try:
        title = await _run(_do_ingest)
    except TranscriptError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(video_id=video_id, title=title, status="indexed")


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Ask a question about an already-indexed video.
    Returns the answer text and timestamped citations.
    """
    if not _store.has_video(req.video_id):
        raise HTTPException(
            status_code=404,
            detail="Video not indexed yet. Call POST /ingest first."
        )

    try:
        answer = await _run(_qa.ask, req.video_id, req.question)
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=f"Embedding error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    citations = [
        CitationOut(
            timestamp=c.timestamp,
            start_time=c.start_time,
            url=f"https://youtu.be/{req.video_id}?t={int(c.start_time)}",
        )
        for c in answer.citations
    ]

    return AskResponse(answer=answer.text, citations=citations)


@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Remove a video and its embeddings from the index."""
    if not _store.has_video(video_id):
        raise HTTPException(status_code=404, detail="Video not found.")
    _store.delete_video(video_id)
    return {"deleted": video_id}
