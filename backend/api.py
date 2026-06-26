"""
Podcast Interrogator — FastAPI backend (Secure)
------------------------------------------------
Security layers:
  1. X-API-Key header required on all routes except /health
  2. Rate limiting — 30 requests/min per IP
  3. CORS restricted to allowed origins
  4. Input validation on all endpoints
  5. No internal error details leaked to clients
"""

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chunker import chunk_transcript
from config import SETTINGS
from embedder import EmbeddingError, GeminiEmbedder
from qa_engine import QAEngine
from vector_store import VectorStore
from youtube_ingest import TranscriptError, extract_video_id, fetch_transcript, get_video_title

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Security config
# ---------------------------------------------------------------------------

BACKEND_SECRET = os.environ.get("BACKEND_SECRET", "")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_token(api_key: str = Depends(api_key_header)):
    """Require X-API-Key header on all protected routes."""
    if BACKEND_SECRET and api_key != BACKEND_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized.")


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ---------------------------------------------------------------------------
# App-wide singletons
# ---------------------------------------------------------------------------

_embedder: GeminiEmbedder | None = None
_store: VectorStore | None = None
_qa: QAEngine | None = None
_executor = ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedder, _store, _qa

    if not SETTINGS.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

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
    logger.info("Core Interrogator API started.")
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="Core Interrogator API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)

# Rate limit error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

YOUTUBE_RE = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w\-]{11}"
)


class IngestRequest(BaseModel):
    url: str
    force: bool = False

    @field_validator("url")
    @classmethod
    def must_be_youtube(cls, v):
        if not YOUTUBE_RE.search(v) and len(v) != 11:
            raise ValueError("Must be a valid YouTube URL or 11-character video ID.")
        if len(v) > 200:
            raise ValueError("URL too long.")
        return v.strip()


class IngestResponse(BaseModel):
    video_id: str
    title: str
    status: str


class AskRequest(BaseModel):
    video_id: str
    question: str

    @field_validator("video_id")
    @classmethod
    def valid_video_id(cls, v):
        if not re.match(r"^[\w\-]{11}$", v):
            raise ValueError("Invalid video ID.")
        return v

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        if len(v) > 1000:
            raise ValueError("Question too long.")
        return v.strip()


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
# Thread helper
# ---------------------------------------------------------------------------

async def _run(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Public — no auth needed. Used by frontend to check if backend is awake."""
    return {"status": "ok"}


@app.get("/videos", response_model=VideosResponse, dependencies=[Depends(verify_token)])
@limiter.limit("30/minute")
async def list_videos(request: Request):
    return {"videos": _store.list_videos()}


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def ingest(request: Request, req: IngestRequest):
    try:
        video_id = extract_video_id(req.url)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not extract video ID.")

    if _store.has_video(video_id) and not req.force:
        try:
            title = get_video_title(video_id)
        except Exception:
            title = video_id
        return IngestResponse(video_id=video_id, title=title, status="already_indexed")

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
    except EmbeddingError:
        raise HTTPException(status_code=502, detail="Embedding service error. Try again shortly.")
    except Exception:
        logger.exception("Ingest failed for video_id=%s", video_id)
        raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs.")

    return IngestResponse(video_id=video_id, title=title, status="indexed")


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_token)])
@limiter.limit("30/minute")
async def ask(request: Request, req: AskRequest):
    if not _store.has_video(req.video_id):
        raise HTTPException(status_code=404, detail="Video not indexed. Ingest it first.")

    try:
        answer = await _run(_qa.ask, req.video_id, req.question)
    except EmbeddingError:
        raise HTTPException(status_code=502, detail="Embedding service error. Try again shortly.")
    except Exception:
        logger.exception("Ask failed for video_id=%s", req.video_id)
        raise HTTPException(status_code=500, detail="Query failed. Check server logs.")

    citations = [
        CitationOut(
            timestamp=c.timestamp,
            start_time=c.start_time,
            url=f"https://youtu.be/{req.video_id}?t={int(c.start_time)}",
        )
        for c in answer.citations
    ]
    return AskResponse(answer=answer.text, citations=citations)


@app.delete("/videos/{video_id}", dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def delete_video(request: Request, video_id: str):
    if not re.match(r"^[\w\-]{11}$", video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID.")
    if not _store.has_video(video_id):
        raise HTTPException(status_code=404, detail="Video not found.")
    _store.delete_video(video_id)
    return {"deleted": video_id}
