"""
Central configuration for the Podcast Interrogator.

Everything here can be overridden with environment variables (or a .env
file — see .env.example) so you never have to edit code to tweak behavior.
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; if it's missing we just rely on real
    # environment variables (e.g. exported in the shell or set by the OS).
    pass


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- Gemini API ---
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))

    # Embedding model. text-embedding-004 (named in the original spec) was
    # deprecated by Google on 2026-01-14. gemini-embedding-001 is its
    # direct, GA replacement and is what we use by default.
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
    )
    # Gemini embeddings use Matryoshka representation learning, so they can be
    # safely truncated. 768 is Google's recommended sweet spot of quality vs.
    # storage cost. Use 1536 or 3072 for max fidelity on very nuanced content.
    embedding_dimensions: int = field(default_factory=lambda: _env_int("EMBEDDING_DIMENSIONS", 768))

    # Generation model. gemini-2.0-flash (named in the original spec) was
    # shut down by Google on 2026-06-01. gemini-2.5-flash is the
    # closest current equivalent (cheap, fast, same general tier).
    generation_model: str = field(
        default_factory=lambda: os.environ.get("GENERATION_MODEL", "gemini-2.5-flash")
    )

    # --- Chunking ---
    # Target words per chunk and how many words of overlap between
    # consecutive chunks, so an answer near a chunk boundary doesn't lose
    # context.
    chunk_size_words: int = field(default_factory=lambda: _env_int("CHUNK_SIZE_WORDS", 220))
    chunk_overlap_words: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_WORDS", 40))

    # --- Retrieval ---
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))

    # --- Storage ---
    db_path: str = field(default_factory=lambda: os.environ.get("CHROMA_DB_PATH", "./chroma_db"))

    # --- Transcript ---
    transcript_languages: tuple = ("en",)


SETTINGS = Settings()
