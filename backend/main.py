"""
The "3-Hour Podcast" Interrogator
---------------------------------
A command-line RAG tool for chatting with long-form YouTube videos.

Usage:
    python main.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python main.py VIDEO_ID --ask "What did they say about X?"
    python main.py --list
    python main.py "<url>" --reindex
"""

import argparse
import sys

from config import SETTINGS
from chunker import chunk_transcript, format_timestamp
from embedder import EmbeddingError, GeminiEmbedder
from qa_engine import QAEngine
from vector_store import VectorStore
from youtube_ingest import TranscriptError, extract_video_id, fetch_transcript, get_video_title

BANNER = """\
┌─────────────────────────────────────────────┐
│   The "3-Hour Podcast" Interrogator (RAG)    │
└─────────────────────────────────────────────┘
"""


def build_components():
    if not SETTINGS.gemini_api_key:
        print(
            "ERROR: GEMINI_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key, or export it:\n"
            "  export GEMINI_API_KEY=your-key-here",
            file=sys.stderr,
        )
        sys.exit(1)

    embedder = GeminiEmbedder(
        api_key=SETTINGS.gemini_api_key,
        model=SETTINGS.embedding_model,
        output_dimensionality=SETTINGS.embedding_dimensions,
    )
    store = VectorStore(db_path=SETTINGS.db_path)
    qa = QAEngine(
        embedder=embedder,
        store=store,
        api_key=SETTINGS.gemini_api_key,
        generation_model=SETTINGS.generation_model,
        top_k=SETTINGS.top_k,
    )
    return embedder, store, qa


def ingest_video(url_or_id: str, embedder: GeminiEmbedder, store: VectorStore, force: bool = False) -> str:
    video_id = extract_video_id(url_or_id)

    if store.has_video(video_id) and not force:
        print(f"'{video_id}' is already indexed — jumping straight to chat. (Use --reindex to refresh it.)")
        return video_id

    print(f"[1/4] Fetching transcript for {video_id} ...")
    snippets = fetch_transcript(video_id, languages=SETTINGS.transcript_languages)
    total_seconds = snippets[-1].start + snippets[-1].duration
    print(f"      Got {len(snippets)} caption snippets covering {format_timestamp(total_seconds)}.")

    title = get_video_title(video_id)

    print("[2/4] Chunking transcript into overlapping segments ...")
    chunks = chunk_transcript(
        snippets,
        chunk_size_words=SETTINGS.chunk_size_words,
        overlap_words=SETTINGS.chunk_overlap_words,
    )
    print(f"      Created {len(chunks)} chunks (~{SETTINGS.chunk_size_words} words each, "
          f"{SETTINGS.chunk_overlap_words}-word overlap).")

    print(f"[3/4] Embedding chunks with {SETTINGS.embedding_model} ...")
    try:
        embeddings = embedder.embed_documents([c.text for c in chunks])
    except EmbeddingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("[4/4] Storing vectors in ChromaDB ...")
    if store.has_video(video_id):
        store.delete_video(video_id)
    store.add_chunks(video_id, chunks, embeddings, video_title=title)

    print(f"\nIndexed: \"{title}\"\n")
    return video_id


def print_answer(video_id: str, answer) -> None:
    print(f"\nAssistant: {answer.text}\n")
    if answer.citations:
        seen = set()
        print("Sources:")
        for c in answer.citations:
            if c.timestamp in seen:
                continue
            seen.add(c.timestamp)
            print(f"  [{c.timestamp}]  https://youtu.be/{video_id}?t={int(c.start_time)}")
        print()


def chat_loop(video_id: str, qa: QAEngine) -> None:
    print("Ask anything about this video. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        try:
            answer = qa.ask(video_id, question)
        except EmbeddingError as e:
            print(f"Error embedding your question: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error generating an answer: {e}", file=sys.stderr)
            continue

        print_answer(video_id, answer)


def main():
    parser = argparse.ArgumentParser(description='Chat with a long YouTube video/podcast via RAG.')
    parser.add_argument("url", nargs="?", help="YouTube URL or video ID to ingest and chat with")
    parser.add_argument("--reindex", action="store_true", help="Re-fetch and re-embed even if already indexed")
    parser.add_argument("--list", action="store_true", help="List videos already indexed locally")
    parser.add_argument("--ask", metavar="QUESTION", help="Ask one question non-interactively and exit")
    args = parser.parse_args()

    _, store, qa = build_components()

    if args.list:
        videos = store.list_videos()
        if not videos:
            print("No videos indexed yet.")
        else:
            print("Indexed videos:")
            for v in videos:
                print(f"  - {v}  (https://youtu.be/{v})")
        return

    if not args.url:
        print(BANNER)
        args.url = input("Paste a YouTube URL or video ID to begin: ").strip()

    embedder = qa.embedder
    try:
        video_id = ingest_video(args.url, embedder, store, force=args.reindex)
    except TranscriptError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.ask:
        answer = qa.ask(video_id, args.ask)
        print_answer(video_id, answer)
        return

    chat_loop(video_id, qa)


if __name__ == "__main__":
    main()
