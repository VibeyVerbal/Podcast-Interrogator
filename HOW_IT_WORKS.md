# Core Interrogator — How It Works
### Built by [Your Club Name] · [College/University]

---

## The Problem

Podcasts are 2–3 hours long.  
Nobody has time to scrub through them.  
Timestamps help, but searching by topic is still manual.

**Core Interrogator solves this.**  
Ask any question. Get a precise answer with a clickable timestamp.

---

## Architecture

```
User pastes YouTube URL
        │
        ▼
┌─────────────────────────────────────────────┐
│              PIPELINE A                     │
│         High-Fidelity Transcript            │
│  YouTube Captions → Chunked Text Segments   │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│              PIPELINE B                     │
│        768-Dim Semantic Embeddings          │
│   Gemini Embedding Model → ChromaDB Store   │
└─────────────────────────────────────────────┘
        │
        ▼
     User asks a question
        │
        ▼
  Question embedded → Top-K chunks retrieved
        │
        ▼
  Gemini 2.5 Flash generates answer
  (grounded strictly in retrieved chunks)
        │
        ▼
  Answer + timestamped citations → UI
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Transcription | YouTube Transcript API |
| Chunking | Custom overlapping word-window chunker |
| Embeddings | Gemini Embedding Model (`gemini-embedding-001`, 768-dim) |
| Vector Store | ChromaDB (local persistent store) |
| Generation | Gemini 2.5 Flash (RAG-grounded, temp 0.2) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Deployment | Render (API) + Streamlit Community Cloud (UI) |

---

## RAG Pipeline Explained

**RAG = Retrieval-Augmented Generation**

Instead of asking the AI to "remember" the podcast (impossible for 3-hour audio), we:

1. **Store** the transcript as vector embeddings in ChromaDB
2. **Retrieve** the most semantically relevant chunks when a question is asked
3. **Generate** an answer grounded only in those chunks — with no hallucination

The AI is explicitly instructed: *"Use only the transcript excerpts provided. Never speculate."*

---

## Key Design Decisions

**Why YouTube captions over Whisper transcription?**  
Faster and free. Whisper is more accurate but requires GPU and minutes of processing. For most podcasts, YouTube's auto-captions are sufficient.

**Why 768 dimensions?**  
Google's recommended sweet spot of quality vs. storage cost for `gemini-embedding-001`. Matryoshka embeddings allow truncation without quality loss.

**Why ChromaDB?**  
Lightweight, file-based, zero infrastructure. Perfect for a demo that runs on a free tier.

**Why temperature 0.2?**  
Low temperature = deterministic, citation-faithful answers. We're interrogating a source, not brainstorming.

---

## Live Demo

**App:** `https://your-app.streamlit.app`  
**API:** `https://podcast-interrogator.onrender.com/docs`  
**Repo:** `https://github.com/VibeyVerbal/Podcast-Interrogator`

---

## What's Next

- [ ] Whisper fallback for videos without captions
- [ ] Multi-podcast cross-referencing ("What do Lex and Huberman both say about sleep?")
- [ ] Speaker diarization (who said what)
- [ ] Shareable Q&A links
- [ ] Chrome extension for any YouTube video

---

*Built with Python · FastAPI · Streamlit · Google Gemini · ChromaDB*
