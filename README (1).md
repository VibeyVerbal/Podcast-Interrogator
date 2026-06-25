# 🎙️ Podcast Interrogator

> Ask questions to any podcast episode and get AI-powered answers with timestamps.
> Built by [Your Club Name] — [College/University Name]

---

## What it does

Paste a podcast URL (or upload an audio file), and the app transcribes it and lets you have a conversation with the episode — ask anything and get answers grounded in what was actually said, with source timestamps.

---

## Project structure

```
podcast-interrogator/
├── backend/                  # Core logic (interrogator engine)
│   ├── main.py               # Entry point / FastAPI app
│   ├── interrogator.py       # Q&A engine (LangChain / your logic)
│   ├── transcriber.py        # Audio → text (Whisper)
│   └── requirements.txt
│
├── frontend/                 # UI layer
│   ├── app.py                # Streamlit / Chainlit interface
│   └── requirements.txt
│
├── .env.example              # Template for environment variables
├── .gitignore
└── README.md
```

---

## Team & ownership

| Area | Owner | Responsibility |
|------|-------|----------------|
| Backend / API | [Name] | Wrapping interrogator as REST API, session management |
| Frontend / UI | [Name] | Streamlit/Chainlit interface, chat components |
| Deployment | [Name] | Streamlit Cloud setup, environment secrets, CI |
| Docs / Demo | [Name] | README, demo video, club presentation |

---

## Getting started (local)

### 1. Clone the repo
```bash
git clone https://github.com/your-org/podcast-interrogator.git
cd podcast-interrogator
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Install dependencies
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
pip install -r requirements.txt
```

### 4. Run locally
```bash
# Backend (in one terminal)
cd backend
uvicorn main:app --reload --port 8000

# Frontend (in another terminal)
cd frontend
streamlit run app.py
```

---

## Environment variables

See `.env.example` for all required keys. Never commit `.env` to git.

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | GPT / Whisper access |
| `ANTHROPIC_API_KEY` | Claude access (if used) |
| `HUGGINGFACE_TOKEN` | HuggingFace models (optional) |

---

## Deployment

The app is deployed on **Streamlit Community Cloud**.

Live link: `https://your-app-name.streamlit.app` *(update after deploy)*

To deploy your own instance:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets from `.env.example` in the Streamlit dashboard

---

## Tech stack

- **Transcription** — OpenAI Whisper
- **Q&A engine** — LangChain + GPT-4 / Claude
- **UI** — Streamlit
- **Deployment** — Streamlit Community Cloud

---

## Contributing

1. Create a branch: `git checkout -b feature/your-feature`
2. Commit your changes: `git commit -m "add: your feature"`
3. Push and open a PR on GitHub

---

*Made with ❤️ at [Club Name]*
