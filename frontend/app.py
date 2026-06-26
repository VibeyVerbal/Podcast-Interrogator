"""
Core Interrogator — Streamlit UI
----------------------------------
Apple-inspired, minimalist interface.

Run with:
    streamlit run app.py
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

SUGGESTED_QUESTIONS = [
    "What are the central arguments made?",
    "What contradictions exist in the conversation?",
    "What assumptions go unchallenged?",
    "What are the most actionable insights?",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Core Interrogator",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Apple-inspired CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    letter-spacing: -0.01em;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
    max-width: 860px;
}
.pipeline-tag {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888;
    border: 0.5px solid #444;
    padding: 2px 7px;
    border-radius: 4px;
    margin-bottom: 6px;
}
.welcome-headline {
    font-size: 2.6rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.welcome-sub {
    font-size: 1rem;
    color: #888;
    font-weight: 400;
    letter-spacing: -0.01em;
    margin-bottom: 2rem;
}
.citation-row {
    font-family: "SF Mono", "Fira Code", "Consolas", monospace;
    font-size: 0.72rem;
    color: #aaa;
    padding: 6px 0;
    border-bottom: 0.5px solid rgba(128,128,128,0.12);
}
.stat-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #666;
    font-weight: 600;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 500;
    letter-spacing: -0.03em;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "video_title" not in st.session_state:
    st.session_state.video_title = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path):
    try:
        return requests.get(f"{API_URL}{path}", timeout=10)
    except requests.ConnectionError:
        return None


def api_post(path, payload):
    try:
        return requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    except requests.ConnectionError:
        return None


def ask_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    r = api_post("/ask", {"video_id": st.session_state.video_id, "question": question})

    if r is None:
        st.session_state.messages.append({
            "role": "assistant", "content": "Connection lost.", "citations": []
        })
        return

    if r.status_code == 200:
        data = r.json()
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "citations": data["citations"],
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": r.json().get("detail", "Error."),
            "citations": [],
        })

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⬡ Core Interrogator")
    st.caption("Long-form podcast intelligence.")
    st.divider()

    # Load new podcast
    st.markdown('<div class="pipeline-tag">Input</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        load_btn = st.button("Analyze", type="primary", use_container_width=True)
    with col2:
        force_cb = st.checkbox("↺", help="Force re-index")

    # Pipeline display on load
    if load_btn and url_input:
        st.divider()
        st.markdown('<div class="pipeline-tag">Processing</div>', unsafe_allow_html=True)

        pipeline_a = st.status("Pipeline A — Transcript", expanded=True)
        with pipeline_a:
            st.caption("Fetching YouTube transcript...")
            st.caption("Chunking into segments...")

        pipeline_b = st.status("Pipeline B — Embeddings", expanded=True)
        with pipeline_b:
            st.caption("Awaiting transcript completion...")
            st.caption("Generating 768-dim vectors...")

        r = api_post("/ingest", {"url": url_input, "force": force_cb})

        if r is None:
            pipeline_a.update(label="Pipeline A — Failed", state="error")
            pipeline_b.update(label="Pipeline B — Failed", state="error")
            st.error("Backend unreachable.")
        elif r.status_code == 200:
            data = r.json()
            pipeline_a.update(label="Pipeline A — Complete", state="complete", expanded=False)
            pipeline_b.update(label="Pipeline B — Complete", state="complete", expanded=False)
            st.session_state.video_id = data["video_id"]
            st.session_state.video_title = data["title"]
            st.session_state.messages = []
            st.rerun()
        else:
            detail = r.json().get("detail", "Unknown error.")
            pipeline_a.update(label="Pipeline A — Error", state="error")
            pipeline_b.update(label="Pipeline B — Error", state="error")
            st.error(detail)

    st.divider()

    # Indexed podcasts
    r = api_get("/videos")
    if r and r.status_code == 200:
        videos = r.json().get("videos", [])
        if videos:
            st.markdown('<div class="pipeline-tag">Indexed</div>', unsafe_allow_html=True)
            for vid in videos:
                if st.button(f"▸ {vid[:28]}…" if len(vid) > 28 else f"▸ {vid}",
                             key=f"switch_{vid}", use_container_width=True):
                    st.session_state.video_id = vid
                    st.session_state.video_title = vid
                    st.session_state.messages = []
                    st.rerun()

    st.divider()

    if st.session_state.video_id:
        if st.button("Clear session", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Status
    health = api_get("/health")
    st.caption("🟢 Engine online" if health and health.status_code == 200 else "🔴 Engine offline")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not st.session_state.video_id:

    # Welcome screen
    st.markdown('<div class="welcome-headline">Core Interrogator.</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-sub">Long-form podcast analysis. Timestamped. Precise. Zero filler.</div>', unsafe_allow_html=True)

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-label">Pipeline A</div><div class="stat-value">Transcript</div>', unsafe_allow_html=True)
        st.caption("High-fidelity text extraction from YouTube captions.")
    with col2:
        st.markdown('<div class="stat-label">Pipeline B</div><div class="stat-value">Embeddings</div>', unsafe_allow_html=True)
        st.caption("768-dimensional semantic vector indexing via Gemini.")
    with col3:
        st.markdown('<div class="stat-label">Engine</div><div class="stat-value">Gemini 2.5</div>', unsafe_allow_html=True)
        st.caption("RAG retrieval with timestamp-cited generation.")

    st.divider()
    st.markdown("**Sample queries**")
    for q in SUGGESTED_QUESTIONS:
        st.markdown(f"— *{q}*")

else:

    # Chat screen
    st.markdown(f"#### {st.session_state.video_title or st.session_state.video_id}")
    st.caption(f"youtube.com/watch?v={st.session_state.video_id}")
    st.divider()

    # Suggested questions when chat is empty
    if not st.session_state.messages:
        st.markdown('<div class="pipeline-tag">Suggested queries</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()
        st.divider()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("citations"):
                seen = set()
                lines = []
                for c in msg["citations"]:
                    if c["timestamp"] in seen:
                        continue
                    seen.add(c["timestamp"])
                    lines.append(f'<div class="citation-row"><a href="{c["url"]}" target="_blank">[{c["timestamp"]}]</a> — {c["url"]}</div>')
                if lines:
                    with st.expander(f"Sources  ·  {len(lines)} timestamp(s)"):
                        st.markdown("".join(lines), unsafe_allow_html=True)

    # Handle pending question from suggestion buttons
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        with st.spinner("Interrogating transcript..."):
            ask_question(question)
        st.rerun()

    # Chat input
    if question := st.chat_input("Query the transcript..."):
        with st.spinner("Interrogating transcript..."):
            ask_question(question)
        st.rerun()
