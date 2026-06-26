"""
Core Interrogator — Streamlit UI (Secure + Stable)
"""

import datetime
import os
import time

import requests
import streamlit as st

# Read secrets safely
API_URL    = st.secrets.get("API_URL",        os.getenv("API_URL",        "http://localhost:8000"))
API_SECRET = st.secrets.get("BACKEND_SECRET", os.getenv("BACKEND_SECRET", ""))

SUGGESTED_QUESTIONS = [
    "What are the central arguments made?",
    "What contradictions exist in the conversation?",
    "What assumptions go unchallenged?",
    "What are the most actionable insights?",
]

HEADERS = {"X-API-Key": API_SECRET} if API_SECRET else {}

st.set_page_config(
    page_title="Core Interrogator",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    letter-spacing: -0.01em;
}
.block-container { padding-top: 2.5rem; padding-bottom: 2.5rem; max-width: 860px; }
.pipeline-tag {
    display: inline-block; font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #888; border: 0.5px solid #444;
    padding: 2px 7px; border-radius: 4px; margin-bottom: 6px;
}
.welcome-headline { font-size: 2.6rem; font-weight: 600; letter-spacing: -0.04em; line-height: 1.1; margin-bottom: 0.5rem; }
.welcome-sub { font-size: 1rem; color: #888; font-weight: 400; letter-spacing: -0.01em; margin-bottom: 2rem; }
.citation-row {
    font-family: "SF Mono","Fira Code","Consolas",monospace;
    font-size: 0.72rem; color: #aaa;
    padding: 6px 0; border-bottom: 0.5px solid rgba(128,128,128,0.12);
}
.stat-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.12em; color: #666; font-weight: 600; }
.stat-value { font-size: 1.4rem; font-weight: 500; letter-spacing: -0.03em; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

defaults = {
    "video_id": None, "video_title": None,
    "messages": [], "pending_question": None, "show_video": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helpers — all errors caught, never crash the app
# ---------------------------------------------------------------------------

def api_get(path: str, timeout: int = 10):
    try:
        r = requests.get(f"{API_URL}{path}", headers=HEADERS, timeout=timeout)
        return r
    except Exception:
        return None


def api_post(path: str, payload: dict, timeout: int = 120):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, headers=HEADERS, timeout=timeout)
        return r
    except Exception:
        return None


def wait_for_backend(max_wait: int = 70) -> bool:
    """
    Render free tier sleeps after inactivity and needs up to 60s to wake.
    Poll /health until it responds or we time out.
    """
    placeholder = st.empty()
    for i in range(max_wait):
        r = api_get("/health", timeout=5)
        if r and r.status_code == 200:
            placeholder.empty()
            return True
        placeholder.info(f"⏳ Backend is waking up… ({i + 1}s) This takes up to 60s on the free tier.")
        time.sleep(1)
    placeholder.empty()
    return False


def ask_question(question: str):
    st.session_state.messages.append({"role": "user", "content": question})
    r = api_post("/ask", {"video_id": st.session_state.video_id, "question": question})
    if r is None:
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ Backend unreachable.", "citations": []})
        return
    if r.status_code == 200:
        data = r.json()
        st.session_state.messages.append({
            "role": "assistant", "content": data["answer"], "citations": data["citations"]
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {r.json().get('detail', 'Something went wrong.')}",
            "citations": [],
        })


def export_markdown() -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# {st.session_state.video_title or st.session_state.video_id}",
        f"**Source:** https://youtube.com/watch?v={st.session_state.video_id}",
        f"**Exported:** {now}", "", "---", "",
    ]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lines.append(f"## Q: {msg['content']}\n")
        else:
            lines.append(msg["content"])
            if msg.get("citations"):
                lines.append("\n**Sources:**")
                seen = set()
                for c in msg["citations"]:
                    if c["timestamp"] not in seen:
                        seen.add(c["timestamp"])
                        lines.append(f"- [{c['timestamp']}]({c['url']})")
            lines.append("\n---\n")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⬡ Core Interrogator")
    st.caption("Long-form podcast intelligence.")
    st.divider()

    st.markdown('<div class="pipeline-tag">Input</div>', unsafe_allow_html=True)
    url_input = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
    col1, col2 = st.columns([3, 1])
    with col1:
        load_btn = st.button("Analyze", type="primary", use_container_width=True)
    with col2:
        force_cb = st.checkbox("↺", help="Force re-index")

    if load_btn and url_input:
        st.divider()
        # Wake up Render if needed
        health = api_get("/health", timeout=5)
        if health is None or health.status_code != 200:
            alive = wait_for_backend()
            if not alive:
                st.error("Backend did not respond after 70s. Try again.")
                st.stop()

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
            st.session_state.video_id    = data["video_id"]
            st.session_state.video_title = data["title"]
            st.session_state.messages    = []
            st.session_state.show_video  = False
            st.rerun()
        else:
            detail = r.json().get("detail", "Unknown error.")
            pipeline_a.update(label="Pipeline A — Error", state="error")
            pipeline_b.update(label="Pipeline B — Error", state="error")
            st.error(detail)

    st.divider()

    r = api_get("/videos")
    if r and r.status_code == 200:
        videos = r.json().get("videos", [])
        if videos:
            st.markdown('<div class="pipeline-tag">Indexed</div>', unsafe_allow_html=True)
            for vid in videos:
                label = f"▸ {vid[:26]}…" if len(vid) > 26 else f"▸ {vid}"
                if st.button(label, key=f"switch_{vid}", use_container_width=True):
                    st.session_state.video_id    = vid
                    st.session_state.video_title = vid
                    st.session_state.messages    = []
                    st.session_state.show_video  = False
                    st.rerun()

    st.divider()

    if st.session_state.video_id:
        if st.session_state.messages:
            title    = st.session_state.video_title or st.session_state.video_id
            filename = f"{title[:30].replace(' ', '_')}_notes.md"
            st.download_button(
                label="⬇ Export notes", data=export_markdown(),
                file_name=filename, mime="text/markdown", use_container_width=True,
            )
        embed_label = "Hide video" if st.session_state.show_video else "▶ Show video"
        if st.button(embed_label, use_container_width=True):
            st.session_state.show_video = not st.session_state.show_video
            st.rerun()
        if st.button("Clear session", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    health = api_get("/health", timeout=5)
    if health and health.status_code == 200:
        st.caption("🟢 Engine online")
    else:
        st.caption("🟡 Engine sleeping — will wake on first request")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if not st.session_state.video_id:
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
    col_title, col_stats = st.columns([3, 1])
    with col_title:
        st.markdown(f"#### {st.session_state.video_title or st.session_state.video_id}")
        st.caption(f"youtube.com/watch?v={st.session_state.video_id}")
    with col_stats:
        q_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
        st.markdown(f'<div class="stat-label">Queries</div><div class="stat-value">{q_count}</div>', unsafe_allow_html=True)

    if st.session_state.show_video:
        st.video(f"https://youtube.com/watch?v={st.session_state.video_id}")

    st.divider()

    if not st.session_state.messages:
        st.markdown('<div class="pipeline-tag">Suggested queries</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.pending_question = q
                    st.rerun()
        st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("citations"):
                seen, lines = set(), []
                for c in msg["citations"]:
                    if c["timestamp"] in seen:
                        continue
                    seen.add(c["timestamp"])
                    lines.append(
                        f'<div class="citation-row">'
                        f'<a href="{c["url"]}" target="_blank">[{c["timestamp"]}]</a>'
                        f' — {c["url"]}</div>'
                    )
                if lines:
                    with st.expander(f"Sources  ·  {len(lines)} timestamp(s)"):
                        st.markdown("".join(lines), unsafe_allow_html=True)

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        with st.spinner("Interrogating transcript..."):
            ask_question(question)
        st.rerun()

    if question := st.chat_input("Query the transcript..."):
        with st.spinner("Interrogating transcript..."):
            ask_question(question)
        st.rerun()
