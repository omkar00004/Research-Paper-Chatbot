"""
Research Paper QA Chatbot -  Streamlit Frontend
Premium chat interface with RAG-powered question answering.
"""
import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "app.log", mode="a"),
    ],
)
logger = logging.getLogger("ResearchQA")

import streamlit as st

from backend.pipeline import ingest_papers, ingest_single_paper, query
from backend.indexing import get_collection_stats, delete_paper, clear_index
from backend.config import PAPERS_DIR, UPLOAD_DIR

# ── Page Configuration ──────────────────────────────────
st.set_page_config(
    page_title="Research Paper QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Custom CSS ──────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.95rem;
        margin: 0.3rem 0 0;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1923 0%, #1a2332 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #cbd5e0 !important;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        margin: 0.5rem 0 !important;
        animation: fadeInUp 0.3s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Source reference cards */
    .source-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.3rem 0;
        transition: all 0.2s ease;
        border-left: 3px solid #4299e1;
    }
    .source-card:hover {
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.15);
        transform: translateY(-1px);
    }
    .source-card .paper-name {
        font-weight: 600;
        color: #2d3748;
        font-size: 0.85rem;
    }
    .source-card .page-info {
        color: #718096;
        font-size: 0.78rem;
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a88 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #63b3ed;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Pipeline info */
    .pipeline-info {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        font-size: 0.78rem;
        color: #718096;
        margin-top: 0.5rem;
    }

    /* Paper list items */
    .paper-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.85rem;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Improve text input */
    .stChatInput {
        border-radius: 12px !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ──────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "papers_ingested" not in st.session_state:
    st.session_state.papers_ingested = False
if "ingestion_stats" not in st.session_state:
    st.session_state.ingestion_stats = None


# ── Helper Functions ────────────────────────────────────
def render_sources(sources):
    """Render source references as styled cards."""
    if not sources:
        return

    st.markdown("**📑 Sources:**")
    for src in sources:
        paper = src.get("paper", "Unknown")
        page = src.get("page", "?")
        section = src.get("section", "")
        section_text = f" • {section}" if section and section != "Unknown" else ""

        st.markdown(
            f'<div class="source-card">'
            f'<span class="paper-name">📄 {paper}</span>'
            f'<span class="page-info"> -  Page {page}{section_text}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_pipeline_info(pipeline_data, total_time):
    """Render pipeline execution info."""
    if not pipeline_data:
        return

    steps = []
    for step in pipeline_data:
        name = step["step"].title()
        t = step.get("time", 0)
        count = step.get("count", "")
        count_str = f" ({count} results)" if count != "" else ""
        steps.append(f"**{name}**: {t:.2f}s{count_str}")

    info = " → ".join(steps)
    st.markdown(
        f'<div class="pipeline-info">⚡ Pipeline: {info} | Total: {total_time:.2f}s</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 Research Paper QA")
    st.markdown("*RAG-powered chatbot*")
    st.divider()

    # ── Paper Ingestion ──
    st.markdown("### 📥 Paper Management")

    if st.button("🔄 Ingest Research Papers", use_container_width=True, type="primary"):
        with st.spinner("Ingesting papers... This may take a moment."):
            stats = ingest_papers()
            st.session_state.papers_ingested = True
            st.session_state.ingestion_stats = stats

            if stats["status"] == "success":
                st.success(f"✅ Indexed {stats['new_chunks']} new chunks in {stats['time']}s")
            elif stats["status"] == "up_to_date":
                st.info("All papers already indexed!")
            else:
                st.warning(stats.get("message", "No papers found"))

    # ── Upload New Paper ──
    st.divider()
    st.markdown("### ⬆️ Upload Paper")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        upload_path = UPLOAD_DIR / uploaded_file.name
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            result = ingest_single_paper(upload_path)

        if result["status"] == "success":
            st.success(f"✅ {uploaded_file.name}: {result['chunks']} chunks indexed")
        else:
            st.error(result.get("message", "Ingestion failed"))

    # ── Stats ──
    st.divider()
    st.markdown("### 📊 Index Statistics")
    try:
        stats = get_collection_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{stats["total_papers"]}</div>'
                f'<div class="stat-label">Papers</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-value">{stats["total_chunks"]}</div>'
                f'<div class="stat-label">Chunks</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if stats["papers"]:
            st.markdown("### 📄 Indexed Papers")
            for paper in stats["papers"]:
                col_name, col_del = st.columns([0.8, 0.2])
                with col_name:
                    st.markdown(
                        f'<div class="paper-item">📄 {paper}</div>',
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("🗑️", key=f"del_{paper}", help=f"Delete {paper}"):
                        delete_paper(paper)
                        st.success(f"Deleted {paper}")
                        st.rerun()

        st.divider()
        if st.button("🚨 Reset Vector Index", use_container_width=True, help="Delete ALL indexed chunks"):
            if clear_index():
                st.success("Vector index cleared!")
                st.rerun()
            else:
                st.error("Failed to clear index.")
    except Exception as e:
        st.caption(f"No papers indexed yet.")

    # ── Clear Chat ──
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # ── Info ──
    st.divider()
    st.caption(
        "**Tech Stack:** all-MiniLM-L6-v2 · ChromaDB · FlashRank · Groq LLM"
    )


# ── Main Chat Area ───────────────────────────────────────
st.markdown(
    '<div class="main-header">'
    '<h1>📚 Research Paper QA Chatbot</h1>'
    '<p>Ask questions about your research papers -  answers grounded in evidence with source citations</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

        # Show pipeline info
        if msg["role"] == "assistant" and msg.get("pipeline"):
            render_pipeline_info(msg["pipeline"], msg.get("time", 0))

# Chat input
if prompt := st.chat_input("Ask a question about your research papers..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching papers and generating answer..."):
            result = query(prompt)

        st.markdown(result["answer"])

        if result.get("sources"):
            render_sources(result["sources"])

        if result.get("pipeline"):
            render_pipeline_info(result["pipeline"], result.get("time", 0))

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "pipeline": result.get("pipeline", []),
        "time": result.get("time", 0),
    })
