"""
Streamlit Web UI – RFP Intelligence Platform (Simple Version)
Upload PDF/HTML bid files, search, chat and extract structured data.
"""
from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RFP Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*, body { font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0f1117; }

/* Header */
h1 { color: #a78bfa !important; font-weight: 700 !important; }
h2, h3 { color: #e2e8f0 !important; }

/* Cards */
.card {
    background: #1e1e2e;
    border: 1px solid #2d2d44;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}

/* Result highlight */
.result-box {
    background: #1a1a2e;
    border-left: 3px solid #7c3aed;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    color: #cbd5e0;
    font-size: 0.9rem;
    line-height: 1.6;
}

.tag {
    display: inline-block;
    background: rgba(124, 58, 237, 0.2);
    color: #a78bfa;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 5px;
}

.tag-green {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #7c3aed !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_engine():
    """Load search engine once and cache it."""
    try:
        from search.vector_store import MongoVectorStore
        from search.retrieval import HybridSearchEngine
        vs = MongoVectorStore()
        engine = HybridSearchEngine(vs)
        return engine, vs
    except Exception as e:
        return None, None


def get_indexed_bids(vs) -> list[str]:
    if vs is None:
        return []
    try:
        return [b["_id"] for b in vs.chunks.aggregate([{"$group": {"_id": "$bid_id"}}, {"$sort": {"_id": 1}}])]
    except Exception:
        return []


def save_uploads_to_temp(uploaded_files, bid_id: str) -> str:
    """Save uploaded files to a temp folder and return the folder path."""
    tmp_dir = ROOT / "data" / "bids" / bid_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for uf in uploaded_files:
        dest = tmp_dir / uf.name
        with open(dest, "wb") as f:
            f.write(uf.getbuffer())
    return str(tmp_dir)


# ── Load engine ───────────────────────────────────────────────────────────────
engine, vs = load_engine()
indexed_bids = get_indexed_bids(vs)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 RFP Intelligence")
    st.caption("AI-powered bid document analysis")
    st.markdown("---")

    # Connection status
    if engine and vs:
        st.success("✅ Database connected")
    else:
        st.error("❌ Database not connected")
        st.caption("Check your .env MongoDB URI")

    st.markdown("---")

    # Indexed bids info
    if indexed_bids:
        st.markdown("### 📁 Indexed Bids")
        for b in indexed_bids:
            st.markdown(f"- `{b}`")
    else:
        st.info("No bids indexed yet.\nUpload files in the **Upload** tab.")

    st.markdown("---")
    st.caption("💡 Start with the **Upload** tab to add your documents.")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📄 RFP Intelligence Platform")
st.markdown("Upload your bid documents, then search, ask questions, or extract key data automatically.")
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_search, tab_chat, tab_extract = st.tabs([
    "📤 Upload & Index",
    "🔍 Search",
    "💬 Ask a Question",
    "📊 Extract Data",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Upload & Index
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Step 1 — Upload your bid documents")
    st.markdown("Supported formats: **PDF** and **HTML** files (base RFP + addendums, all at once)")

    bid_id = st.text_input(
        "Give this bid a name (Bid ID)",
        placeholder="e.g.  Bid1  or  DellLaptop2024",
        help="This is just a label so you can filter results later.",
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "html", "htm"],
        accept_multiple_files=True,
        help="Upload the RFP PDF, any addendum PDFs, and the bid webpage (HTML) together.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) ready to upload:**")
        for f in uploaded_files:
            size_kb = round(len(f.getbuffer()) / 1024, 1)
            st.markdown(f"  - 📄 `{f.name}` ({size_kb} KB)")

    col1, col2 = st.columns([1, 3])
    with col1:
        index_btn = st.button("🚀 Index Bid", use_container_width=True, disabled=not (bid_id and uploaded_files and vs))

    if index_btn:
        with st.spinner(f"Saving and indexing **{bid_id}**… please wait"):
            try:
                folder_path = save_uploads_to_temp(uploaded_files, bid_id)
                from search.indexing import index_bid_folder
                summary = index_bid_folder(folder_path, bid_id, vs, force_reindex=False)

                status = summary.get("status", "")
                if status == "already_indexed":
                    st.warning(f"⚠️ **{bid_id}** is already indexed. Delete it from MongoDB and re-upload to re-index.")
                else:
                    chunks = summary.get("chunks_indexed", 0)
                    files  = summary.get("files_parsed", 0)
                    st.success(f"✅ Done! Indexed **{chunks} chunks** from **{files} files**.")
                    st.balloons()
                    st.info("👉 Now go to **Search**, **Ask a Question**, or **Extract Data** tabs.")
                if summary.get("errors"):
                    with st.expander(f"⚠️ {len(summary['errors'])} warning(s)"):
                        for e in summary["errors"]:
                            st.caption(e)
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")
    elif not vs:
        st.warning("Database not connected. Check your .env MongoDB URI.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Search
# ═══════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### Search your documents")
    st.caption("Type anything — the AI finds the most relevant passages.")

    col_q, col_f = st.columns([3, 1])
    with col_q:
        query = st.text_input("Search query", placeholder="e.g. submission deadline, bid bond, warranty requirements")
    with col_f:
        bid_filter = st.selectbox("Filter by bid", ["All bids"] + indexed_bids)

    top_k = st.slider("Number of results", 3, 15, 5)
    search_btn = st.button("🔍 Search", disabled=not (engine and query))

    if search_btn and query and engine:
        with st.spinner("Searching…"):
            try:
                from models import SearchQuery
                resp = engine.search(SearchQuery(
                    query=query,
                    bid_id=None if bid_filter == "All bids" else bid_filter,
                    top_k=top_k,
                    use_rerank=True,
                    expand_query=True,
                ))

                if not resp.results:
                    st.info("No results found. Try a different query or index your documents first.")
                else:
                    st.markdown(f"**{resp.total_found} relevant passages found:**")
                    for r in resp.results:
                        st.markdown(f"""
<div class="result-box">
  <span class="tag">📄 {r.file_name}</span>
  <span class="tag">Page {r.page_number}</span>
  <span class="tag tag-green">Score {r.score:.2f}</span>
  <br><br>
  {r.text[:500]}{"…" if len(r.text) > 500 else ""}
</div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Search error: {e}")

    elif not indexed_bids:
        st.info("📤 Upload and index your documents first (see the **Upload & Index** tab).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Chat / Q&A
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### Ask a question about your bid documents")
    st.caption("The AI reads the relevant sections and gives you a cited answer.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Bid filter
    qa_bid = st.selectbox("Bid to query", ["All bids"] + indexed_bids, key="qa_bid")

    # Example questions
    st.markdown("**Quick examples — click to ask:**")
    examples = [
        "What is the submission deadline?",
        "Is a bid bond required?",
        "What are the technical specifications?",
        "What affidavits are required?",
        "What changed in the addendum?",
        "Who is the procurement contact?",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(examples):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["pending_q"] = ex

    st.markdown("---")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle pending example question
    if "pending_q" in st.session_state:
        q = st.session_state.pop("pending_q")
        st.session_state.messages.append({"role": "user", "content": q})
        if engine:
            with st.spinner("Reading documents…"):
                try:
                    from agents.qa_agent import answer_question
                    bid_filter_qa = None if qa_bid == "All bids" else qa_bid
                    answer = answer_question(q, bid_filter_qa, engine)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
        st.rerun()

    # Chat input
    user_q = st.chat_input("Type your question here…")
    if user_q and engine:
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.spinner("Reading documents…"):
            try:
                from agents.qa_agent import answer_question
                bid_filter_qa = None if qa_bid == "All bids" else qa_bid
                answer = answer_question(user_q, bid_filter_qa, engine)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Error: {e}"})
        st.rerun()
    elif user_q and not engine:
        st.error("Database not connected.")

    if st.session_state.messages:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    if not indexed_bids:
        st.info("📤 Upload and index your documents first (see the **Upload & Index** tab).")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Extract Data
# ═══════════════════════════════════════════════════════════════════════════════
with tab_extract:
    st.markdown("### Auto-extract structured data")
    st.caption("The AI reads the entire bid and fills in 20 key fields automatically.")

    if not indexed_bids:
        st.info("📤 Upload and index your documents first (see the **Upload & Index** tab).")
    else:
        extract_bid = st.selectbox("Choose a bid to extract", indexed_bids, key="extract_bid_sel")

        st.markdown("**What gets extracted:**")
        fields_preview = [
            "Bid Title & Number", "Company / Agency Name", "Submission Deadline",
            "Bid Bond Requirement", "Pre-bid Meeting", "Contract Term",
            "Payment Terms", "Delivery Requirements", "Technical Specifications",
            "Model & Part Numbers", "Warranty", "Installation Requirements",
            "Required Affidavits", "Procurement Contact", "Submission Method",
            "Addendum Changes", "Bid Summary", "Cooperative Contract", "MFG Registration", "Product List",
        ]
        cols = st.columns(4)
        for i, f in enumerate(fields_preview):
            cols[i % 4].markdown(f"✔ {f}")

        st.markdown("")
        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            extract_btn = st.button("🤖 Run Live Extraction", use_container_width=True)
        with col_btn2:
            existing_file = ROOT / "output" / f"{extract_bid}_extraction.json"
            load_existing_btn = st.button(
                "📂 Load Saved Results",
                use_container_width=False,
                disabled=not existing_file.exists(),
                help="Instantly loads pre-computed extraction JSON if available",
            )

        if load_existing_btn and existing_file.exists():
            with open(existing_file, "r", encoding="utf-8") as f:
                st.session_state["extraction_result"] = json.load(f)
            st.success(f"✅ Loaded saved extraction results for **{extract_bid}**!")

        if extract_btn:
            bid_folder = str(ROOT / "data" / "bids" / extract_bid)
            with st.spinner(f"Running multi-agent extraction on **{extract_bid}**… (takes ~1.5 minutes)"):
                try:
                    from agents.orchestrator import RFPOrchestrator
                    orch = RFPOrchestrator()
                    result = orch.run_extraction(bid_folder, extract_bid, str(ROOT / "output"))
                    st.session_state["extraction_result"] = result
                    st.success("✅ Extraction complete!")
                except Exception as e:
                    st.error(f"❌ Extraction error: {e}")

        if "extraction_result" in st.session_state:
            data = st.session_state["extraction_result"]
            fields = data.get("fields", {})
            validation = data.get("validation", {})
            changes = data.get("addendum_changes", [])

            # Summary metrics
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("✅ Extracted", validation.get("passed", 0))
            m2.metric("⚠️ Not Found", validation.get("not_found", 0))
            m3.metric("❌ Failed", validation.get("failed", 0))
            m4.metric("📝 Total", len(fields))

            # Fields table
            st.markdown("#### Extracted Fields")
            for name, fd in fields.items():
                val = fd.get("value")
                conf = fd.get("confidence", 0)
                color = "#34d399" if conf >= 0.7 else "#fbbf24" if conf >= 0.4 else "#f87171"
                val_str = str(val)[:120] if val else "*Not found*"
                with st.expander(f"**{name}** → {val_str}"):
                    st.markdown(f"**Value:** {str(val) if val else 'Not found in documents'}")
                    st.markdown(f"**Confidence:** :{color.lstrip('#')}[{conf:.0%}]")
                    if fd.get("notes"):
                        st.caption(fd["notes"])
                    if fd.get("sources"):
                        for src in fd["sources"]:
                            st.caption(f"📄 {src.get('file')} — Page {src.get('page')}")

            # Addendum changes
            if changes:
                st.markdown("#### 📝 Addendum Changes Detected")
                for c in changes:
                    st.markdown(
                        f"- **{c.get('field')}**: ~~{c.get('original_value')}~~ → **{c.get('new_value')}** "
                        f"*(Addendum {c.get('addendum_number')})*"
                    )

            # Download
            st.markdown("---")
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(data, indent=2, default=str),
                file_name=f"{extract_bid}_extraction.json",
                mime="application/json",
            )
