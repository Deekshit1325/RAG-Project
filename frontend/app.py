import sys
import os
import shutil
import streamlit as st

# Ensure parent directory is in sys.path so backend imports work on Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sync Streamlit secrets to environment variables if deployed on Streamlit Cloud
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "LLM_PROVIDER" in st.secrets:
        os.environ["LLM_PROVIDER"] = st.secrets["LLM_PROVIDER"]
except Exception:
    pass

from backend.config import UPLOAD_DIR
from backend.document_loader import load_document, SUPPORTED_EXTENSIONS
from backend.chunker import chunk_document
from backend.vector_store import add_chunks, search, get_all_documents, delete_document
from backend.llm import generate_answer

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
        font-size: 0.9em;
    }
    .chat-msg {
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---- Session state init ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---- Sidebar: File upload + Document list ----
with st.sidebar:
    st.header("📁 Document Manager")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "xlsx", "txt"],
        accept_multiple_files=True,
        help="Supported: PDF, Word, PowerPoint, Excel, Text"
    )

    if uploaded_files and st.button("🚀 Process Documents", use_container_width=True):
        with st.spinner("Uploading and processing..."):
            for f in uploaded_files:
                ext = os.path.splitext(f.name)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    st.warning(f"⚠️ {f.name}: Unsupported type")
                    continue

                filepath = os.path.join(UPLOAD_DIR, f.name)
                with open(filepath, "wb") as out_file:
                    out_file.write(f.getvalue())

                pages = load_document(filepath)
                if not pages:
                    st.warning(f"⚠️ {f.name}: No text extracted")
                    continue

                chunks = chunk_document(pages)
                num_stored = add_chunks(chunks)
                st.success(f"✅ {f.name}: {num_stored} chunks indexed")

    st.divider()
    st.subheader("📄 Indexed Documents")

    if st.button("🔄 Refresh List", use_container_width=True):
        st.rerun()

    try:
        docs = get_all_documents()
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                col1.write(f"📄 {doc}")
                if col2.button("🗑️", key=f"del_{doc}"):
                    delete_document(doc)
                    st.success(f"Deleted {doc}")
                    st.rerun()
        else:
            st.info("No documents uploaded yet")
    except Exception as e:
        st.warning(f"Error fetching documents: {e}")


# ---- Main area: Chat interface ----
st.title("📚 Enterprise RAG Assistant")
st.caption("Upload documents and ask questions. Answers are grounded in your uploaded content.")

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">📄 <b>{src["document"]}</b> — '
                            f'Page/Section: {src["page"]} '
                            f'(Relevance: {src["relevance_score"]:.1%})</div>',
                            unsafe_allow_html=True
                        )

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            hits = search(question, top_k=5)

            if not hits:
                answer = "No documents have been uploaded yet, or no relevant content was found."
                sources = []
            else:
                answer = generate_answer(question, hits)
                sources = []
                seen = set()
                for h in hits:
                    key = (h["source"], h["page"])
                    if key not in seen:
                        seen.add(key)
                        sources.append({
                            "document": h["source"],
                            "page": h["page"],
                            "relevance_score": round(h["score"], 3)
                        })

            st.write(answer)

            if sources:
                with st.expander("📌 Sources", expanded=True):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-box">📄 <b>{src["document"]}</b> — '
                            f'Page/Section: {src["page"]} '
                            f'(Relevance: {src["relevance_score"]:.1%})</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

# Footer
st.divider()
st.caption("Built with ChromaDB, Sentence-Transformers, Groq, and Streamlit")
