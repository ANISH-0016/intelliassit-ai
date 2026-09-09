"""
IntelliAssist AI - Smart Document AI Assistant
================================================
Streamlit front-end tying together document ingestion, RAG-based chat,
summarisation, and sentiment/intent analysis.

Run locally:
    streamlit run app.py
"""

import time
import pandas as pd
import streamlit as st

from src.document_processor import process_document
from src.vectorstore import VectorStore
from src.rag_engine import RAGEngine
from src.summarizer import summarize
from src.sentiment_intent import analyze

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Session state initialisation
# --------------------------------------------------------------------------- #
def init_state():
    defaults = {
        "vector_store": VectorStore(),
        "documents": [],          # list of {"name":..., "chunks": n}
        "chat_history": [],       # list of {"question","answer","sources","sentiment","intent"}
        "openai_key": "",
        "gemini_key": "",
        "full_text_cache": {},    # filename -> concatenated text (for summarisation)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧠 IntelliAssist AI")
    st.caption("Smart Document AI Assistant · RAG-powered")

    st.markdown("### ⚙️ Settings")
    with st.expander("API Keys (optional)", expanded=False):
        st.session_state.openai_key = st.text_input(
            "OpenAI API Key", type="password", value=st.session_state.openai_key,
            help="Optional. Enables higher-quality answers via GPT-4o-mini. "
                 "Leave blank to use the built-in local/extractive engine.",
        )
        st.session_state.gemini_key = st.text_input(
            "Google Gemini API Key", type="password", value=st.session_state.gemini_key,
            help="Optional alternative to OpenAI.",
        )
        st.caption(
            "No key? No problem — IntelliAssist automatically falls back to a "
            "local model / extractive engine so the app always works."
        )

    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            already = any(d["name"] == f.name for d in st.session_state.documents)
            if already:
                continue
            with st.spinner(f"Processing {f.name}..."):
                file_bytes = f.read()
                try:
                    chunks = process_document(f.name, file_bytes)
                except Exception as e:
                    st.error(f"Failed to process {f.name}: {e}")
                    continue

                st.session_state.vector_store.add(chunks)
                st.session_state.documents.append({"name": f.name, "chunks": len(chunks)})
                st.session_state.full_text_cache[f.name] = " ".join(c.text for c in chunks)
            st.success(f"✅ {f.name} indexed ({len(chunks)} chunks)")

    if st.session_state.documents:
        st.markdown("### 📚 Indexed Documents")
        for d in st.session_state.documents:
            st.markdown(f"- **{d['name']}** — {d['chunks']} chunks")

        backend = st.session_state.vector_store.backend
        st.caption(f"Embedding backend: `{backend.upper()}`")

        if st.button("🗑️ Clear all documents"):
            st.session_state.vector_store = VectorStore()
            st.session_state.documents = []
            st.session_state.full_text_cache = {}
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload a document to get started.")

# --------------------------------------------------------------------------- #
# Main tabs
# --------------------------------------------------------------------------- #
st.title("IntelliAssist AI 🧠")
st.caption("Ask questions, get summaries, and explore insights from your documents.")

tab_chat, tab_summary, tab_insights, tab_history, tab_about = st.tabs(
    ["💬 Chat", "📝 Summarize", "📊 Insights", "🕘 History", "ℹ️ About"]
)

# --------------------------------------------------------------------------- #
# Chat tab
# --------------------------------------------------------------------------- #
with tab_chat:
    if st.session_state.vector_store.is_empty():
        st.warning("Upload at least one document from the sidebar to start chatting.")
    else:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
            with st.chat_message("assistant"):
                st.markdown(turn["answer"])
                if turn.get("sources"):
                    with st.expander("📎 Sources"):
                        for s in turn["sources"]:
                            loc = s["source"] + (f", page {s['page']}" if s.get("page") else "")
                            st.markdown(f"**{loc}**")
                            st.caption(s["text"][:300] + ("..." if len(s["text"]) > 300 else ""))

        question = st.chat_input("Ask a question about your documents...")
        if question:
            with st.chat_message("user"):
                st.markdown(question)

            engine = RAGEngine(
                st.session_state.vector_store,
                openai_api_key=st.session_state.openai_key or None,
                gemini_api_key=st.session_state.gemini_key or None,
            )

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = engine.answer(question, chat_history=st.session_state.chat_history)
                    meta = analyze(question)
                st.markdown(result.answer)
                st.caption(f"Engine: `{result.backend_used}`  ·  Intent: `{meta['intent']}`  ·  "
                           f"Sentiment: `{meta['sentiment']['label']}`")
                if result.sources:
                    with st.expander("📎 Sources"):
                        for s, sc in zip(result.sources, result.scores):
                            loc = s.source + (f", page {s.page}" if s.page else "")
                            st.markdown(f"**{loc}**  ·  relevance {sc:.0%}")
                            st.caption(s.text[:300] + ("..." if len(s.text) > 300 else ""))

            st.session_state.chat_history.append({
                "question": question,
                "answer": result.answer,
                "sources": [
                    {"source": s.source, "page": s.page, "text": s.text} for s in result.sources
                ],
                "sentiment": meta["sentiment"],
                "intent": meta["intent"],
                "engine": result.backend_used,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            st.rerun()

# --------------------------------------------------------------------------- #
# Summarize tab
# --------------------------------------------------------------------------- #
with tab_summary:
    if not st.session_state.documents:
        st.warning("Upload a document first.")
    else:
        doc_name = st.selectbox("Choose a document to summarize", 
                                  [d["name"] for d in st.session_state.documents])
        num_sentences = st.slider("Summary length (sentences, used for extractive fallback)", 3, 10, 5)

        if st.button("Generate Summary", type="primary"):
            with st.spinner("Summarizing..."):
                text = st.session_state.full_text_cache.get(doc_name, "")
                result = summarize(text, num_sentences=num_sentences)
            st.markdown("### Summary")
            st.info(result["summary"])
            st.caption(f"Method: `{result['method']}`")

# --------------------------------------------------------------------------- #
# Insights tab
# --------------------------------------------------------------------------- #
with tab_insights:
    if not st.session_state.chat_history:
        st.info("Ask a few questions in the Chat tab to see sentiment/intent analytics here.")
    else:
        df = pd.DataFrame([
            {
                "Question": t["question"],
                "Sentiment": t["sentiment"]["label"],
                "Confidence": round(t["sentiment"]["score"], 2),
                "Intent": t["intent"],
                "Engine": t.get("engine", "-"),
                "Time": t.get("timestamp", ""),
            }
            for t in st.session_state.chat_history
        ])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Sentiment distribution")
            st.bar_chart(df["Sentiment"].value_counts())
        with col2:
            st.markdown("#### Intent distribution")
            st.bar_chart(df["Intent"].value_counts())

        st.markdown("#### Query Log")
        st.dataframe(df, use_container_width=True)

# --------------------------------------------------------------------------- #
# History tab
# --------------------------------------------------------------------------- #
with tab_history:
    if not st.session_state.chat_history:
        st.info("No conversation history yet.")
    else:
        for i, turn in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {turn['question'][:80]}"):
                st.markdown(f"**Q:** {turn['question']}")
                st.markdown(f"**A:** {turn['answer']}")
                st.caption(f"{turn.get('timestamp','')} · sentiment={turn['sentiment']['label']} · "
                           f"intent={turn['intent']} · engine={turn.get('engine','-')}")

        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown("""
    ### About IntelliAssist AI
    **IntelliAssist AI** is an AI-powered document assistant that lets you upload
    PDFs, DOCX, or TXT files and interact with them using natural language.

    **Core capabilities**
    - 📄 Multi-format document upload (PDF / TXT / DOCX)
    - 🔎 Semantic search using TF-IDF / BERT embeddings + FAISS
    - 💬 RAG-based chatbot with source citations
    - 📝 Abstractive & extractive summarisation
    - 😊 Sentiment & intent analysis on user queries
    - 🕘 Persistent conversation history within a session

    **Tech stack:** Streamlit · Python · LangChain-style RAG · Hugging Face ·
    OpenAI / Gemini (optional) · TF-IDF · Sentence-Transformers (BERT) · FAISS

    Built as an AI capstone project.
    """)
