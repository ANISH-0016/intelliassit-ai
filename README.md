# 🧠 IntelliAssist AI — Smart Document AI Assistant

An AI-powered chatbot that lets you upload documents (PDF / DOCX / TXT) and ask
questions about them using **Retrieval-Augmented Generation (RAG)**. Built as an
Artificial Intelligence Major Capstone Project.

> **Why this project is demo-safe:** IntelliAssist AI works **fully offline with
> zero API keys** (TF-IDF retrieval + extractive answers/summaries), and
> automatically upgrades to BERT embeddings, a local Flan-T5 model, or
> OpenAI/Gemini if those are available. You will never see a blank screen or a
> crash during your live demo, even with no internet.

---

## ✨ Features

| Feature | How it's implemented |
|---|---|
| Document Upload (PDF/TXT/DOCX) | `pypdf`, `python-docx` |
| Semantic Search | TF-IDF (scikit-learn) → auto-upgrades to BERT (`sentence-transformers`) |
| Vector Database | FAISS (auto-upgrades from NumPy cosine similarity) |
| RAG Chatbot | Custom retrieval + generation pipeline (OpenAI / Gemini / local Flan-T5 / extractive) |
| Summarisation | Abstractive (BART) with TF-IDF extractive fallback |
| Sentiment & Intent Analysis | DistilBERT / TextBlob + rule-based intent classifier |
| Source Citation | Every answer links back to filename + page number |
| Conversation History | Session-based chat log + analytics dashboard |
| Web Interface | Streamlit |

---

## 🗂️ Project Structure

```
intelliassist/
├── app.py                     # Streamlit UI (entry point)
├── requirements.txt
├── .env.example                # Optional API keys template
├── .streamlit/config.toml      # Theme config
├── src/
│   ├── document_processor.py   # File parsing + chunking
│   ├── vectorstore.py          # Embeddings + FAISS/TF-IDF search
│   ├── rag_engine.py           # Retrieval + multi-backend generation
│   ├── summarizer.py           # Abstractive/extractive summarisation
│   └── sentiment_intent.py     # Sentiment & intent analysis
└── data/
    └── sample_docs/            # Sample document for demo/testing
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone / unzip the project, then cd into it
cd intelliassist

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add API keys for higher-quality answers
cp .env.example .env
# edit .env and add OPENAI_API_KEY or GEMINI_API_KEY — or skip this entirely

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload a file from
`data/sample_docs/about_intelliassist.txt` (or your own PDF/DOCX/TXT) and start
chatting.

**No API key? No problem.** Leave the key fields blank in the sidebar — the app
still fully works using local TF-IDF search and an extractive answering engine.
If `sentence-transformers` / `transformers` are installed, it silently upgrades
to BERT embeddings and a local Flan-T5 generator, no key needed.

---

## 🧩 Architecture / RAG Workflow

```
 Upload File → Extract Text → Chunk (800 chars, 150 overlap)
        │
        ▼
 Generate Embeddings (TF-IDF or BERT) → Store in Vector Index (FAISS)
        │
        ▼
 User Question → Embed Query → Similarity Search (top-k chunks)
        │
        ▼
 Build Context from retrieved chunks
        │
        ▼
 Generate Answer:
   1) OpenAI (if key set)
   2) Gemini (if key set)
   3) Local Flan-T5 (if transformers installed)
   4) Extractive fallback (always works)
        │
        ▼
 Display Answer + Source Citations + Sentiment/Intent of the query
```

---

## ☁️ Deployment

### Streamlit Community Cloud (recommended, free)
1. Push this project to a public/private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Select your repo, branch, and set the main file to `app.py`.
4. (Optional) Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   GEMINI_API_KEY = "..."
   ```
5. Click **Deploy**. Done.

### Render
1. Create a new **Web Service** from your GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add environment variables for API keys if desired.

---

## 🎥 For Your Demo Video (5–10 min)

Suggested flow to hit all the submission-guideline points:
1. **Intro (English, mandatory):** your name, college, course, year, internship domain, one-line project overview.
2. **Problem statement:** explain the document-search pain point.
3. **Architecture walkthrough:** show the diagram above / explain RAG briefly.
4. **Live demo:**
   - Upload `data/sample_docs/about_intelliassist.txt` (or your own PDF).
   - Ask 2–3 questions in the **Chat** tab, show the source citations.
   - Show the **Summarize** tab generating a summary.
   - Show the **Insights** tab (sentiment/intent charts).
   - Show the **History** tab.
5. **Challenges & solutions:** e.g. "handled missing API keys with automatic fallback to local models so the app never breaks."
6. **Final output/results + wrap-up.**

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend / AI:** Python, custom RAG pipeline, Hugging Face, OpenAI/Gemini API (optional)
- **NLP:** TF-IDF, BERT Embeddings (sentence-transformers)
- **Vector DB:** FAISS (NumPy cosine-similarity fallback)
- **Deployment:** Streamlit Cloud / Render

---

## 📌 Notes for Evaluators

- All four "graceful fallback" layers were a deliberate design choice to
  guarantee reliability during grading/demo without requiring paid API access.
- Chunking uses sentence-aware splitting with overlap to preserve context
  across chunk boundaries, improving retrieval quality.
- The app is stateless between sessions by default (in-memory vector store);
  `VectorStore.save()` / `.load()` are provided if persistence across runs is
  needed.
