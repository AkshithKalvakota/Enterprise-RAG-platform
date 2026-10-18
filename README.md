<div align="center">

# 🤖 Enterprise RAG Platform

**A production-grade Retrieval-Augmented Generation platform for multi-format enterprise document intelligence**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-FC1C4F?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Gemini_3.5_Flash--Lite-8E75B2?style=for-the-badge&logo=google&logoColor=white)

</div>

A production-grade **Retrieval-Augmented Generation (RAG)** application designed to ingest multi-format enterprise documents, generate high-dimensional vector embeddings, and provide accurate, context-aware answers using natural language querying.

Built with a decoupled microservice architecture, the platform separates a robust asynchronous backend from an interactive frontend web application.

---

## 🚀 Live Demo

* **Application Link :** https://enterprise-rag-platform-ip8elc4rcvtw89ssqsuy2s.streamlit.app/


---

## ✨ Key Features

* **Multi-Format Ingestion:** Seamlessly parses PDF, DOCX, PPTX, XLSX, CSV, TXT, and Markdown files.
* **Vector Intelligence:** Leverages Qdrant Cloud for high-performance, dense vector similarity search.
* **Advanced LLM Integration:** Powered by Google's `gemini-3.5-flash-lite` (configurable via `LLM_MODEL`) for fast, grounded generation and `gemini-embedding-001` for semantic chunk mapping.
* **Source Citations:** Every answer lists the files it used, down to the PDF page, slide, or Excel sheet.
* **Memory-Aware Chat:** Follow-up questions ("what about her department?") are rewritten using the chat history before searching.
* **Per-Session Isolation:** Each browser session only sees and deletes its own documents.
* **Secured API:** API-key authentication, per-session rate limiting, upload validation, and prompt-injection guarding.
* **Streaming Answers:** Responses appear word by word as Gemini writes them.
* **Resilient to Model Overload:** Busy or rate-limited Gemini calls are retried, and answers fall back to a second model.
* **Automatic Cleanup:** Documents from abandoned sessions are deleted after 24 hours.
* **Decoupled Architecture:** Clean separation of concerns between the frontend UI and the backend ingestion/inference engine.
* **Cloud Native:** Fully deployed to the cloud using Render (Backend) and Streamlit Community Cloud (Frontend).

---

## 🔄 How It Works (The Workflow)

The platform operates on a two-part asynchronous pipeline: **Ingestion** and **Retrieval/Generation**.

### 1. Document Ingestion Pipeline
1. **Upload:** A user uploads business documents via the Streamlit frontend.
2. **Transfer:** The frontend clears the session's previous documents, then sends the files to the FastAPI backend via the `/upload` endpoint (type and size are validated).
3. **Parsing & Chunking:** The `document_parser.py` and `chunker.py` services extract text page by page (PDF), slide by slide (PPTX), and sheet by sheet (XLSX), including Word tables, and split it into overlapping chunks that remember their source and location.
4. **Embedding & Indexing:** The `vector_store.py` service embeds the chunks with Google's embedding model and stores them in Qdrant Cloud, tagged with the user's session ID.

### 2. Retrieval & Generation Pipeline
1. **Query Formulation:** The user asks a natural language question in the chat interface.
2. **Question Rewriting:** If the question is a follow-up, it is rewritten into a standalone question using the recent chat history.
3. **Context Retrieval:** The query is vectorized, and Qdrant performs a cosine similarity search over **this session's chunks only** to retrieve the top-K most relevant ones.
4. **Grounded Prompting:** The `rag_engine.py` service places the numbered chunks inside `<documents>` tags, marked as untrusted data, in a strict system prompt with recent chat history.
5. **LLM Synthesis:** Gemini answers *strictly* from that context and reports which chunks it used, which are returned to the user as source citations.

---

## 📂 Project Structure

```text
Enterprise-RAG-platform/
│
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── routes.py               # API endpoint definitions (/upload, /index, /query, /session)
│   ├── security.py             # API-key check, session-ID validation, rate limiting
│   ├── cleanup.py              # Hourly deletion of abandoned sessions
│   └── dependencies.py         # Shared Qdrant/Gemini clients, created once
│
├── database/
│   └── vector_store.py         # Vector database connection and indexing
│
├── frontend/
│   ├── app.py                  # Streamlit web interface
│   └── requirements.txt        # Frontend-only dependencies (Streamlit Cloud)
│
├── services/
│   ├── chunker.py               # Semantic text chunking logic
│   ├── document_parser.py       # Multi-format document loading and parsing
│   └── rag_engine.py            # LLM integration and retrieval chain logic
│
├── utils/
│   ├── config.py                # All settings, read from environment variables
│   ├── retry.py                 # Retry-with-backoff for busy Gemini calls
│   └── logger.py                # Application-wide logging configuration
│
├── .env.example                 # Example environment variable file
├── .gitignore
├── README.md
├── requirements.txt              # Backend dependencies (pinned)
└── runtime.txt                   # Runtime environment configuration
```

---

## 🔌 API Endpoints

The FastAPI backend exposes the following core endpoints. All endpoints except `/status` require two headers:
`X-API-Key` (when `API_KEY` is set on the server) and `X-Session-ID` (any 8–64 character ID chosen by the client).

| Endpoint | Method | Description |
|:--|:--:|:--|
| `/api/v1/status` | `GET` | Health check, also used to wake the server. No API key needed. |
| `/api/v1/health` | `GET` | Checks that Qdrant answers (used by the keep-alive workflow). No API key needed. |
| `/api/v1/upload` | `POST` | Accepts multi-part form data and saves the session's documents (allowed types only, max 20 MB). |
| `/api/v1/index/{filename}` | `POST` | Parses a saved document, chunks the text, and stores embeddings in Qdrant. |
| `/api/v1/query` | `POST` | Accepts a question and chat history; returns `{"answer", "sources"}`. |
| `/api/v1/query/stream` | `POST` | Same as `/query`, but streams the answer as NDJSON: `token` events, then one `done` event with the sources. Used by the frontend. |
| `/api/v1/session` | `DELETE` | Deletes the session's indexed chunks and uploaded files. |
| `/docs` | `GET` | Auto-generated interactive Swagger UI for API testing. |

Requests are rate limited per session (default `30/minute`); over the limit returns HTTP `429`.

---

## 🛠️ Local Development Setup

To run this project locally, follow these steps:

### 1. Prerequisites

Ensure you have the following installed:
- Python 3.11+
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/AkshithKalvakota/Enterprise-RAG-platform.git
cd Enterprise-RAG-platform
```

### 3. Create a Virtual Environment and Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt -r frontend/requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` in the root directory and fill in your keys:

```text
GOOGLE_API_KEY=your_google_ai_studio_key
QDRANT_URL=your_qdrant_cloud_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
API_KEY=any_long_random_string
```

Without `QDRANT_URL`, the backend falls back to a local Qdrant store in `storage/qdrant`. Without `API_KEY`, authentication is disabled (local development only). Other settings (model, chunk size, rate limit, ...) are listed in `.env.example`.

### 5. Run the Application

You will need two separate terminal windows.

**Terminal 1 (Backend):**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 (Frontend):** point it at the local backend and use the same `API_KEY`:
```bash
BACKEND_URL=http://localhost:8000 API_KEY=your_api_key streamlit run frontend/app.py
```
(On Windows PowerShell: `$env:BACKEND_URL="http://localhost:8000"; $env:API_KEY="your_api_key"; streamlit run frontend/app.py`)

---

## ☁️ Cloud Deployment

This platform is production-ready and deployed across two cloud environments.

**Backend (Render)**
- **Environment:** Python 3.11.8 (Configured via `PYTHON_VERSION` environment variable to prevent C++ compilation crashes)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port 10000`
- **Environment Variables:** `GOOGLE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, `API_KEY`, `LLM_MODEL`

**Keep-alive (GitHub Actions)**
- `.github/workflows/keep-alive.yml` calls `/api/v1/health` twice a day so the free Qdrant cluster is never idle long enough to be suspended. A failed check fails the run, and GitHub emails the repository owner.

**Frontend (Streamlit Community Cloud)**
- Deployed directly from the `main` branch
- Installs `frontend/requirements.txt` (found next to `app.py`)
- **Secrets:** `API_KEY` (same value as the backend) and optionally `BACKEND_URL` (defaults to the live Render backend)

---

<div align="center">

Built with ⚡ FastAPI, 🎈 Streamlit, and 🧠 Google Gemini

</div>


 # **- AKSHITH KALVAKOTA**
