<div align="center">

# 🤖 Enterprise RAG Platform

**A production-grade Retrieval-Augmented Generation platform for multi-format enterprise document intelligence**

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-FC1C4F?style=for-the-badge)
![Google Gemini](https://img.shields.io/badge/Gemini_3.5_Flash-8E75B2?style=for-the-badge&logo=google&logoColor=white)

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
* **Advanced LLM Integration:** Powered by Google's `gemini-3.5-flash` for fast, accurate generation and `gemini-embedding-001` for semantic chunk mapping.
* **Memory-Aware Chat:** Maintains session history for multi-turn, contextual conversations.
* **Decoupled Architecture:** Clean separation of concerns between the frontend UI and the backend ingestion/inference engine.
* **Cloud Native:** Fully deployed to the cloud using Render (Backend) and Streamlit Community Cloud (Frontend).

---

## 🔄 How It Works (The Workflow)

The platform operates on a two-part asynchronous pipeline: **Ingestion** and **Retrieval/Generation**.

### 1. Document Ingestion Pipeline
1. **Upload:** A user uploads business documents via the Streamlit frontend.
2. **Transfer:** The frontend sends the files to the FastAPI backend via the `/upload` endpoint.
3. **Parsing & Chunking:** The `document_parser.py` and `chunker.py` services extract raw text from various formats (PDFs, Word docs, etc.) and split it into semantically meaningful overlapping chunks using LangChain text splitters.
4. **Embedding & Indexing:** The `vector_store.py` service sends these chunks to Google's embedding model to generate high-dimensional vectors, which are then stored securely in a Qdrant Cloud cluster.

### 2. Retrieval & Generation Pipeline
1. **Query Formulation:** The user asks a natural language question in the chat interface.
2. **Context Retrieval:** The query is vectorized, and Qdrant performs a cosine similarity search to retrieve the top-K most relevant document chunks.
3. **Prompt Injection:** The `rag_engine.py` service injects these retrieved chunks, along with the user's chat history, into a strict system prompt.
4. **LLM Synthesis:** The Google Gemini LLM synthesizes an accurate, grounded answer based *strictly* on the provided enterprise context, returning it to the user.

---

## 📂 Project Structure

```text
Enterprise-RAG-platform/
│
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   └── routes.py               # API endpoint definitions (/upload, /index, /query)
│
├── database/
│   └── vector_store.py         # Vector database connection and indexing
│
├── frontend/
│   └── app.py                  # Streamlit web interface
│
├── services/
│   ├── chunker.py               # Semantic text chunking logic
│   ├── document_parser.py       # Multi-format document loading and parsing
│   └── rag_engine.py            # LLM integration and retrieval chain logic
│
├── utils/
│   └── logger.py                # Application-wide logging configuration
│
├── .env.example                 # Example environment variable file
├── .gitignore
├── README.md
├── requirements.txt              # Deployment dependencies
└── runtime.txt                   # Runtime environment configuration
```

---

## 🔌 API Endpoints

The FastAPI backend exposes the following core endpoints:

| Endpoint | Method | Description |
|:--|:--:|:--|
| `/api/v1/upload` | `POST` | Accepts multi-part form data and saves documents to the server. |
| `/api/v1/index/{filename}` | `POST` | Parses a saved document, chunks the text, and stores embeddings in Qdrant. |
| `/api/v1/query` | `POST` | Accepts a user question and chat history, performs RAG, and returns the AI response. |
| `/docs` | `GET` | Auto-generated interactive Swagger UI for API testing. |

---

## 🛠️ Local Development Setup

To run this project locally, follow these steps:

### 1. Prerequisites

Ensure you have the following installed:
- Python 3.11+
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/YourUsername/Enterprise-RAG-platform.git
cd Enterprise-RAG-platform
```

### 3. Create a Virtual Environment and Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your API keys:

```text
GEMINI_API_KEY="your_google_ai_studio_key"
QDRANT_URL="your_qdrant_cloud_cluster_url"
QDRANT_API_KEY="your_qdrant_api_key"
```

### 5. Run the Application

You will need two separate terminal windows.

**Terminal 1 (Backend):**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
streamlit run frontend/app.py
```

---

## ☁️ Cloud Deployment

This platform is production-ready and deployed across two cloud environments.

**Backend (Render)**
- **Environment:** Python 3.11.8 (Configured via `PYTHON_VERSION` environment variable to prevent C++ compilation crashes)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port 10000`

**Frontend (Streamlit Community Cloud)**
- Deployed directly from the `main` branch
- Communicates with the live Render backend via the injected `FASTAPI_URL` variable

---

<div align="center">

Built with ⚡ FastAPI, 🎈 Streamlit, and 🧠 Google Gemini

</div>


 # **- AKSHITH KALVAKOTA**
