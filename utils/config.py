import os
from dotenv import load_dotenv

load_dotenv()

# All tunable settings live here and can be overridden with environment variables
# (locally in .env, on Render under Environment).

# --- Models ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_THINKING_LEVEL = os.getenv("LLM_THINKING_LEVEL", "low")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# --- Vector store ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_rag_docs")

# --- Chunking & retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))

# --- Uploads ---
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "storage/uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md"}

# --- Security ---
# When API_KEY is set, every /api/v1 request must send it in the X-API-Key header.
# Leave it empty only for local development.
API_KEY = os.getenv("API_KEY", "")
# Requests allowed per session, e.g. "30/minute"
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
# Comma-separated browser origins allowed to call the API (Streamlit calls it server-side,
# so this only matters for a future JavaScript frontend)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
