import os
from dotenv import load_dotenv

load_dotenv()

# All tunable settings live here and can be overridden with environment variables
# (locally in .env, on Render under Environment).

# --- Models ---
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
LLM_THINKING_LEVEL = os.getenv("LLM_THINKING_LEVEL", "low")
# Used automatically when LLM_MODEL fails (overloaded, quota exceeded, timeout).
# A different model has its own quota and load. Set to "" to disable.
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemini-3.1-flash-lite")
# Each model gets this many attempts in total (1 = no retry) and this many seconds per attempt.
# The Google SDK default is 6 attempts with backoff, which made users wait ~65s before an error.
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "2"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
# Chunks sent per embedding request, and attempts per request when Gemini is busy
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "50"))
EMBED_MAX_ATTEMPTS = int(os.getenv("EMBED_MAX_ATTEMPTS", "4"))

# --- Vector store ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_rag_docs")

# --- Chunking & retrieval ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "20"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))

# --- Cleanup ---
# Documents older than this are deleted automatically (abandoned browser sessions)
SESSION_TTL_HOURS = float(os.getenv("SESSION_TTL_HOURS", "24"))
CLEANUP_INTERVAL_MINUTES = float(os.getenv("CLEANUP_INTERVAL_MINUTES", "60"))

# --- Uploads ---
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "storage/uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md"}

# --- Security ---
# When API_KEY is set, every /api/v1 request must send it in the X-API-Key header.
# Leave it empty only for local development.
# Stray spaces/quotes from pasting into a hosting dashboard are ignored.
API_KEY = os.getenv("API_KEY", "").strip().strip("\"'")
# Requests allowed per session, e.g. "30/minute"
RATE_LIMIT = os.getenv("RATE_LIMIT", "30/minute")
# Comma-separated browser origins allowed to call the API (Streamlit calls it server-side,
# so this only matters for a future JavaScript frontend)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
