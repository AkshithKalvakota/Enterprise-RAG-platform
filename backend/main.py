import truststore

# Use the operating system's certificate store for HTTPS. Without this, local runs
# fail on PCs whose antivirus scans HTTPS (e.g. Avast) with CERTIFICATE_VERIFY_FAILED.
truststore.inject_into_ssl()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from backend.routes import public_router, router as api_router
from backend.dependencies import get_vector_db
from utils import config
from utils.logger import setup_logger

# Load environment variables
load_dotenv()

# Initialize our new enterprise logger
logger = setup_logger()

app = FastAPI(
    title="Enterprise RAG API",
    version="2.0.0",
    description="A robust backend for document parsing, vector embedding, and LLM generation."
)

# ==========================================
# Startup: build the shared Qdrant + Gemini clients once
# ==========================================
@app.on_event("startup")
def warm_up_clients():
    if not config.API_KEY:
        logger.warning("API_KEY is not set: the API is open to anyone. Set it in production.")
    try:
        get_vector_db()
        logger.info("Successfully connected to Qdrant.")
    except Exception as e:
        logger.error(f"Warning: Could not connect to Qdrant on startup. {e}")


# ==========================================
# 1. Global Exception Handler (The Safety Net)
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catches ALL unhandled exceptions in the application to prevent server crashes.
    Logs the error securely on the backend and returns a safe JSON response.
    """
    logger.error(f"CRITICAL ERROR on {request.method} {request.url.path} - Details: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Our engineering team has been notified."}
    )


# ==========================================
# 2. Middleware (CORS, Request Tracker)
# Rate limiting is a route dependency (see backend/security.py)
# ==========================================
# Only listed browser origins may call the API from JavaScript
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "X-Session-ID", "Content-Type"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Intercepts every HTTP request, logs the incoming action,
    processes the request, and logs the final status code.
    """
    logger.info(f"Incoming Request: {request.method} {request.url.path}")

    # Process the request
    response = await call_next(request)

    logger.info(f"Completed Request: {request.method} {request.url.path} with status code {response.status_code}")
    return response


# ==========================================
# Register our API routes
# ==========================================
# The old global DELETE /reset is gone: it let anyone wipe every user's documents.
# Each user now clears only their own documents with DELETE /api/v1/session.
app.include_router(public_router, prefix="/api/v1")
app.include_router(api_router, prefix="/api/v1")
