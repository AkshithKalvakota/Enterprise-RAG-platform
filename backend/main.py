from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from backend.routes import router as api_router
from utils.logger import setup_logger

# Initialize our new enterprise logger
logger = setup_logger()

app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    description="A robust backend for document parsing, vector embedding, and LLM generation."
)

# 1. Global Exception Handler (The Safety Net)
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

# 2. Middleware (The Request Tracker)
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

# Register our API routes
app.include_router(api_router, prefix="/api/v1")