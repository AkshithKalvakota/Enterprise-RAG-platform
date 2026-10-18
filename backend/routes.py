import os
import shutil
import time
import json
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from services.document_parser import DocumentParser
from services.chunker import TextChunker
from utils.retry import is_model_busy_error
from backend.dependencies import get_vector_db, get_rag_engine
from backend.security import get_session_id, rate_limit, require_api_key
from utils import config

# /status stays open so the frontend can wake the server; everything else needs the API key.
# Both are rate limited per session.
public_router = APIRouter(dependencies=[Depends(rate_limit)])
router = APIRouter(dependencies=[Depends(rate_limit), Depends(require_api_key)])

BUSY_MESSAGE = "The AI model is busy right now. Please try again in a few seconds."

# Define the request payload models
class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_history: List[Dict[str, Any]] = []
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K)


def _session_dir(session_id: str) -> str:
    return os.path.join(config.UPLOAD_DIR, session_id)


def _safe_filename(filename: str) -> str:
    """Strips any directory parts so a name like '../../x' cannot escape the upload folder."""
    name = os.path.basename((filename or "").replace("\\", "/")).strip().lstrip(".")
    if not name:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    return name


def _session_file(session_id: str, filename: str) -> str:
    file_path = os.path.join(_session_dir(session_id), _safe_filename(filename))
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return file_path


@public_router.get("/status", tags=["System"])
async def get_status():
    return {"status": "ok", "message": "API is live."}

@public_router.get("/health", tags=["System"])
def get_health():
    """
    Checks that Qdrant answers. Called on a schedule by .github/workflows/keep-alive.yml:
    the request counts as cluster activity (free Qdrant clusters are suspended when idle),
    and a failure here turns the workflow red so GitHub emails the owner.
    """
    try:
        db = get_vector_db()
        db.client.count(collection_name=db.collection_name, exact=False)
        return {"status": "ok", "vector_db": "ok"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Vector database unavailable: {str(e)}")

@router.post("/upload", tags=["Document Processing"])
async def upload_files(files: List[UploadFile] = File(...), session_id: str = Depends(get_session_id)):
    saved_files = []
    rejected_files = []
    upload_dir = _session_dir(session_id)
    os.makedirs(upload_dir, exist_ok=True)
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024

    for file in files:
        try:
            filename = _safe_filename(file.filename)
        except HTTPException:
            rejected_files.append({"file": file.filename, "reason": "Invalid file name."})
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in config.ALLOWED_EXTENSIONS:
            rejected_files.append({"file": filename, "reason": f"Unsupported file type: {ext or 'none'}"})
            continue

        # Read one byte past the limit so oversized files are detected without loading all of them
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            rejected_files.append({"file": filename, "reason": f"File is larger than {config.MAX_UPLOAD_MB} MB."})
            continue

        with open(os.path.join(upload_dir, filename), "wb") as buffer:
            buffer.write(content)
        saved_files.append(filename)

    return {"message": "Files uploaded successfully", "saved_files": saved_files, "rejected_files": rejected_files}

# The routes below are plain `def`: they do blocking work (parsing, Qdrant, Gemini),
# so FastAPI runs them in a threadpool instead of freezing the event loop.

@router.post("/parse/{filename}", tags=["Document Processing"])
def parse_document(filename: str, session_id: str = Depends(get_session_id)):
    file_path = _session_file(session_id, filename)
    try:
        return DocumentParser.parse(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

@router.post("/chunk/{filename}", tags=["Document Processing"])
def chunk_document(filename: str, session_id: str = Depends(get_session_id)):
    file_path = _session_file(session_id, filename)
    try:
        parsed_data = DocumentParser.parse(file_path)
        chunks = TextChunker.split_segments(parsed_data["segments"], {"source": parsed_data["metadata"]["source"]})
        return {
            "message": "Document chunked successfully",
            "total_chunks": len(chunks),
            "preview": [chunk.page_content for chunk in chunks[:2]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking document: {str(e)}")

@router.post("/index/{filename}", tags=["AI Pipeline"])
def index_document(filename: str, session_id: str = Depends(get_session_id)):
    file_path = _session_file(session_id, filename)
    try:
        # Parse and chunk the text; every chunk is tagged with its source file, session,
        # and indexing time (used to delete abandoned sessions after SESSION_TTL_HOURS)
        parsed_data = DocumentParser.parse(file_path)
        base_metadata = {
            "source": parsed_data["metadata"]["source"],
            "session_id": session_id,
            "indexed_at": time.time(),
        }
        chunks = TextChunker.split_segments(parsed_data["segments"], base_metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

    # Scanned PDFs and empty files produce no text; say so instead of reporting success
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No text could be extracted from {filename}. If it is a scanned PDF, it needs OCR.",
        )

    try:
        # Embed and save vectors using the shared VectorDB
        get_vector_db().add_documents(chunks)
        return {"message": f"Successfully indexed {filename} into vector database.", "chunks_indexed": len(chunks)}
    except Exception as e:
        if is_model_busy_error(e):
            # Still busy after all embedding retries
            raise HTTPException(
                status_code=503,
                detail=f"The AI service is busy, so {filename} could not be indexed. Please try again in a minute.",
            )
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")

@router.post("/query", tags=["AI Pipeline"])
def query_knowledge_base(payload: QueryRequest, session_id: str = Depends(get_session_id)):
    try:
        return get_rag_engine().query(
            question=payload.question,
            session_id=session_id,
            chat_history=payload.chat_history,
            top_k=payload.top_k,
        )
    except Exception as e:
        if is_model_busy_error(e):
            # Both the primary and fallback models were busy: tell the user to retry, not a stack of API errors
            raise HTTPException(
                status_code=503,
                detail=BUSY_MESSAGE,
            )
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

@router.post("/query/stream", tags=["AI Pipeline"])
def stream_knowledge_base(payload: QueryRequest, session_id: str = Depends(get_session_id)):
    """
    Like /query, but streams the answer as it is written. The response is NDJSON
    (one JSON object per line): {"type": "token", "text"} pieces, then one
    {"type": "done", "answer", "sources"} event, or {"type": "error", "detail"} if generation fails.
    """
    try:
        # Retrieval happens here, so its errors are normal HTTP errors
        events = get_rag_engine().stream(
            question=payload.question,
            session_id=session_id,
            chat_history=payload.chat_history,
            top_k=payload.top_k,
        )
    except Exception as e:
        if is_model_busy_error(e):
            raise HTTPException(status_code=503, detail=BUSY_MESSAGE)
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")

    def ndjson():
        try:
            for event in events:
                yield json.dumps(event) + "\n"
        except Exception as e:
            # The 200 status is already sent once streaming starts, so report the failure in-band
            detail = BUSY_MESSAGE if is_model_busy_error(e) else f"Error generating answer: {str(e)}"
            yield json.dumps({"type": "error", "detail": detail}) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")

@router.delete("/session", tags=["Document Processing"])
def clear_session(session_id: str = Depends(get_session_id)):
    """Deletes this session's indexed chunks and uploaded files. Other sessions are untouched."""
    try:
        get_vector_db().delete_session(session_id)
        shutil.rmtree(_session_dir(session_id), ignore_errors=True)
        return {"message": "Session documents cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")
