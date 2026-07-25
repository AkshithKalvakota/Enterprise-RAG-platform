import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from services.document_parser import DocumentParser
from services.chunker import TextChunker
from database.vector_store import VectorDB
from services.rag_engine import RAGEngine

router = APIRouter()

# Define the request payload models
class QueryRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, Any]] = []
    top_k: int = 5

@router.get("/status", tags=["System"])
async def get_status():
    return {"status": "ok", "message": "API is live."}

@router.post("/upload", tags=["Document Processing"])
async def upload_files(files: List[UploadFile] = File(...)):
    saved_files = []
    upload_dir = "storage/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        saved_files.append(file.filename)
        
    return {"message": "Files uploaded successfully", "saved_files": saved_files}

@router.post("/parse/{filename}", tags=["Document Processing"])
async def parse_document(filename: str):
    file_path = os.path.join("storage/uploads", filename)
    try:
        return DocumentParser.parse(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing document: {str(e)}")

@router.post("/chunk/{filename}", tags=["Document Processing"])
async def chunk_document(filename: str):
    file_path = os.path.join("storage/uploads", filename)
    try:
        parsed_data = DocumentParser.parse(file_path)
        chunks = TextChunker.split_text(parsed_data["text"])
        return {
            "message": "Document chunked successfully", 
            "total_chunks": len(chunks),
            "preview": [chunk.page_content for chunk in chunks[:2]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chunking document: {str(e)}")

@router.post("/index/{filename}", tags=["AI Pipeline"])
async def index_document(filename: str):
    file_path = os.path.join("storage/uploads", filename)
    try:
        # Parse and chunk the text
        parsed_data = DocumentParser.parse(file_path)
        chunks = TextChunker.split_text(parsed_data["text"])
        
        # Initialize VectorDB and save vectors
        db = VectorDB()
        
        # Handle standard Langchain insertion methods robustly
        if hasattr(db, 'add_documents'):
            db.add_documents(chunks)
        elif hasattr(db, 'index_documents'):
            db.index_documents(chunks)
        elif hasattr(db, 'db') and hasattr(db.db, 'add_documents'):
            db.db.add_documents(chunks)
            
        return {"message": f"Successfully indexed {filename} into vector database."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")

@router.post("/query", tags=["AI Pipeline"])
async def query_knowledge_base(payload: QueryRequest):
    try:
        # THE FIX: Properly initialize the class, then call .query()
        engine = RAGEngine()
        answer = engine.query(question=payload.question, chat_history=payload.chat_history)
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying knowledge base: {str(e)}")