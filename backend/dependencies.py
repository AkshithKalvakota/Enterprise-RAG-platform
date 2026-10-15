from functools import lru_cache

from database.vector_store import VectorDB
from services.rag_engine import RAGEngine


# Built on first use and then reused by every request, instead of
# reconnecting to Qdrant and Gemini on each call.
@lru_cache(maxsize=1)
def get_vector_db() -> VectorDB:
    return VectorDB()


@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    return RAGEngine(get_vector_db())
