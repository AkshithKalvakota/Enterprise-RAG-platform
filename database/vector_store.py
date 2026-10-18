import os
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from utils import config

# gemini-embedding-001 returns 3072-dimensional vectors by default
EMBEDDING_DIM = 3072

# LangChain stores each chunk's metadata under the "metadata" payload key
SESSION_FIELD = "metadata.session_id"


class VectorDB:
    """
    Manages vector embeddings and connection to Qdrant (supports both local and cloud).
    Create it once and reuse it: building the clients is slow.
    Every chunk is tagged with a session_id, and every search/delete is scoped to one session.
    """
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.collection_name = config.COLLECTION_NAME

        # Check if Qdrant Cloud credentials are provided in environment variables
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            # Connect to Qdrant Cloud
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                timeout=60,
            )
        else:
            # Fallback to local storage for development
            os.makedirs("storage/qdrant", exist_ok=True)
            self.client = QdrantClient(path="storage/qdrant")

        try:
            self._ensure_collection()

            self.store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
        except Exception:
            # Release the client so a retry can reconnect (local storage is single-client)
            self.client.close()
            raise

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
        # Index the session field so filtered searches stay fast (no-op if it exists;
        # local mode ignores payload indexes)
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name=SESSION_FIELD,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def _session_filter(session_id: str) -> Filter:
        return Filter(must=[FieldCondition(key=SESSION_FIELD, match=MatchValue(value=session_id))])

    def add_documents(self, chunks: list):
        return self.store.add_documents(chunks)

    def search(self, query: str, session_id: str, top_k: int = config.DEFAULT_TOP_K) -> list:
        return self.store.similarity_search(query, k=top_k, filter=self._session_filter(session_id))

    def has_documents(self, session_id: str) -> bool:
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=self._session_filter(session_id),
            exact=False,
        ).count > 0

    def delete_session(self, session_id: str):
        """Deletes every chunk belonging to one session; other sessions are untouched."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=self._session_filter(session_id)),
            wait=True,
        )
