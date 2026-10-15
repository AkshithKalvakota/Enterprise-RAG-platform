import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, Filter, FilterSelector, VectorParams

load_dotenv()

# gemini-embedding-001 returns 3072-dimensional vectors by default
EMBEDDING_DIM = 3072


class VectorDB:
    """
    Manages vector embeddings and connection to Qdrant (supports both local and cloud).
    Create it once and reuse it: building the clients is slow.
    """
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        # Single source of truth for the collection name (used by /reset too)
        self.collection_name = os.getenv("COLLECTION_NAME", "enterprise_rag_docs")

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

    def add_documents(self, chunks: list):
        return self.store.add_documents(chunks)

    def search(self, query: str, top_k: int = 5) -> list:
        return self.store.similarity_search(query, k=top_k)

    def reset(self):
        """Deletes every stored chunk but keeps the collection itself."""
        # An empty filter matches every point. Deleting points (instead of dropping the
        # collection) also works in local mode, where Windows keeps the old files locked.
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=Filter()),
            wait=True,
        )
