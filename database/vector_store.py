import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

class VectorDB:
    """
    Manages vector embeddings and connection to Qdrant (supports both local and cloud).
    """
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.collection_name = "enterprise_rag_docs"

        # Check if Qdrant Cloud credentials are provided in environment variables
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            # Connect to Qdrant Cloud
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            # Fallback to local storage for development
            os.makedirs("storage/qdrant", exist_ok=True)
            self.client = QdrantClient(path="storage/qdrant")

        # Ensure collection exists (3072 dimensions for gemini-embedding-001)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def add_documents(self, chunks: list):
        return self.store.add_documents(chunks)

    def search(self, query: str, top_k: int = 4) -> list:
        return self.store.similarity_search(query, k=top_k)