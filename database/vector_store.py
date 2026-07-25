import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Ensure environment variables are loaded
load_dotenv()

class VectorDB:
    """
    A service class that manages the connection to Qdrant and handles
    vector embeddings using Google's models.
    """
    def __init__(self):
        # 1. Initialize the embedding model (Massive 3072 dimensions)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        # 2. Connect to local Qdrant
        self.client = QdrantClient(path="storage/qdrant")
        self.collection_name = "enterprise_rag_docs"

        # 3. Ensure the collection exists with the correct dimensions
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )

        # 4. Bind LangChain to Qdrant
        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def add_documents(self, chunks: list):
        """Explicitly allows routes.py to add documents."""
        return self.store.add_documents(chunks)

    def search(self, query: str, top_k: int = 4) -> list:
        """Explicitly allows rag_engine.py to search for answers."""
        return self.store.similarity_search(query, k=top_k)