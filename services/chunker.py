from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import config

class TextChunker:
    """
    A service class responsible for breaking down large blocks of text
    into smaller, overlapping chunks for accurate AI retrieval.
    """

    @staticmethod
    def _splitter() -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

    @staticmethod
    def split_text(text: str) -> list:
        """
        Splits raw text into overlapping chunks (sizes come from config).
        Returns a list of LangChain Document objects.
        """
        return TextChunker._splitter().create_documents([text])

    @staticmethod
    def split_segments(segments: list, base_metadata: dict) -> list:
        """
        Splits parser segments into chunks. Each chunk carries base_metadata
        (source, session_id, ...) plus its segment's location (page, slide or sheet),
        which is what lets answers cite where they came from.
        """
        texts = [segment["text"] for segment in segments]
        metadatas = [{**base_metadata, **segment["location"]} for segment in segments]
        chunks = TextChunker._splitter().create_documents(texts, metadatas=metadatas)
        return [chunk for chunk in chunks if chunk.page_content.strip()]
