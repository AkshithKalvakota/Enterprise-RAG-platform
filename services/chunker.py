from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """
    A service class responsible for breaking down large blocks of text 
    into smaller, overlapping chunks for accurate AI retrieval.
    """
    
    @staticmethod
    def split_text(text: str) -> list:
        """
        Splits raw text into 1000-character chunks with a 200-character overlap.
        Returns a list of LangChain Document objects.
        """
        # Define our chunking strategy
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # create_documents returns a list of Document objects 
        # which have .page_content and .metadata attributes
        chunks = text_splitter.create_documents([text])
        
        return chunks