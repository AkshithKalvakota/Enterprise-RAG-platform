import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from database.vector_store import VectorDB

# Ensure environment variables are loaded
load_dotenv()

class RAGEngine:
    """
    A service class that connects our vector database to the Gemini LLM
    and manages conversation history to answer user queries accurately.
    """
    def __init__(self):
        # We are using the bleeding edge gemini-3.5-flash
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0.0 # Keep temperature at 0 for factual, grounded answers
        )
        # Initialize our connection to Qdrant
        self.db = VectorDB()

    def query(self, question: str, chat_history: list = None) -> str:
        """
        Takes a user question and past chat history, retrieves relevant documents,
        and generates a clean, direct answer using Gemini.
        """
        if chat_history is None:
            chat_history = []

        # 1. Retrieve the most relevant chunks from Qdrant (ULTRA-ROBUST SEARCH)
        docs = None
        
        # Direct method checks
        if hasattr(self.db, 'similarity_search'):
            docs = self.db.similarity_search(question)
        elif hasattr(self.db, 'search'):
            docs = self.db.search(question)
        else:
            # Hunt for the internal LangChain store inside VectorDB
            for prop_name in ['store', 'qdrant', 'vector_store', 'db', 'vstore', 'collection']:
                inner_store = getattr(self.db, prop_name, None)
                if inner_store and hasattr(inner_store, 'similarity_search'):
                    docs = inner_store.similarity_search(question)
                    break
                    
        if docs is None:
            raise AttributeError("CRITICAL: Could not find the Qdrant similarity_search method inside VectorDB. Your database class might be missing the store initialization.")
        
        # Combine the text of all retrieved chunks into one large context string
        context_text = "\n\n".join([doc.page_content for doc in docs])

        # 2. Strict System Prompt (Forces direct answers without conversational filler)
        system_prompt = f"""You are a precise enterprise assistant.
        Use ONLY the provided context to answer the user's question.
        Answer directly and concisely. 
        Do NOT use introductory phrases like 'Based on the provided document'.
        Do NOT add conversational filler.
        Just provide the precise answer.
        If the answer is not in the context, say "I cannot answer this based on the provided documents."
        
        Context:
        {context_text}"""

        # 3. Format the chat history for LangChain
        formatted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))

        # 4. Construct the final prompt with history
        messages = [SystemMessage(content=system_prompt)] + formatted_history + [HumanMessage(content=question)]

        # 5. Call Gemini
        response = self.llm.invoke(messages)
        
        # 6. Clean the Output
        raw_answer = response.content
        
        if isinstance(raw_answer, list):
            clean_answer = raw_answer[0].get("text", "")
        else:
            clean_answer = str(raw_answer)
            
        return clean_answer