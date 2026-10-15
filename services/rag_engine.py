import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from database.vector_store import VectorDB

# Ensure environment variables are loaded
load_dotenv()

# Only the most recent messages are sent to the LLM, so prompts don't grow forever
MAX_HISTORY_MESSAGES = 6


class RAGEngine:
    """
    A service class that connects our vector database to the Gemini LLM
    and manages conversation history to answer user queries accurately.
    """
    def __init__(self, db: VectorDB):
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-3.5-flash"),
            # Gemini 3.x thinks before answering by default; "low" cuts latency a lot
            # for document lookups. Options: minimal, low, medium, high.
            thinking_level=os.getenv("LLM_THINKING_LEVEL", "low"),
        )
        self.db = db

    def query(self, question: str, chat_history: list = None, top_k: int = 5) -> str:
        """
        Takes a user question and past chat history, retrieves relevant documents,
        and generates a clean, direct answer using Gemini.
        """
        if chat_history is None:
            chat_history = []

        # 1. Retrieve the most relevant chunks from Qdrant
        docs = self.db.search(question, top_k=top_k)

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

        # 3. Format the recent chat history for LangChain
        formatted_history = []
        for msg in chat_history[-MAX_HISTORY_MESSAGES:]:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))

        # 4. Construct the final prompt with history
        messages = [SystemMessage(content=system_prompt)] + formatted_history + [HumanMessage(content=question)]

        # 5. Call Gemini
        response = self.llm.invoke(messages)

        # 6. Clean the Output: Gemini 3 can return several content parts, keep all text parts
        raw_answer = response.content

        if isinstance(raw_answer, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_answer
            )
        return str(raw_answer)
