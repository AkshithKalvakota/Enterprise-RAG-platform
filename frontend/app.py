import truststore

# Use the operating system's certificate store for HTTPS (fixes antivirus HTTPS scanning locally)
truststore.inject_into_ssl()

import os
import time
import uuid

import requests
import streamlit as st


def _setting(name: str, default: str = "") -> str:
    """Reads a setting from Streamlit secrets (Streamlit Cloud), then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # No secrets file (e.g. running locally without .streamlit/secrets.toml)
    return os.getenv(name, default)


# Backend API Configuration
BACKEND_URL = _setting("BACKEND_URL", "https://enterprise-rag-assistant-6n3h.onrender.com").rstrip("/")
API_BASE_URL = f"{BACKEND_URL}/api/v1"
API_KEY = _setting("API_KEY").strip().strip("\"'")

# Request timeouts in seconds (Render free tier can take ~60s to wake up)
WAKE_TIMEOUT = 90
UPLOAD_TIMEOUT = 120
INDEX_TIMEOUT = 300
QUERY_TIMEOUT = 120
SESSION_TIMEOUT = 60

WELCOME_MESSAGE = {"role": "assistant", "content": "Hello! I am your Enterprise AI Assistant. Upload your documents in the sidebar, click **Process & Index**, and ask me anything."}
CONNECTION_ERRORS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

# Configure the Streamlit page
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Each browser session gets its own ID, so users only ever see and clear their own documents
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


def api_headers() -> dict:
    headers = {"X-Session-ID": st.session_state.session_id}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def error_detail(response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def wake_backend() -> bool:
    """Pings the backend, giving a sleeping Render instance time to start."""
    try:
        return requests.get(f"{API_BASE_URL}/status", timeout=WAKE_TIMEOUT).status_code == 200
    except CONNECTION_ERRORS:
        return False


def clear_session_documents():
    response = requests.delete(f"{API_BASE_URL}/session", headers=api_headers(), timeout=SESSION_TIMEOUT)
    response.raise_for_status()


def format_sources(sources: list) -> str:
    labels = [f"{s['source']} ({s['location']})" if s.get("location") else s["source"] for s in sources]
    return "📄 Sources: " + " · ".join(labels)


# Custom CSS for a polished look
st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.title("🤖 Enterprise AI Assistant")
st.markdown("Upload your business documents and ask natural language questions. Powered by **FastAPI, Qdrant, and Google Gemini**.")

# Sidebar Configuration
with st.sidebar:
    st.header("📂 Document Management")

    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, PPTX, XLSX, CSV, TXT)",
        type=["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Process & Index Documents", type="primary", use_container_width=True):
        if uploaded_files:
            progress_bar = st.progress(0, text="Waking up server (can take up to a minute)...")

            if not wake_backend():
                progress_bar.empty()
                st.error("🔌 Cannot connect to the backend server. Please try again in a minute.")
                st.stop()

            try:
                # Replace this session's previous documents (other users are not affected)
                progress_bar.progress(10, text="Clearing your previous documents...")
                clear_session_documents()

                files_to_upload = [
                    ("files", (file.name, file.getvalue(), file.type))
                    for file in uploaded_files
                ]

                progress_bar.progress(25, text="Uploading files to server...")
                upload_response = requests.post(
                    f"{API_BASE_URL}/upload", files=files_to_upload, headers=api_headers(), timeout=UPLOAD_TIMEOUT
                )

                if upload_response.status_code == 200:
                    upload_data = upload_response.json()
                    saved_filenames = upload_data.get("saved_files", [])
                    failures = [f"{r['file']}: {r['reason']}" for r in upload_data.get("rejected_files", [])]
                    total_files = len(saved_filenames)
                    index_success_count = 0

                    for i, filename in enumerate(saved_filenames):
                        current_progress = 25 + int(((i + 1) / total_files) * 75)
                        progress_bar.progress(current_progress, text=f"Indexing {filename} into vector database...")

                        index_response = requests.post(
                            f"{API_BASE_URL}/index/{filename}", headers=api_headers(), timeout=INDEX_TIMEOUT
                        )
                        if index_response.status_code == 200:
                            index_success_count += 1
                        else:
                            failures.append(f"{filename}: {error_detail(index_response)}")

                    progress_bar.progress(100, text="Indexing complete!")
                    time.sleep(1)
                    progress_bar.empty()

                    if failures:
                        # Kept in session state so the warning survives the rerun below
                        st.session_state.index_failures = failures

                    # Clear chat history automatically since it's a new document
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"Successfully loaded {index_success_count} document(s). Ask me anything!"}
                    ]

                    st.toast(f"✅ Successfully processed {index_success_count} document(s)!", icon="🎉")
                    time.sleep(1)
                    st.rerun()
                else:
                    progress_bar.empty()
                    st.error(f"Failed to upload files: {error_detail(upload_response)}")

            except CONNECTION_ERRORS:
                progress_bar.empty()
                st.error("🔌 Lost connection to the backend server. Please try again.")
            except requests.exceptions.HTTPError as e:
                progress_bar.empty()
                st.error(f"🚨 Error from backend: {error_detail(e.response)}")
        else:
            st.warning("⚠️ Please upload at least one document first.")

    if st.session_state.get("index_failures"):
        st.warning("Some files could not be processed:\n\n" + "\n".join(f"- {f}" for f in st.session_state.index_failures))

    st.divider()

    st.header("⚙️ Settings")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = [WELCOME_MESSAGE]
        st.rerun()

    if st.button("🗑️ Remove My Documents", use_container_width=True):
        try:
            clear_session_documents()
            st.session_state.messages = [WELCOME_MESSAGE]
            st.session_state.index_failures = []
            st.toast("Your documents were removed.", icon="🗑️")
            time.sleep(1)
            st.rerun()
        except (*CONNECTION_ERRORS, requests.exceptions.HTTPError):
            st.error("🔌 Could not reach the backend server. Please try again.")

# Initialize conversation memory
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [WELCOME_MESSAGE]

# Display all previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(format_sources(message["sources"]))

# Chat input box
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display user's prompt in the UI
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Save user's prompt to memory without using .append()
    st.session_state.messages = st.session_state.messages + [{"role": "user", "content": prompt}]

    # 3. Call the FastAPI /query endpoint with sanitized chat history
    with st.chat_message("assistant"):
        with st.spinner("🤖 Analyzing documents and generating response..."):
            try:
                clean_history = [
                    {"role": msg["role"], "content": str(msg["content"])}
                    for msg in st.session_state.messages[:-1]
                ]

                payload = {
                    "question": prompt,
                    "chat_history": clean_history,
                    "top_k": 5
                }

                response = requests.post(
                    f"{API_BASE_URL}/query", json=payload, headers=api_headers(), timeout=QUERY_TIMEOUT
                )

                if response.status_code == 200:
                    data = response.json()
                    ai_answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])

                    st.markdown(ai_answer)
                    if sources:
                        st.caption(format_sources(sources))

                    # 4. Save assistant response to memory without using .append()
                    st.session_state.messages = st.session_state.messages + [
                        {"role": "assistant", "content": ai_answer, "sources": sources}
                    ]
                elif response.status_code == 429:
                    st.error("⏳ Too many questions in a short time. Please wait a minute and try again.")
                else:
                    st.error(f"🚨 Error from backend: {error_detail(response)}")

            except CONNECTION_ERRORS:
                st.error("🔌 Cannot connect to the backend server. It may be waking up; please try again in a minute.")
