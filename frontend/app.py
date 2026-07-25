import streamlit as st
import requests
import time

# Backend API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Configure the Streamlit page
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    # UI Improvement: Full-width button
    if st.button("Process & Index Documents", type="primary", use_container_width=True):
        if uploaded_files:
            # UI Improvement: Dynamic Progress Bar
            progress_bar = st.progress(0, text="Starting ingestion pipeline...")
            
            files_to_upload = [
                ("files", (file.name, file.getvalue(), file.type)) 
                for file in uploaded_files
            ]
            
            try:
                progress_bar.progress(25, text="Uploading files to server...")
                upload_response = requests.post(f"{API_BASE_URL}/upload", files=files_to_upload)
                
                if upload_response.status_code == 200:
                    saved_filenames = upload_response.json().get("saved_files", [])
                    total_files = len(saved_filenames)
                    index_success_count = 0
                    
                    for i, filename in enumerate(saved_filenames):
                        # Update progress mathematically based on file count
                        current_progress = 25 + int(((i + 1) / total_files) * 75)
                        progress_bar.progress(current_progress, text=f"Indexing {filename} into vector database...")
                        
                        index_response = requests.post(f"{API_BASE_URL}/index/{filename}")
                        if index_response.status_code == 200:
                            index_success_count += 1
                            
                    progress_bar.progress(100, text="Indexing complete!")
                    time.sleep(1) # Let the user see 100% for a brief moment
                    progress_bar.empty() # Clean up the progress bar from the UI
                    
                    # UI Improvement: Sleek Toast notification
                    st.toast(f"✅ Successfully processed {index_success_count} document(s)!", icon="🎉")
                else:
                    progress_bar.empty()
                    st.error(f"Failed to upload files: {upload_response.text}")
                    
            except requests.exceptions.ConnectionError:
                progress_bar.empty()
                st.error("🔌 Cannot connect to the backend server. Is FastAPI running?")
        else:
            st.warning("⚠️ Please upload at least one document first.")
            
    st.divider()
    
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize conversation memory in Streamlit's session state
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Enterprise AI Assistant. Upload your documents in the sidebar, click **Process & Index**, and ask me anything."}
    ]

# Display all previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box at the bottom of the screen
if prompt := st.chat_input("Ask a question about your documents..."):
    # 1. Display user's prompt in the UI
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Save user's prompt to memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 3. Call the FastAPI /query endpoint with sanitized chat history
    with st.chat_message("assistant"):
        # UI Improvement: Better thinking spinner
        with st.spinner("🤖 Analyzing documents and generating response..."):
            try:
                clean_history = []
                for msg in st.session_state.messages[:-1]:
                    clean_history.append({
                        "role": msg["role"],
                        "content": str(msg["content"])
                    })

                payload = {
                    "question": prompt,
                    "chat_history": clean_history,
                    "top_k": 5
                }
                
                response = requests.post(f"{API_BASE_URL}/query", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    ai_answer = data.get("answer", "No answer returned.")
                    
                    st.markdown(ai_answer)
                                
                    # 4. Save assistant response to session state memory
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": ai_answer
                    })
                else:
                    error_msg = response.json().get("detail", "Unknown server error")
                    st.error(f"🚨 Error from backend: {error_msg}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to the backend server. Make sure FastAPI is running.")