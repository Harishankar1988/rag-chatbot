import streamlit as st
import requests
import json
from typing import List, Dict, Any
import time

# Configuration
BACKEND_URL = "http://localhost:5001"

class StreamlitUI:
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="PDF Chatbot RAG",
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = None
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'current_batch_id' not in st.session_state:
            st.session_state.current_batch_id = None
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = {}
    
    def check_backend_health(self) -> bool:
        """Check if backend is healthy"""
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def upload_files(self, uploaded_files) -> str:
        """Upload files to backend and return batch ID"""
        if not uploaded_files:
            return None
        
        try:
            files = [('files', file) for file in uploaded_files]
            response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.current_batch_id = result.get('batch_id')
                st.success(f"✅ Files queued for processing (Batch: {result.get('batch_id')})")
                return result.get('batch_id')
            else:
                error_msg = response.json().get('error', 'Unknown error')
                st.error(f"❌ Error uploading files: {error_msg}")
                return None
                
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
            return None
    
    def get_processing_status(self, batch_id: str) -> Dict[str, Any]:
        """Get processing status for a batch"""
        try:
            response = requests.get(f"{BACKEND_URL}/processing-status/{batch_id}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Status not available"}
        except:
            return {"error": "Connection error"}
    
    def monitor_processing(self, batch_id: str):
        """Monitor processing progress"""
        if not batch_id:
            return
        
        status = self.get_processing_status(batch_id)
        st.session_state.processing_status = status
        
        if status.get("status") in ["completed", "completed_with_errors", "error"]:
            st.session_state.processing = False
            st.session_state.current_batch_id = None
            
            if status.get("status") == "completed":
                st.success("✅ All files processed successfully!")
                # Update uploaded files list
                uploaded_files = status.get("uploaded_files", [])
                st.session_state.uploaded_files.extend(uploaded_files)
            elif status.get("status") == "completed_with_errors":
                st.warning("⚠️ Processing completed with some errors")
                uploaded_files = status.get("uploaded_files", [])
                st.session_state.uploaded_files.extend(uploaded_files)
                for error in status.get("errors", []):
                    st.error(f"❌ {error}")
            else:
                st.error("❌ Processing failed")
    
    def send_message(self, question: str) -> Dict[str, Any]:
        """Send message to backend"""
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={"question": question},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = response.json().get('error', 'Unknown error')
                return {"error": error_msg, "answer": f"Error: {error_msg}"}
                
        except Exception as e:
            return {"error": str(e), "answer": f"Connection error: {str(e)}"}
    
    def clear_all_data(self) -> bool:
        """Clear all data from backend"""
        try:
            response = requests.post(f"{BACKEND_URL}/clear", timeout=10)
            if response.status_code == 200:
                st.session_state.messages = []
                st.session_state.uploaded_files = []
                st.success("✅ All data cleared successfully")
                return True
            else:
                st.error(f"❌ Error clearing data: {response.json().get('error', 'Unknown error')}")
                return False
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            response = requests.get(f"{BACKEND_URL}/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {}
        except:
            return {}
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.title("📚 PDF Chatbot RAG")
            
            # Health check
            if self.check_backend_health():
                st.success("🟢 Backend Connected")
            else:
                st.error("🔴 Backend Disconnected")
                st.info("Please start the backend server:\n```bash\ncd backend\npython app.py\n```")
                return False
            
            if st.session_state.uploaded_files is None:
                try:
                    resp = requests.get(f"{BACKEND_URL}/files", timeout=5)
                    if resp.status_code == 200:
                        st.session_state.uploaded_files = resp.json().get("files", [])
                    else:
                        st.session_state.uploaded_files = []
                except:
                    st.session_state.uploaded_files = []

            # File upload section
            st.subheader("📁 Upload PDFs")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files to add to the knowledge base"
            )
            
            if st.button("📤 Upload Files", type="primary", disabled=st.session_state.processing):
                if uploaded_files:
                    batch_id = self.upload_files(uploaded_files)
                    if batch_id:
                        st.session_state.processing = True
                        st.rerun()
                else:
                    st.warning("Please select files to upload")
            
            # Show processing status
            if st.session_state.processing and st.session_state.current_batch_id:
                st.subheader("⏳ Processing Status")
                self.monitor_processing(st.session_state.current_batch_id)
                
                status = st.session_state.processing_status
                if status.get("status") == "processing":
                    # Progress bar
                    progress = status.get("progress", 0) / 100
                    st.progress(progress)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Files Processed", f"{status.get('processed_files', 0)}/{status.get('total_files', 0)}")
                    with col2:
                        if status.get("estimated_remaining_time"):
                            st.metric("Est. Time", f"{status.get('estimated_remaining_time', 0)}s")
                    
                    st.info(f"📝 {status.get('message', 'Processing...')}")
                    
                    # Auto-refresh
                    time.sleep(2)
                    st.rerun()
                elif status.get("status") in ["queued"]:
                    st.info("📋 Files are queued for processing...")
                    time.sleep(2)
                    st.rerun()
            
            # Display uploaded files
            if st.session_state.uploaded_files:
                st.subheader("📋 Uploaded Files")
                for file_info in st.session_state.uploaded_files:
                    with st.expander(f"📄 {file_info.get('filename', 'Unknown')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Pages:** {file_info.get('info', {}).get('num_pages', 'Unknown')}")
                            st.write(f"**Size:** {file_info.get('info', {}).get('file_size', 0):,} bytes")
                        with col2:
                            st.write(f"**Chunks:** {file_info.get('chunks_created', 0)}")
                            st.write(f"**Title:** {file_info.get('info', {}).get('title', 'Unknown')}")
            
            # System stats
            if st.button("🔄 Refresh Stats"):
                st.session_state.system_stats = self.get_system_stats()
                st.rerun()
            
            if not st.session_state.system_stats:
                st.session_state.system_stats = self.get_system_stats()
            
            if st.session_state.system_stats:
                st.subheader("📊 System Stats")
                stats = st.session_state.system_stats
                vector_stats = stats.get('vector_store', {})
                
                st.metric("Documents", vector_stats.get('document_count', 0))
                st.metric("Chat Messages", stats.get('chat_history_length', 0))
                st.metric("Embedding Model", vector_stats.get('embedding_model', 'Unknown')[:15] + "...")
            
            # Clear data button
            if st.button("🗑️ Clear All Data", type="secondary"):
                if st.confirm("Are you sure you want to clear all uploaded files and chat history?"):
                    self.clear_all_data()
                    st.rerun()
            
            return True
    
    def render_chat_interface(self):
        """Render main chat interface"""
        st.title("💬 Chat with your PDFs")
        
        if not st.session_state.uploaded_files or len(st.session_state.uploaded_files) == 0:
            st.info("👆 Please upload some PDF files in the sidebar to start chatting!")
            return
        
        # Chat messages
        chat_container = st.container()
        with chat_container:
            # Display chat history
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    with st.chat_message("user"):
                        st.markdown(msg['content'])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg['content'])

        
        # Chat input
        if prompt := st.chat_input("Ask a question about your PDFs..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message immediately
            #message(prompt, is_user=True, key=f"user_latest")
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.spinner("Thinking..."):
                response = self.send_message(prompt)
                
                if 'error' in response:
                    bot_message = response.get('answer', 'Sorry, an error occurred.')
                else:
                    bot_message = response.get('answer', 'Sorry, I could not generate a response.')
                    
                    # Display sources if available
                    sources = response.get('sources', [])
                    if sources:
                        with st.expander(f"📖 Sources ({len(sources)} documents)"):
                            # Show search metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if response.get('query_expansion_used'):
                                    st.info("🔍 Query rewriting used")
                                    rewritten = response.get('rewritten_query')
                                    if rewritten:
                                        st.caption(f"Rewritten: {rewritten[:100]}...")
                            
                            with col2:
                                if response.get('reranking_used'):
                                    st.info("🔄 Re-ranking applied")
                                    st.caption(f"Initial: {response.get('initial_docs_retrieved', 0)} → Final: {response.get('final_docs_after_reranking', 0)}")
                            
                            with col3:
                                confidence = response.get('confidence_score', 0)
                                if confidence > 0:
                                    color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                                    st.metric(f"🎯 Confidence", f"{confidence:.2f}", delta=None)
                            
                            st.divider()
                            
                            for i, source in enumerate(sources):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**📄 Document:** {source.get('source', 'Unknown')}")
                                    pages = source.get('pages', [])
                                    if pages:
                                        if len(pages) == 1:
                                            st.write(f"**📑 Page:** {pages[0]}")
                                        else:
                                            st.write(f"**📑 Pages:** {min(pages)}-{max(pages)}")
                                    else:
                                        page = source.get('primary_page')
                                        if page:
                                            st.write(f"**📑 Page:** {page}")
                                    
                                    st.write(f"**🏷️ Type:** {source.get('chunk_type', 'paragraph').title()}")
                                    st.write(f"**🔍 Relevance:** {source.get('score', 0):.3f}")
                                    st.write(f"**📊 Sentences:** {source.get('sentence_count', 0)}")
                                    
                                    # Show search type scores
                                    if source.get('search_type') == 'hybrid':
                                        st.write(f"**🔬 Hybrid Score:** {source.get('hybrid_score', 0):.3f}")
                                        st.write(f"**🧮 Vector Score:** {source.get('vector_score', 0):.3f}")
                                        st.write(f"**🔤 Keyword Score:** {source.get('keyword_score', 0):.3f}")
                                    
                                    # Show re-ranking info
                                    if source.get('rerank_position'):
                                        st.write(f"**🏆 Re-rank:** #{source.get('rerank_position')}")
                                
                                with col2:
                                    st.write(f"**Rank:** #{source.get('relevance_rank', i+1)}")
                                    st.write(f"**Chunk:** {source.get('chunk_index', 0) + 1}")
                                
                                # Preview
                                preview = source.get('preview', '')
                                if preview:
                                    st.code(preview[:300] + "..." if len(preview) > 300 else preview, 
                                           language="text")
                                
                                st.divider()
                
                # Add bot message to history
                st.session_state.messages.append({"role": "assistant", "content": bot_message})
                
                # Display bot message
                #message(bot_message, is_user=False, key=f"bot_latest")
                with st.chat_message("assistant"):
                    st.markdown(bot_message)
                
                # Rerun to update the interface
                time.sleep(0.1)
                st.rerun()
    
    def render_main_page(self):
        """Render the main application page"""
        # Render sidebar
        sidebar_ok = self.render_sidebar()
        
        if sidebar_ok:
            # Render main content
            self.render_chat_interface()
        else:
            st.error("Cannot connect to backend. Please check the server status.")
    
    def run(self):
        """Run the Streamlit application"""
        self.render_main_page()

# Main execution
if __name__ == "__main__":
    app = StreamlitUI()
    app.run()
