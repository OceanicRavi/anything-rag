"""
Streamlit Web Interface for RAG-Anything.
Beautiful UI with proper markdown rendering for RAG responses.
"""

import streamlit as st
import asyncio
import requests
import json
from pathlib import Path
import time
import pandas as pd
from typing import Dict, List, Any

# Configure Streamlit page
st.set_page_config(
    page_title="RAG-Anything",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .stAlert > div {
        background-color: rgba(102, 126, 234, 0.1);
        border: 1px solid #667eea;
        border-radius: 10px;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffa8a8 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .query-response {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE = "http://127.0.0.1:8000"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_status" not in st.session_state:
    st.session_state.api_status = "unknown"
if "query_counter" not in st.session_state:
    st.session_state.query_counter = 0

def check_api_connection():
    """Check if API server is running."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            st.session_state.api_status = "connected"
            return True
        else:
            st.session_state.api_status = "error"
            return False
    except requests.exceptions.RequestException:
        st.session_state.api_status = "disconnected"
        return False

def get_system_status():
    """Get system status from API."""
    try:
        response = requests.get(f"{API_BASE}/status", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def query_documents(question: str, mode: str = "hybrid"):
    """Query documents via API."""
    try:
        response = requests.post(
            f"{API_BASE}/query",
            json={"question": question, "mode": mode},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"error": "Query timed out (30s). Try a simpler question."}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def upload_file(uploaded_file, force_reprocess=False):
    """Upload file via API."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"force_reprocess": force_reprocess}
        
        response = requests.post(
            f"{API_BASE}/upload",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.status_code}"}
    except Exception as e:
        return {"error": f"Upload error: {str(e)}"}

def get_documents():
    """Get list of processed documents."""
    try:
        response = requests.get(f"{API_BASE}/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

def remove_document(doc_name: str):
    """Remove a document via API."""
    try:
        response = requests.delete(f"{API_BASE}/documents/{doc_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Remove failed: {response.status_code}"}
    except Exception as e:
        return {"error": f"Remove error: {str(e)}"}

def remove_all_documents():
    """Remove all documents via API."""
    try:
        response = requests.delete(f"{API_BASE}/documents", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Remove all failed: {response.status_code}"}
    except Exception as e:
        return {"error": f"Remove all error: {str(e)}"}

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 RAG-Anything</h1>', unsafe_allow_html=True)
    st.markdown("### Multimodal Document Processing & Querying System")
    
    # Check API connection
    api_connected = check_api_connection()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # API Status
        if api_connected:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Disconnected")
            st.markdown("**Start the API server:**")
            st.code("python api_server.py", language="bash")
            st.stop()
        
        # System Status
        with st.expander("📊 System Status"):
            status = get_system_status()
            if status:
                st.metric("Processed Files", status.get("processed_files_count", 0))
                st.metric("Pending Files", status.get("pending_files_count", 0))
                st.metric("Total Documents", len(status.get("documents", [])))
            else:
                st.warning("Could not load system status")
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        if st.button("🔄 Refresh Status"):
            st.rerun()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.query_counter = 0
            st.rerun()

    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📤 Upload", "📋 Documents", "📊 Analytics"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.header("💬 Chat with Your Documents")
        
        # Query Mode Selection
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            query_mode = st.selectbox(
                "Query Mode:",
                ["hybrid", "local", "global"],
                help="Hybrid: Best overall results | Local: Document-specific | Global: Knowledge graph"
            )
        with col2:
            clear_chat = st.button("🧹 Clear Chat")
            if clear_chat:
                st.session_state.messages = []
                st.session_state.query_counter = 0
                st.rerun()
        with col3:
            if st.session_state.messages:
                expand_all = st.button("📖 Expand All")
                if expand_all:
                    # Set all queries to expanded
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            msg["expanded"] = True
                    st.rerun()
        
        # Chat input (at the top for better UX)
        if prompt := st.chat_input("Ask anything about your documents..."):
            # Increment query counter
            st.session_state.query_counter += 1
            
            # Add user message with metadata
            user_message = {
                "role": "user", 
                "content": prompt,
                "query_id": st.session_state.query_counter,
                "timestamp": time.time(),
                "expanded": True  # New queries start expanded
            }
            
            # Insert at the beginning (most recent first)
            st.session_state.messages.insert(0, user_message)
            
            # Get AI response
            with st.spinner("Thinking..."):
                start_time = time.time()
                response = query_documents(prompt, query_mode)
                end_time = time.time()
            
            if "error" in response:
                assistant_message = {
                    "role": "assistant", 
                    "content": f"**Error:** {response['error']}",
                    "query_id": st.session_state.query_counter,
                    "mode": query_mode,
                    "time": f"{round(end_time - start_time, 2)}s",
                    "timestamp": time.time(),
                    "error": True
                }
            else:
                # Store the successful response
                answer = response.get("answer", "No response received")
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "query_id": st.session_state.query_counter,
                    "mode": query_mode,
                    "time": f"{round(end_time - start_time, 2)}s",
                    "timestamp": time.time(),
                    "error": False
                }
            
            # Insert assistant response after user message
            st.session_state.messages.insert(1, assistant_message)
            st.rerun()
        
        # Display chat messages (most recent first, collapsible)
        if st.session_state.messages:
            st.markdown("### 📚 Conversation History")
            
            # Group messages by query_id and display in reverse order
            query_groups = {}
            for message in st.session_state.messages:
                query_id = message.get("query_id", 0)
                if query_id not in query_groups:
                    query_groups[query_id] = {"user": None, "assistant": None}
                query_groups[query_id][message["role"]] = message
            
            # Sort by query_id descending (most recent first)
            sorted_queries = sorted(query_groups.items(), key=lambda x: x[0], reverse=True)
            
            for query_id, messages in sorted_queries:
                user_msg = messages.get("user")
                assistant_msg = messages.get("assistant")
                
                if user_msg:
                    # Create collapsible section for each query
                    query_preview = user_msg["content"][:80] + "..." if len(user_msg["content"]) > 80 else user_msg["content"]
                    
                    # Historical queries are collapsed by default (only most recent is expanded)
                    is_expanded = (query_id == st.session_state.query_counter)
                    
                    # Create timestamp
                    import datetime
                    timestamp = datetime.datetime.fromtimestamp(user_msg.get("timestamp", time.time()))
                    time_str = timestamp.strftime("%H:%M:%S")
                    
                    # Status indicator
                    if assistant_msg:
                        status = "❌" if assistant_msg.get("error", False) else "✅"
                        response_time = assistant_msg.get("time", "")
                        mode = assistant_msg.get("mode", "")
                    else:
                        status = "⏳"
                        response_time = ""
                        mode = ""
                    
                    # Collapsible query section with modern styling
                    with st.expander(
                        f"{status} **Query #{query_id}** [{time_str}] {mode} {response_time} | {query_preview}",
                        expanded=is_expanded
                    ):
                        # User question with modern styling
                        st.markdown("**🤔 Question:**")
                        st.markdown(f"> {user_msg['content']}")
                        
                        # Assistant response
                        if assistant_msg:
                            st.markdown("**🤖 Answer:**")
                            if assistant_msg.get("error", False):
                                st.error(assistant_msg["content"])
                            else:
                                # Render markdown properly for RAG responses
                                st.markdown(assistant_msg["content"])
                            
                            # Metadata with modern layout
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.caption(f"🎯 Mode: {assistant_msg.get('mode', 'Unknown')}")
                            with col2:
                                st.caption(f"⏱️ Time: {assistant_msg.get('time', 'Unknown')}")
                            with col3:
                                # Add copy button for the response
                                if st.button("📋 Copy Response", key=f"copy_{query_id}", help="Copy response to clipboard"):
                                    st.code(assistant_msg["content"])
                        else:
                            st.warning("⏳ Processing...")
                        
                        # Action buttons with modern layout
                        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                        with col1:
                            if st.button("🔄 Retry", key=f"retry_{query_id}", help="Retry this query"):
                                # Re-run the same query
                                with st.spinner("Retrying..."):
                                    start_time = time.time()
                                    response = query_documents(user_msg["content"], query_mode)
                                    end_time = time.time()
                                
                                if "error" in response:
                                    retry_message = {
                                        "role": "assistant", 
                                        "content": f"**Retry Error:** {response['error']}",
                                        "query_id": query_id,
                                        "mode": query_mode,
                                        "time": f"{round(end_time - start_time, 2)}s",
                                        "timestamp": time.time(),
                                        "error": True
                                    }
                                else:
                                    retry_message = {
                                        "role": "assistant",
                                        "content": response.get("answer", "No response received"),
                                        "query_id": query_id,
                                        "mode": query_mode,
                                        "time": f"{round(end_time - start_time, 2)}s",
                                        "timestamp": time.time(),
                                        "error": False
                                    }
                                
                                # Update the assistant message
                                for i, msg in enumerate(st.session_state.messages):
                                    if msg.get("query_id") == query_id and msg["role"] == "assistant":
                                        st.session_state.messages[i] = retry_message
                                        break
                                st.rerun()
                        
                        with col2:
                            if st.button("🗑️ Delete", key=f"delete_{query_id}", help="Delete this query"):
                                # Remove both user and assistant messages for this query
                                st.session_state.messages = [
                                    msg for msg in st.session_state.messages 
                                    if msg.get("query_id") != query_id
                                ]
                                st.rerun()
                        
                        with col3:
                            if st.button("📤 Share", key=f"share_{query_id}", help="Share this query"):
                                share_text = f"**Question:** {user_msg['content']}\n\n**Answer:** {assistant_msg.get('content', 'No response') if assistant_msg else 'Processing...'}"
                                st.text_area("Share this conversation:", value=share_text, height=150, key=f"share_text_{query_id}")
                        
                        with col4:
                            if assistant_msg and not assistant_msg.get("error", False):
                                if st.button("🔗 Follow-up", key=f"followup_{query_id}", help="Ask a follow-up question"):
                                    # Pre-populate with a follow-up prompt
                                    followup_prompt = f"Based on your previous answer about '{user_msg['content'][:50]}...', can you tell me more about"
                                    st.text_input("Follow-up question:", value=followup_prompt, key=f"followup_input_{query_id}")
        
        else:
            # Welcome section with modern design
            st.markdown("""
            <div class="welcome-section">
                <h2>👋 Welcome to RAG-Anything!</h2>
                <p>Start by asking a question about your processed documents, or try one of the examples below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show some example queries with working buttons
            st.markdown("### 💡 Example Questions")
            
            example_queries = [
                "What documents have been processed?",
                "Summarize the main findings from my documents",
                "What financial information is available?",
                "List all the entities mentioned in the documents",
                "What are the key dates and amounts mentioned?",
                "Can you extract transaction details from my bank statement?",
                "What tables or structured data are in my documents?",
                "Show me a summary of all processed content"
            ]
            
            # Create a grid layout for example questions
            cols = st.columns(2)
            for i, example in enumerate(example_queries):
                with cols[i % 2]:
                    # Use a unique key and direct query execution
                    if st.button(f"💭 {example}", key=f"example_query_{i}", use_container_width=True):
                        # Directly execute the example query
                        st.session_state.query_counter += 1
                        
                        # Add user message
                        user_message = {
                            "role": "user", 
                            "content": example,
                            "query_id": st.session_state.query_counter,
                            "timestamp": time.time(),
                            "expanded": True
                        }
                        st.session_state.messages.insert(0, user_message)
                        
                        # Get AI response
                        start_time = time.time()
                        response = query_documents(example, query_mode)
                        end_time = time.time()
                        
                        if "error" in response:
                            assistant_message = {
                                "role": "assistant", 
                                "content": f"**Error:** {response['error']}",
                                "query_id": st.session_state.query_counter,
                                "mode": query_mode,
                                "time": f"{round(end_time - start_time, 2)}s",
                                "timestamp": time.time(),
                                "error": True
                            }
                        else:
                            assistant_message = {
                                "role": "assistant",
                                "content": response.get("answer", "No response received"),
                                "query_id": st.session_state.query_counter,
                                "mode": query_mode,
                                "time": f"{round(end_time - start_time, 2)}s",
                                "timestamp": time.time(),
                                "error": False
                            }
                        
                        st.session_state.messages.insert(1, assistant_message)
                        st.rerun()
            
            # Add tips section
            st.markdown("### 🎯 Tips for Better Results")
            tips_col1, tips_col2 = st.columns(2)
            
            with tips_col1:
                st.info("""
                **💡 Query Tips:**
                - Be specific in your questions
                - Use **hybrid mode** for best results
                - Ask about specific documents or topics
                - Try different phrasings if you don't get good results
                """)
            
            with tips_col2:
                st.info("""
                **📊 Query Modes:**
                - **Hybrid**: Best overall results (recommended)
                - **Local**: Focus on specific documents
                - **Global**: Use knowledge graph relationships
                """)
            
            # System stats in welcome
            if st.checkbox("📊 Show System Stats", value=False):
                status = get_system_status()
                if status:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 Documents", len(status.get("documents", [])))
                    with col2:
                        st.metric("💾 Processed", status.get("processed_files_count", 0))
                    with col3:
                        st.metric("⏳ Pending", status.get("pending_files_count", 0))
    
    # Tab 2: Upload Interface
    with tab2:
        st.header("📤 Upload & Process Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word docs, PowerPoint, Text files"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            force_reprocess = st.checkbox("Force reprocess if already exists")
        with col2:
            process_pending = st.button("📁 Process All Pending Files")
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Upload & Process"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {file.name}...")
                    result = upload_file(file, force_reprocess)
                    
                    if "error" in result:
                        st.error(f"❌ {file.name}: {result['error']}")
                    else:
                        st.success(f"✅ {file.name}: {result.get('message', 'Uploaded successfully')}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Upload complete!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
        
        if process_pending:
            with st.spinner("Processing pending files..."):
                try:
                    response = requests.post(f"{API_BASE}/process/pending", timeout=60)
                    if response.status_code == 200:
                        st.success("✅ Processing started for pending files")
                    else:
                        st.error("❌ Failed to start processing")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # Tab 3: Document Management
    with tab3:
        st.header("📋 Document Management")
        
        # Refresh button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Refresh Document List"):
                st.rerun()
        with col2:
            if st.button("🗑️ Remove All Documents", type="secondary"):
                if st.session_state.get("confirm_remove_all", False):
                    result = remove_all_documents()
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success("✅ All documents removed")
                        st.session_state.confirm_remove_all = False
                        st.rerun()
                else:
                    st.session_state.confirm_remove_all = True
                    st.warning("⚠️ Click again to confirm removal of ALL documents")
        
        # Get documents
        documents = get_documents()
        
        if not documents:
            st.info("📄 No documents processed yet. Upload some files to get started!")
        else:
            st.write(f"**Total Documents:** {len(documents)}")
            
            # Create DataFrame for better display
            doc_data = []
            for doc in documents:
                status_icons = ""
                if doc.get("in_cache", False):
                    status_icons += "💾"
                if doc.get("in_processed_dir", False):
                    status_icons += "📁"
                if doc.get("in_knowledge_base", False):
                    status_icons += "🧠"
                
                doc_data.append({
                    "Document": doc["name"],
                    "Status": status_icons,
                    "Size": doc.get("file_size", "Unknown"),
                    "Processed": doc.get("process_date", "Unknown")
                })
            
            df = pd.DataFrame(doc_data)
            
            # Display documents with remove buttons
            for i, doc in enumerate(documents):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{doc['name']}**")
                    with col2:
                        status = ""
                        if doc.get("in_cache"): status += "💾"
                        if doc.get("in_processed_dir"): status += "📁"
                        if doc.get("in_knowledge_base"): status += "🧠"
                        st.write(status)
                    with col3:
                        st.write(doc.get("file_size", "—"))
                    with col4:
                        st.write(doc.get("process_date", "—"))
                    with col5:
                        if st.button("🗑️", key=f"remove_{i}", help=f"Remove {doc['name']}"):
                            result = remove_document(doc["name"])
                            if "error" in result:
                                st.error(f"❌ {result['error']}")
                            else:
                                st.success(f"✅ Removed {doc['name']}")
                                st.rerun()
                    
                    st.divider()
            
            # Legend
            with st.expander("🔍 Status Legend"):
                st.write("💾 = In processing cache")
                st.write("📁 = In processed files directory")
                st.write("🧠 = In knowledge base")

    # Tab 4: Analytics
    with tab4:
        st.header("📊 System Analytics")
        
        status = get_system_status()
        if status:
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Documents",
                    len(status.get("documents", [])),
                    help="Number of processed documents"
                )
            
            with col2:
                st.metric(
                    "Cache Entries",
                    status.get("processed_files_count", 0),
                    help="Documents in processing cache"
                )
            
            with col3:
                st.metric(
                    "Pending Files",
                    status.get("pending_files_count", 0),
                    help="Files waiting to be processed"
                )
            
            with col4:
                kb_active = "✅ Active" if status.get("lightrag_storage_exists") else "❌ Inactive"
                st.metric(
                    "Knowledge Base",
                    kb_active,
                    help="LightRAG knowledge graph status"
                )
            
            # Storage Information
            st.subheader("💾 Storage Directories")
            storage_dirs = status.get("storage_directories", {})
            for name, path in storage_dirs.items():
                st.text(f"{name}: {path}")
            
            # Document Details
            if status.get("documents"):
                st.subheader("📄 Document Details")
                
                # Create charts
                doc_status_counts = {"Cache": 0, "Processed": 0, "Knowledge Base": 0}
                for doc in status["documents"]:
                    if doc.get("in_cache"): doc_status_counts["Cache"] += 1
                    if doc.get("in_processed_dir"): doc_status_counts["Processed"] += 1
                    if doc.get("in_knowledge_base"): doc_status_counts["Knowledge Base"] += 1
                
                # Display as chart
                chart_data = pd.DataFrame(
                    list(doc_status_counts.items()),
                    columns=["Storage Type", "Count"]
                )
                st.bar_chart(chart_data.set_index("Storage Type"))
        
        else:
            st.error("❌ Could not load system analytics")

if __name__ == "__main__":
    main()