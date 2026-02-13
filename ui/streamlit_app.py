"""Career Intelligence Assistant - Streamlit UI.

Step 9: Document upload, chat interface, and RAG query integration.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import httpx
import streamlit as st

# --- Constants ---
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
DOC_TYPES = ["resume", "job_description"]


# --- API Functions ---
def upload_document(file, doc_type: str, api_url: str) -> dict:
    """POST file to /api/upload. Returns response dict or error dict."""
    if not file:
        return {"success": False, "error": "No file selected"}
    if doc_type not in DOC_TYPES:
        return {"success": False, "error": f"Invalid doc_type. Use: {DOC_TYPES}"}
    ext = Path(file.name or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"success": False, "error": f"Invalid file type. Allowed: .pdf, .docx. Got: {ext}"}

    url = f"{api_url.rstrip('/')}/api/upload"
    try:
        file_bytes = file.read()
        files = {"file": (file.name, file_bytes, file.type or "application/octet-stream")}
        data = {"doc_type": doc_type}
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, files=files, data=data)
        resp.raise_for_status()
        return {"success": True, "data": resp.json()}
    except httpx.ConnectError:
        return {"success": False, "error": f"Cannot connect to API at {api_url}. Is the server running?"}
    except httpx.TimeoutException:
        return {"success": False, "error": "Upload timed out. Try a smaller file or check server."}
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return {"success": False, "error": str(detail) if isinstance(detail, list) else detail}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_documents(api_url: str) -> list[dict]:
    """GET from /api/documents. Returns list of documents or empty on error."""
    url = f"{api_url.rstrip('/')}/api/documents"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def delete_document(doc_id: str, api_url: str) -> bool:
    """DELETE from /api/documents/{doc_id}. Returns True on success."""
    url = f"{api_url.rstrip('/')}/api/documents/{doc_id}"
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.delete(url)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def query_rag(query: str, filters: dict, api_url: str, top_k: int = 5) -> dict | None:
    """POST to /api/query. Returns QueryResponse dict or None on error."""
    url = f"{api_url.rstrip('/')}/api/query"
    payload = {
        "query": query,
        "filter_doc_type": filters.get("doc_type"),
        "filter_doc_id": filters.get("doc_id"),
        "max_results": top_k,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        return None  # Handled as connection error in UI
    except Exception:
        return None


def check_api_health(api_url: str) -> bool:
    """Check if API is reachable."""
    url = f"{api_url.rstrip('/')}/api/health"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
        return resp.status_code == 200
    except Exception:
        return False


# --- UI Components ---
def display_sources(sources: list[dict]) -> None:
    """Format and display source citations in expanders."""
    if not sources:
        return
    st.subheader("Sources")
    for i, src in enumerate(sources, 1):
        doc_name = src.get("doc_name", "Unknown")
        doc_type = src.get("doc_type", "")
        score = src.get("relevance_score", 0.0)
        chunk = src.get("chunk_text", "")[:500]
        if len(src.get("chunk_text", "")) > 500:
            chunk += "..."
        with st.expander(f"Source {i}: {doc_name} ({doc_type}) — Score: {score:.2f}"):
            st.text(chunk)


def inject_custom_css() -> None:
    """Optional custom CSS for chat bubbles and styling."""
    st.markdown("""
    <style>
        .user-bubble {
            background: #e3f2fd;
            padding: 1rem 1.25rem;
            border-radius: 1rem 1rem 0 1rem;
            margin: 0.5rem 0;
            margin-left: 2rem;
        }
        .assistant-bubble {
            background: #f5f5f5;
            padding: 1rem 1.25rem;
            border-radius: 1rem 1rem 1rem 0;
            margin: 0.5rem 0;
            margin-right: 2rem;
        }
        .source-citation {
            background: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 0.5rem 1rem;
            margin: 0.25rem 0;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)


# --- Session State Init ---
def init_session_state() -> None:
    """Initialize session state for chat and documents."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []


# --- Main App ---
def main() -> None:
    st.set_page_config(
        page_title="Career Intelligence Assistant",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()

    # --- Sidebar ---
    with st.sidebar:
        st.title("📋 CareerFit")
        st.caption("Upload documents & ask questions")

        # Settings
        st.subheader("Settings")
        api_url = st.text_input(
            "API Endpoint",
            value=DEFAULT_API_URL,
            help="Base URL of the Career Intelligence API",
        )
        top_k = st.slider(
            "Top-k Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of source chunks to retrieve",
        )

        # API status
        if check_api_health(api_url):
            st.success("API connected")
        else:
            st.error("API not reachable. Start the server first.")

        st.divider()
        st.subheader("Upload Documents")

        doc_type = st.radio(
            "Document Type",
            options=DOC_TYPES,
            format_func=lambda x: "Resume" if x == "resume" else "Job Description",
        )

        # Single file upload per action
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx"],
            key="file_uploader",
        )

        if st.button("Upload", type="primary"):
            if uploaded_file:
                with st.spinner("Uploading..."):
                    result = upload_document(uploaded_file, doc_type, api_url)
                if result.get("success"):
                    st.success(f"Uploaded: {result['data'].get('file_name', 'document')}")
                    st.session_state.uploaded_docs = get_documents(api_url)
                    st.rerun()
                else:
                    st.error(result.get("error", "Upload failed"))
            else:
                st.warning("Select a file first")

        # Documents list
        st.divider()
        st.subheader("Uploaded Documents")
        docs = get_documents(api_url)
        if not docs:
            st.info("No documents yet. Upload above.")
        else:
            for doc in docs:
                file_id = doc.get("file_id", "")
                file_name = doc.get("file_name", "Unknown")
                file_type = doc.get("file_type", "")
                status = doc.get("status", "")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"{file_name} ({file_type}) — {status}")
                with col2:
                    if st.button("🗑️", key=f"del_{file_id}", help="Delete document"):
                        if delete_document(file_id, api_url):
                            st.success("Deleted")
                            st.session_state.uploaded_docs = get_documents(api_url)
                            st.rerun()
                        else:
                            st.error("Delete failed")

        st.divider()
        if st.button("Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # --- Main Chat Area ---
    st.header("Chat")
    st.markdown("Ask questions about your resume and job descriptions.")

    # Filter options (optional)
    docs = get_documents(api_url)
    filter_doc_type = None
    filter_doc_id = None
    if docs:
        with st.expander("Filter by document (optional)"):
            type_choices = ["All"] + list(set(d.get("file_type") for d in docs if d.get("file_type")))
            selected_type = st.selectbox("Document type", type_choices)
            if selected_type != "All":
                filter_doc_type = selected_type
            doc_choices = ["All"] + [f"{d.get('file_name', '')} ({d.get('file_id', '')[:8]})" for d in docs]
            selected_doc = st.selectbox("Document", doc_choices)
            if selected_doc != "All" and docs:
                idx = doc_choices.index(selected_doc) - 1
                if 0 <= idx < len(docs):
                    filter_doc_id = docs[idx].get("file_id")

    filters = {"doc_type": filter_doc_type, "doc_id": filter_doc_id}

    # Chat history display
    for entry in st.session_state.chat_history:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if "sources" in entry and entry["sources"]:
                    display_sources(entry["sources"])
                if "confidence" in entry and entry["confidence"] is not None:
                    st.caption(f"Confidence: {entry['confidence']:.0%}")

    # Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not check_api_health(api_url):
            st.error("API is not reachable. Start the server before asking questions.")
            st.stop()

        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query and show response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_rag(prompt, filters, api_url, top_k)
            if response is None:
                err_msg = "Failed to get response. Check API connection and logs."
                st.error(err_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": err_msg,
                    "sources": [],
                    "confidence": None,
                })
            else:
                answer = response.get("answer", "No answer returned.")
                st.markdown(answer)
                sources = response.get("sources", [])
                display_sources(sources)
                confidence = response.get("confidence")
                if confidence is not None:
                    st.caption(f"Confidence: {confidence:.0%}")

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })


if __name__ == "__main__":
    main()
