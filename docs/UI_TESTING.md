# Career Intelligence Assistant - UI Testing

Manual testing checklist for the Streamlit UI (`ui/streamlit_app.py`).

## Prerequisites

1. Start the API server:
   ```powershell
   .\run_server.ps1
   ```
   Or: `uvicorn app.main:app --host 127.0.0.1 --port 8000`

2. Start the Streamlit UI:
   ```powershell
   streamlit run ui/streamlit_app.py
   ```

3. Verify API is healthy: Sidebar should show "API connected" in green.

---

## Test Scenarios

### 1. Document Upload

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| Upload Resume (PDF) | 1. Select "Resume" 2. Choose a PDF file 3. Click Upload | Success message, document appears in "Uploaded Documents" list |
| Upload Resume (DOCX) | 1. Select "Resume" 2. Choose a DOCX file 3. Click Upload | Success message, document appears in list |
| Upload Job Description | 1. Select "Job Description" 2. Choose PDF or DOCX 3. Click Upload | Success message, document appears in list |
| Upload without file | Click Upload without selecting a file | Warning: "Select a file first" |
| Invalid file type | Try to upload .txt or .jpg | Error: "Invalid file type. Allowed: .pdf, .docx" |
| API down | Stop API, try upload | Error: "Cannot connect to API..." |

### 2. Document List & Delete

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| View documents | Upload 1+ documents | List shows file name, type, status |
| Delete document | Click 🗑️ next to a document | Document removed from list |
| Delete non-existent | (Edge case - API 404) | Error message shown |

### 3. Chat & RAG Query

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| Simple question | 1. Upload resume 2. Ask "What skills does this resume highlight?" | Answer displayed with source citations |
| No documents | Ask question with no documents uploaded | "No relevant information found" or similar fallback |
| Question about JD | Upload JD, ask "What are the main requirements?" | Answer based on job description content |
| Mixed query | Upload resume + JD, ask "How do my skills match the job?" | Answer combines both sources |
| API down | Stop API, submit question | Error: "API is not reachable" |

### 4. Filters (doc_type, doc_id)

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| Filter by type | 1. Upload resume + JD 2. Expand "Filter by document" 3. Select doc_type "resume" 4. Ask question | Answer only uses resume sources |
| Filter by document | 1. Select specific document 2. Ask question | Answer only uses that document's content |
| No filter | Leave "All" for type and document | Answer uses all uploaded documents |

### 5. Settings

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| API URL change | Change endpoint to wrong URL | "API not reachable" |
| Top-k slider | Set top-k to 3, ask question | Fewer sources returned |
| Clear chat | Click "Clear Chat" | Chat history cleared |

### 6. Error Handling

| Scenario | Steps | Expected Result |
|----------|-------|-----------------|
| Empty file | Upload 0-byte file | Error from API |
| Large file | Upload >10 MB file | Error: "File too large" |
| Malformed response | (Simulated - requires API modification) | Graceful error display |
| Network timeout | Slow/unreachable API | Timeout error message |

---

## Quick Smoke Test

1. Start API and UI.
2. Upload one resume (PDF).
3. Wait for "completed" status.
4. Ask: "Summarize my experience."
5. Verify answer + sources displayed.
6. Change filter to "resume" only, ask again.
7. Delete the document.
8. Clear chat.

---

## Notes

- `st.spinner()` should appear during upload and query.
- Session state persists chat across sidebar interactions.
- Source citations use expanders; relevance score shown per source.
- Confidence score (if returned) shown as percentage.
