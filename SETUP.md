# Development Environment Setup

Step-by-step commands to set up and verify the Career Intelligence Assistant project.

---

## 1. Virtual Environment

### Windows (PowerShell)

```powershell
# Navigate to project
cd c:\Users\AntenehAY\OneDrive\Desktop\NewPage_Solution\newpage-careerfit-rag-assistant_ayimer

# Create virtual environment
python -m venv .venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# If execution policy blocks the script:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows (Command Prompt)

```cmd
cd c:\Users\AntenehAY\OneDrive\Desktop\NewPage_Solution\newpage-careerfit-rag-assistant_ayimer
python -m venv .venv
.venv\Scripts\activate.bat
```

### Mac / Linux

```bash
# Navigate to project
cd /path/to/newpage-careerfit-rag-assistant_ayimer

# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate
```

**Verify activation:** Your prompt should show `(.venv)` at the start.

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Environment Variables

You already have `.env` with API keys. Ensure it contains:

- `ANTHROPIC_API_KEY` – for Claude (LLM)
- `OPENAI_API_KEY` – for embeddings
- `VECTOR_DB_PATH` – optional (default: `./data/vectordb`)

---

## 4. Run Verification Script

```bash
python verify_setup.py
```

Expected output when successful:

```
Career Intelligence Assistant - Setup Verification

1. Checking folders...
  OK: data
  OK: data\uploads
  OK: data\vectordb
  ...

Setup Complete
```

---

## 5. Test FastAPI Health Endpoint

**Start the server:**

```bash
uvicorn app.main:app --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Test the endpoint:**

- **Browser:** Open http://localhost:8000/health
- **curl:** `curl http://localhost:8000/health`
- **PowerShell:** `Invoke-RestMethod -Uri http://localhost:8000/health`

**Expected response:**
```json
{"status": "ok"}
```

**API docs:** http://localhost:8000/docs

---

## 6. Checklist Before Step 2

| Check | Status |
|-------|--------|
| Virtual environment activated | |
| Dependencies installed | |
| `verify_setup.py` passes | |
| `data/`, `data/uploads/`, `data/vectordb/` exist | |
| All `__init__.py` files present | |
| `.env` has required keys | |
| `/health` returns `{"status": "ok"}` | |

**If all pass → Ready for Step 2: Configuration & Core Models**
