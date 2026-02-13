# Career Intelligence Assistant

RAG-based conversational AI that analyzes resumes against job descriptions to help candidates understand skill gaps, experience alignment, strengths, and interview preparation insights.

*Full documentation in Step 13.*

## Quick Start

```bash
# Clone repo
git clone <repo-url>
cd newpage-careerfit-rag-assistant_ayimer

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload

# Run UI (in separate terminal)
streamlit run ui/streamlit_app.py
```

## Docker Deployment

**Quick start with Docker:**

```bash
cp .env.docker.example .env
# Edit .env with your API keys

docker-compose up -d --build
```

- **API:** http://localhost:8000
- **UI:** http://localhost:8501

**Helper scripts:** `build.ps1`, `run.ps1`, `stop.ps1`, `logs.ps1` (or `.sh` on Unix)

**Configuration:** See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for:
- Environment variables
- Volume management and backups
- Health checks and monitoring
- Production considerations and scaling
