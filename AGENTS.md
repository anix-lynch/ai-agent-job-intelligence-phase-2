## Cursor Cloud specific instructions

### Project overview
AI-powered Job Intelligence Platform (Streamlit + Python 3.10). See `README.md` for full details.

### Python environment
- Python 3.10 is required (`runtime.txt`). The venv lives at `/workspace/.venv`.
- Activate with: `source /workspace/.venv/bin/activate`

### Running the app
- `streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true`
- Health check: `curl http://localhost:8501/_stcore/health`

### Linting
- `flake8 . --exclude=.venv` (flake8 is not in `requirements.txt` but is used in CI; installed in the venv)
- CI critical checks: `flake8 . --select=E9,F63,F7,F82 --exclude=.venv`
- Note: existing code has one F821 error in `features/feature_store/ats.py` (missing top-level `import re`).

### Tests
- `python test_resume_integration.py` — verifies the resume loading pipeline.
- CI import smoke tests: `python -c "from features.vector_store import VectorStore"` etc. (see `.github/workflows/ci.yml`).
- No pytest/unittest suite exists.

### Gotchas
- The first load of Vector Search or ATS Classifier modes in the Streamlit app downloads the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB). This is cached after the first run.
- AI Agent Orchestration mode requires an `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` env var; all other modes work without API keys.
- Always exclude `.venv` when running flake8 or other tools that scan the project tree.
