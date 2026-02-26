## Cursor Cloud specific instructions

### Services

| Service | Command | Port | Notes |
|---------|---------|------|-------|
| Streamlit App | `streamlit run app.py --server.port=8502 --server.address=0.0.0.0 --server.headless=true` | 8502 | Main UI; auto-loads resume from `data/bronze/resume.json` |
| MCP HTTP Server | `cd mcp && uvicorn server_http:http_app --port 8000` | 8000 | Optional — only needed for ChatGPT/OpenAI MCP integration |

### Lint / Test / Build

- **Lint**: `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics` (critical errors only; broader check uses `--exit-zero`)
- **Integration test**: `python3 test_resume_integration.py`
- **Import smoke test**: `python3 -c "import streamlit; import pandas; import numpy; import sklearn; print('OK')"`
- No formal test suite (pytest, unittest) exists. CI runs flake8 + import checks only.

### Known issues (pre-existing)

- `features/feature_store/ats.py` is missing `import re`, causing the ATS Classifier mode to error at runtime. flake8 flags this as F821. The CI marks lint as `continue-on-error`.
- AI Agent Orchestration mode requires an external LLM API key (OpenAI/DeepSeek). Without one, the agent panel shows a warning but the rest of the app works fine.

### Environment notes

- The project targets Python 3.10 (`runtime.txt`), but runs fine on Python 3.12.
- `$HOME/.local/bin` must be on `PATH` for `streamlit`, `flake8`, and other pip-installed scripts.
