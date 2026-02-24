# Architecture: AI Data Platform + GenAI

This repo is structured as an **AI Data Engineering + GenAI** system: data pipeline (bronze → silver → gold), feature/vector layer, and agent layer.

## Data flow

```
bronze (raw) → silver (cleaned) → gold (analytics) → features → agents → API / UI
```

| Layer | Location | Contents |
|-------|----------|----------|
| **Bronze** | `data/bronze/` | Raw job CSV, resume JSON (unaltered) |
| **Silver** | `data/silver/` | Cleaned, normalized jobs and resume features |
| **Gold** | `data/gold/` | Analytics-ready tables (job scoring, ATS features) |

## Pipeline

| Stage | Location | Role |
|-------|----------|------|
| **Ingestion** | `pipelines/ingestion/` | Load jobs, resume from bronze |
| **Transformation** | `pipelines/transformation/` | Clean, normalize, feature engineering |
| **Serving** | `pipelines/serving/` | Prepare outputs for API/agents |

## Feature & vector layer

| Component | Location | Role |
|-----------|----------|------|
| **Embeddings** | `features/embeddings/` | Sentence-transformers / embedding logic |
| **Vector store** | `features/vector_store/` | ChromaDB (semantic search) |
| **Feature store** | `features/feature_store/` | ATS features, scoring inputs, classifier |

## Agent layer

| Component | Location | Role |
|-----------|----------|------|
| **Orchestrator** | `agents/orchestrator/` | Main agent (ReAct / LangChain-style) |
| **Tools** | `agents/tools/` | search_jobs, rank_jobs, ats_score |
| **Prompts** | `agents/prompts/` | System and task prompts |

## Serving

| Layer | Location | Role |
|-------|----------|------|
| **API** | `api/routes/`, `api/schemas/` | FastAPI endpoints (search, match, ATS) |
| **UI** | `app/` | Streamlit UI; entry point `app.py` at repo root |

## Where things live

| Concept | Location |
|---------|----------|
| resume.json | `data/bronze/` |
| Raw job CSV | `data/bronze/` |
| Cleaned jobs | `data/silver/` |
| ATS scores / analytics | `data/gold/` |
| Embeddings | `features/vector_store/` (ChromaDB) |
| LangChain-style agent | `agents/orchestrator/` |
| Job search / match API | `api/` |
| Streamlit app | Root `app.py` |

## Shared

| Role | Location |
|------|----------|
| Config, secrets, helpers | `shared/` |

MCP server and integration live under `mcp/` and consume the pipeline and feature layers as needed.
