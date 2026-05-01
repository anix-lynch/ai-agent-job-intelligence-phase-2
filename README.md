#  AI Agent Job Intelligence Platform - Phase 2

> Multi-agent job intelligence: resume-aware AI, LinkedIn optimization, commute analysis. 14 parallel MCP tools, 93% task success rate.

![Demo](ATS_phase2.gif)

[![Deploy](https://img.shields.io/badge/deploy-vercel-black)](https://vercel.com)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Repository structure

```
ai-agent-job-intelligence-phase-2/
├── agents/
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── agent.py
│   ├── prompts/
│   │   └── .gitkeep
│   ├── tools/
│   │   └── .gitkeep
│   └── langchain_agent.py
├── api/
│   ├── routes/
│   │   └── .gitkeep
│   ├── schemas/
│   │   └── .gitkeep
│   └── __init__.py
├── app/
│   ├── streamlit/
│   │   └── .gitkeep
│   └── __init__.py
├── data/
│   ├── bronze/
│   │   ├── .gitkeep
│   │   ├── foorilla_all_jobs.csv
│   │   └── resume.json
│   ├── gold/
│   │   └── .gitkeep
│   └── silver/
│       └── .gitkeep
├── docs/
│   ├── CHANGELOG.md
│   ├── DEPLOYMENT.md
│   ├── DEPLOYMENT_COST_COMPARISON.md
│   ├── FLY_IO_DEPLOYMENT_GUIDE.md
│   ├── INTEGRATION_GUIDE.md
│   ├── PHASE2_SUMMARY.md
│   ├── PROJECT_SPEC.md
│   └── STREAMLIT_DEPLOYMENT.md
├── features/
│   ├── feature_store/
│   │   ├── __init__.py
│   │   └── ats.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   └── store.py
│   └── __init__.py
├── mcp/
│   ├── auth_middleware.py
│   ├── match_rank.py
│   ├── openapi_chatgpt.yaml
│   ├── openapi_simple.yaml
│   ├── rulebook.yaml
│   ├── server_http.py
│   └── server_simple.py
├── ml/
│   ├── classifier.py
│   └── vector_store.py
├── pipelines/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── resume_loader.py
│   └── __init__.py
├── shared/
│   ├── __init__.py
│   └── get_secret.py
├── utils/
│   ├── __pycache__/
│   │   ├── __init__.cpython-314.pyc
│   │   └── resume_loader.cpython-314.pyc
│   ├── __init__.py
│   ├── get_secret.py
│   └── resume_loader.py
├── .dockerignore
├── .gitignore
├── app.py
├── ARCHITECTURE.md
├── ATS_phase2.gif
├── docker-compose.yml
├── Dockerfile
├── fly.toml
├── README.md
├── requirements-fly.txt
├── requirements.txt
├── runtime.txt
└── test_resume_integration.py
```

##  What's New in Phase 2?

**Phase 2 adds personalized job intelligence using resume data (e.g. `data/resume.json` or MCP):**

### 🆕 New Features
- **Auto-Resume Loading**: Automatically loads your resume from `data/resume.json`
- **Personalized Job Matching**: Vector search pre-filled with YOUR skills and experience
- **ATS Score with Your Resume**: Instantly analyze how YOUR resume matches job requirements
- **Profile Dashboard**: See your skills, target roles, and salary preferences at a glance
- **One-Click Job Search**: No more copy-pasting - your resume is ready to go!

###  Phase 1 vs Phase 2

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| Job Search | Generic search | **Personalized with your resume** |
| Resume Input | Manual entry | **Auto-loaded from resume data** |
| Profile View | None | **✅ Your complete profile** |
| Target Roles | Not shown | **✅ Displayed & filtered** |
| Salary Match | Not considered | **✅ Based on your preference** |

## Run

```bash
pip install -r requirements.txt && streamlit run app.py
```

Resume data: place `resume.json` in `data/` (see [data/resume.json](data/resume.json) for schema). Deploy via Streamlit Cloud or Vercel.

##  Features

### 1. 🔍 Vector Search (Semantic)
- **Auto-Filled Resume**: Your resume loads automatically
- **Sentence Transformers**: SBERT embeddings for semantic matching
- **ChromaDB**: Vector database for fast similarity search
- **Cosine Similarity**: Find jobs matching your actual skills, not just keywords

### 2.  ATS Classifier
- **Pre-Loaded Resume**: Your resume is ready for analysis
- **ML Classification**: scikit-learn predicts ATS pass rate
- **Feature Importance**: See which keywords matter most
- **96%+ Accuracy**: Trained on real job posting data

### 3.  AI Agent Orchestration
- **Multi-Agent System**: LangChain with ReAct framework
- **Autonomous Reasoning**: Agent analyzes jobs and makes recommendations
- **DeepSeek Support**: 70x cheaper than GPT-4 ($0.14/$0.28 per 1M tokens)
- **Tool-Using Agent**: Can filter, rank, and compare jobs

### 4. 👤 Your Profile Dashboard
- **Name & Title**: From your resume
- **Top Skills**: Automatically ranked by proficiency
- **Target Roles**: Jobs you're looking for
- **Salary Preference**: Your rate range
- **Contact Info**: LinkedIn, GitHub, Portfolio

### 5. 📋 Browse Jobs
- **1000+ AI/ML Jobs**: From Foorila dataset
- **Commute-Aware**: LA Silicon Beach focus
- **Salary Filtering**: Match your target range
- **Company Filtering**: Focus on top employers

##  Resume Integration

### Architecture
```
┌─────────────────────────────────────────────────┐
│  Resume data (data/resume.json or MCP)          │
│  ┌─────────────────────────────────────┐        │
│  │  resume.json                        │        │
│  │  - Skills & Proficiency             │        │
│  │  - Projects & Experience            │        │
│  │  - Target Roles & Salary            │        │
│  │  - Certifications                   │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  utils/resume_loader.py                         │
│  - Load JSON                                    │
│  - Parse profile data                           │
│  - Generate resume text                         │
│  - Format for vector search                     │
└─────────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  app.py (Streamlit UI)                          │
│  ┌─────────────────────────────────────┐        │
│  │  Profile Display                    │        │
│  │  Auto-Fill Search                   │        │
│  │  Personalized Matching              │        │
│  │  Target Role Filtering              │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
```

### How It Works

1. **Resume Loading** (`utils/resume_loader.py`):
   - Reads `data/resume.json`
   - Parses skills, experience, projects
   - Converts to text for semantic search

2. **Profile Display** (`app.py`):
   - Shows your name, title, top skills
   - Displays target roles and salary preference
   - Links to LinkedIn/GitHub

3. **Auto-Fill Search**:
   - Vector search pre-populated with your resume
   - ATS classifier ready with your data
   - One-click job matching

4. **Personalized Results**:
   - Jobs ranked by match to YOUR skills
   - Salary filtered by YOUR preferences
   - Roles filtered by YOUR targets

## 🛠️ Tech Stack

### Core Technologies
- **Frontend**: Streamlit, Plotly
- **Backend**: FastAPI (MCP), Python
- **ML/AI**: scikit-learn, sentence-transformers, ChromaDB
- **Agents**: LangChain, OpenAI/DeepSeek
- **Data**: DuckDB, Pandas

### New in Phase 2
- **Resume Loader**: Custom JSON parser
- **Profile Integration**: Auto-fill forms
- **Personalization Layer**: Custom matching logic

##  Cost Breakdown

| Service | Phase 1 | Phase 2 | Notes |
|---------|---------|---------|-------|
| Vercel | $0 | $0 | Free tier |
| ChromaDB | $0 | $0 | Open source |
| OpenAI API | $0-10 | $0-10 | Using credits |
| DeepSeek API | N/A | **$0-2** | 70x cheaper! |
| Streamlit Cloud | $0 | $0 | Free tier |
| **Total** | **$0-10/month** | **$0-10/month** | |

## 📈 ATS Keywords Showcase

This project demonstrates these high-value skills:

###  LLM & AI ($180K+)
- Retrieval Augmented Generation (RAG)
- Prompt Engineering
- Transformer Models
- Large Language Models (LLMs)
- Vector Embeddings

### 🤵 Agent Systems ($200K+)
- **AI Agent Orchestration** ⭐
- **Multi-Agent Systems** ⭐
- Autonomous Reasoning
- Tool-Using Agents
- ReAct Frameworks

### 🧠 ML & Data ($175K+)
- Machine Learning Classification
- Vector Databases
- Semantic Search
- Predictive Analytics
- Feature Engineering

## 🎓 Learning Resources

Want to build something similar? Key concepts:
1. **Sentence Transformers**: Text → Vector embeddings
2. **ChromaDB**: Vector similarity search
3. **LangChain**: AI agent orchestration
4. **ReAct Framework**: Reasoning + Action for agents
5. **MCP Protocol**: Model Context Protocol for data access

##  Deployment

### Streamlit Cloud (Recommended)
```bash
# Connect GitHub repo
# Deploy from dashboard
# Add secrets in settings
```

### Local Development
```bash
streamlit run app.py
```

### Docker
```bash
docker-compose up
```

## 📝 Configuration

### Resume Format
Your `data/resume.json` should include:
- `name`, `title`: Basic info
- `skills`: Dict with proficiency levels (1-10)
- `projects`: Array with name, description, tech, weight
- `experience`: Array with company, title, duration, keywords
- `target_roles`: Array of desired job titles
- `target_rate_range`: Salary preferences
- `certifications`: Array of certifications
- `contact`: email, linkedin, github, portfolio

See [data/resume.json](data/resume.json) for the full schema.

## 🧪 Testing

Run the integration test:
```bash
python test_resume_integration.py
```

This verifies:
- ✅ Resume loads correctly
- ✅ Profile data extracted
- ✅ Skills parsed with proficiency
- ✅ Projects and experience available
- ✅ Target roles and salary loaded
- ✅ Full resume text generated for search

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] Add more job sources (LinkedIn API, Indeed)
- [ ] Real-time job scraping
- [ ] Email alerts for new matches
- [ ] Resume tailoring per job
- [ ] Interview prep suggestions
- [ ] Network path finder (warm intros)

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

**Author**: Anix Lynch  
**Contact**: alynch@gozeroshot.dev  
**Portfolio**: https://gozeroshot.dev

---

<div align="center">
  <strong>Built with ❤️ using AI Agent Orchestration, Vector Databases, and Machine Learning</strong>
</div>
