# 🔗 Resume MCP Integration Guide

Complete guide for integrating your Resume MCP server with the AI Job Intelligence Platform.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup Steps](#setup-steps)
4. [Resume JSON Schema](#resume-json-schema)
5. [API Integration](#api-integration)
6. [Troubleshooting](#troubleshooting)

## Overview

Phase 2 integrates with your [Resume MCP](https://github.com/anix-lynch/resume-mcp) to provide personalized job matching. The integration:

- **Auto-loads** your resume from `data/resume.json`
- **Pre-fills** search forms with your skills and experience
- **Personalizes** job matches based on your profile
- **Filters** by your target roles and salary preferences

## Prerequisites

### 1. Resume MCP Server (Optional)
If you want to use the live MCP server:
```bash
git clone https://github.com/anix-lynch/resume-mcp.git
cd resume-mcp
pip install -r requirements.txt
python server_http.py
```

### 2. Resume JSON File (Required)
At minimum, you need a `resume.json` file in the `data/` directory.

## Setup Steps

### Step 1: Get Your Resume JSON

**Option A: From Resume MCP Server**
```bash
curl https://your-resume-mcp-url.vercel.app/api/resume > data/resume.json
```

**Option B: Use Template**
Copy the example from `data/resume.json` and customize:
```bash
cp data/resume.json data/my-resume.json
# Edit my-resume.json with your information
```

### Step 2: Verify Resume Format
Run the test script:
```bash
python test_resume_integration.py
```

You should see:
```
✅ Resume loaded successfully!
✅ Profile Summary: Your Name - Your Title
✅ Top Skills: Python, ML, Data Engineering...
✅ All tests passed!
```

### Step 3: Run Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` and you should see your profile auto-loaded!

## Resume JSON Schema

### Minimal Schema
```json
{
  "name": "Your Name",
  "title": "Your Professional Title",
  "skills": {
    "Python": 10,
    "Machine Learning": 9
  },
  "target_roles": ["AI Engineer"],
  "target_rate_range": {
    "min": 70,
    "max": 200,
    "currency": "USD",
    "unit": "hour"
  }
}
```

### Full Schema
```json
{
  "name": "string",
  "title": "string",
  "contact": {
    "email": "string",
    "linkedin": "url",
    "github": "url",
    "portfolio": "url"
  },
  "skills": {
    "SkillName": 1-10
  },
  "projects": [
    {
      "name": "string",
      "description": "string",
      "tech": ["string"],
      "weight": 1-10
    }
  ],
  "experience": [
    {
      "company": "string",
      "title": "string",
      "duration": "string",
      "keywords": ["string"]
    }
  ],
  "certifications": ["string"],
  "target_roles": ["string"],
  "target_rate_range": {
    "min": number,
    "max": number,
    "currency": "string",
    "unit": "string"
  }
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ | Your full name |
| `title` | string | ✅ | Professional title/headline |
| `skills` | object | ✅ | Skills with proficiency (1-10) |
| `target_roles` | array | ✅ | Job titles you're seeking |
| `target_rate_range` | object | ✅ | Salary/rate preferences |
| `contact` | object | ❌ | Contact information |
| `projects` | array | ❌ | Portfolio projects |
| `experience` | array | ❌ | Work history |
| `certifications` | array | ❌ | Certifications & courses |

## API Integration

### Resume Loader Class

The `ResumeLoader` class provides easy access to all resume data:

```python
from utils.resume_loader import ResumeLoader

# Initialize
resume = ResumeLoader()

# Get profile summary
print(resume.get_profile_summary())
# Output: "Anix Lynch - AI Architect & VC Strategist"

# Get skills
print(resume.get_skills_text())
# Output: "Python, Machine Learning, Data Engineering..."

# Get target roles
print(resume.get_target_roles())
# Output: ["AI Architect", "Data Engineer", "ML Engineer"]

# Get full resume text for vector search
resume_text = resume.get_resume_text()
```

### Available Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_profile_summary()` | str | Name and title |
| `get_skills_text()` | str | Comma-separated skills |
| `get_skills_dict()` | dict | Skills with proficiency |
| `get_projects()` | list | Project details |
| `get_experience()` | list | Work experience |
| `get_target_roles()` | list | Desired job titles |
| `get_target_salary()` | dict | Salary preferences |
| `get_certifications()` | list | Certifications |
| `get_resume_text()` | str | Full resume for matching |
| `get_contact_info()` | dict | Contact details |
| `format_salary_preference()` | str | Formatted salary string |

### Using in Your Code

```python
import streamlit as st
from utils.resume_loader import ResumeLoader

# Load once and cache
@st.cache_resource
def load_resume():
    return ResumeLoader()

resume = load_resume()

# Display profile
st.write(f"**Name:** {resume.get_profile_summary()}")
st.write(f"**Skills:** {resume.get_skills_text()}")
st.write(f"**Salary:** {resume.format_salary_preference()}")

# Use in vector search
search_query = resume.get_resume_text()
results = vector_store.search(search_query)

# Filter by target roles
target_roles = resume.get_target_roles()
filtered_jobs = jobs_df[jobs_df['title'].isin(target_roles)]
```

## Troubleshooting

### Issue: "Resume file not found"

**Problem:** `data/resume.json` doesn't exist

**Solution:**
```bash
# Create the data directory
mkdir -p data

# Download from Resume MCP or copy template
curl -o data/resume.json https://raw.githubusercontent.com/anix-lynch/resume-mcp/main/resume.json
```

### Issue: "Invalid JSON format"

**Problem:** Resume JSON is malformed

**Solution:**
```bash
# Validate JSON
python -m json.tool data/resume.json

# Or use the test script
python test_resume_integration.py
```

### Issue: "Missing required fields"

**Problem:** Resume JSON lacks required keys

**Solution:**
Ensure these fields exist:
```json
{
  "name": "...",
  "title": "...",
  "skills": {...},
  "target_roles": [...],
  "target_rate_range": {...}
}
```

### Issue: "Skills not showing in search"

**Problem:** Skills proficiency too low or empty

**Solution:**
- Skills should be rated 1-10
- At least 5-10 skills recommended
- Higher proficiency = higher priority

```json
{
  "skills": {
    "Python": 10,
    "Machine Learning": 9,
    "Data Engineering": 9
  }
}
```

### Issue: "No jobs match my profile"

**Problem:** Target roles or skills don't match dataset

**Solution:**
1. Check target roles match dataset:
```python
# See available job titles
df['title'].unique()
```

2. Broaden your search:
```json
{
  "target_roles": [
    "AI Engineer",
    "ML Engineer", 
    "Data Scientist",
    "Data Engineer"
  ]
}
```

3. Adjust salary expectations if too restrictive

## Advanced Integration

### Custom Data Sources

To load from different sources:

```python
# Load from URL
import requests
response = requests.get("https://api.example.com/resume")
resume_data = response.json()

# Load from database
import sqlite3
conn = sqlite3.connect("resumes.db")
resume_data = conn.execute("SELECT * FROM resumes WHERE id=1").fetchone()

# Initialize ResumeLoader with custom path
resume = ResumeLoader("path/to/custom/resume.json")
```

### Live MCP Integration

To fetch from Resume MCP server in real-time:

```python
import requests

def fetch_from_mcp(mcp_url):
    """Fetch resume from MCP server"""
    response = requests.get(f"{mcp_url}/api/resume")
    if response.status_code == 200:
        return response.json()
    return None

# Usage
mcp_url = "https://resume-mcp.vercel.app"
resume_data = fetch_from_mcp(mcp_url)

if resume_data:
    # Save locally
    with open("data/resume.json", "w") as f:
        json.dump(resume_data, f, indent=2)
```

### Multi-Resume Support

To switch between multiple resumes:

```python
def load_resume_by_id(resume_id: str):
    """Load specific resume version"""
    path = f"data/resume_{resume_id}.json"
    return ResumeLoader(path)

# Usage
tech_resume = load_resume_by_id("tech")
finance_resume = load_resume_by_id("finance")

# Use appropriate resume based on job type
if job['industry'] == 'Tech':
    resume = tech_resume
else:
    resume = finance_resume
```

## Next Steps

1. ✅ Verify integration with test script
2. ✅ Customize your resume.json
3. ✅ Run the application
4. 🚀 Deploy to Streamlit Cloud
5. 📊 Monitor job matches
6. 🎯 Refine your profile based on results

## Support

- **GitHub Issues**: [Report bugs](https://github.com/anix-lynch/ai-agent-job-intelligence-phase-2/issues)
- **Discussions**: [Ask questions](https://github.com/anix-lynch/ai-agent-job-intelligence-phase-2/discussions)
- **Resume MCP**: [MCP Server docs](https://github.com/anix-lynch/resume-mcp)

---

**Last Updated**: 2025-11-19  
**Version**: Phase 2.0