# 🚀 Streamlit Cloud Deployment - AI Job Intelligence Phase 2

**Deployment Status:** Ready to deploy  
**Repository:** `anix-lynch/ai-agent-job-intelligence-phase-2`  
**Custom Domain:** `gozeroshot.dev` (if configured)

---

## ✅ Pre-Deployment Checklist

- [x] Code pushed to GitHub (`main` branch)
- [x] `requirements.txt` with full ML dependencies
- [x] `.streamlit/config.toml` configured
- [x] `app.py` as main entry point
- [x] `data/resume.json` included
- [x] `data/foorilla_all_jobs.csv` included

---

## 📋 Deployment Configuration

### Repository Details

| Setting | Value |
|---------|-------|
| **Repository** | `anix-lynch/ai-agent-job-intelligence-phase-2` |
| **Branch** | `main` |
| **Main file** | `app.py` |
| **Python version** | 3.10 (auto-detected) |

### App Settings

| Setting | Value |
|---------|-------|
| **App name** | `ai-agent-job-intelligence-phase2` |
| **URL** | `https://ai-agent-job-intelligence-phase2.streamlit.app` |
| **Custom domain** | `jobs.gozeroshot.dev` (optional) |

---

## 🎯 Deployment Steps

### Method 1: Via Streamlit Cloud Dashboard (Recommended)

1. **Go to Streamlit Cloud**
   - URL: https://share.streamlit.io
   - Sign in with GitHub account

2. **Create New App**
   - Click "New app" button
   - Or: https://share.streamlit.io/deploy

3. **Configure Deployment**
   ```
   Repository: anix-lynch/ai-agent-job-intelligence-phase-2
   Branch: main
   Main file path: app.py
   ```

4. **Advanced Settings (Optional)**
   ```
   Python version: 3.10
   Secrets: (none required for public demo)
   ```

5. **Deploy**
   - Click "Deploy!" button
   - Wait 3-5 minutes for build
   - App will be live at: `https://ai-agent-job-intelligence-phase2.streamlit.app`

---

### Method 2: Direct Deploy URL

**One-Click Deploy:**
```
https://share.streamlit.io/deploy?repository=anix-lynch/ai-agent-job-intelligence-phase-2&branch=main&mainModule=app.py
```

Just click the link above when logged into Streamlit Cloud!

---

## 🌐 Custom Domain Setup (gozeroshot.dev)

### Option 1: Subdomain (Recommended)

**Setup:**
1. Go to Streamlit Cloud app settings
2. Navigate to "Custom domain"
3. Add subdomain: `jobs.gozeroshot.dev`

**DNS Configuration:**
```
Type: CNAME
Name: jobs
Value: ai-agent-job-intelligence-phase2.streamlit.app
TTL: 3600
```

**Result:** `https://jobs.gozeroshot.dev`

---

### Option 2: Root Domain

**Setup:**
1. Add domain in Streamlit Cloud settings
2. Configure DNS with A records

**DNS Configuration:**
```
Type: A
Name: @
Value: [Streamlit Cloud IP - provided in settings]
TTL: 3600
```

**Result:** `https://gozeroshot.dev/jobs` (requires path routing)

---

## 🔧 Environment Variables / Secrets

### Current Setup: No Secrets Required

Your app auto-loads API keys from:
1. `utils/get_secret.py` (local)
2. Environment variables (fallback)

### If You Need Secrets (Future):

**Add in Streamlit Cloud:**
1. Go to app settings
2. Click "Secrets"
3. Add in TOML format:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
DEEPSEEK_API_KEY = "..."
ANTHROPIC_API_KEY = "..."
```

**Access in code:**
```python
import streamlit as st
api_key = st.secrets["OPENAI_API_KEY"]
```

---

## 📊 Expected Build Output

### Build Process (3-5 minutes)

```
✓ Cloning repository...
✓ Installing Python 3.10...
✓ Installing dependencies from requirements.txt...
  - streamlit>=1.28.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scikit-learn>=1.3.0
  - sentence-transformers>=2.2.2  ← ~2 GB (takes time)
  - chromadb>=0.4.15              ← ~1.5 GB
  - faiss-cpu>=1.7.4              ← ~500 MB
  - langchain>=0.1.0
  - plotly>=5.15.0
✓ Starting app...
✓ App is live!
```

**Note:** First deployment takes longer due to ML dependencies.

---

## 🎨 App Features (After Deployment)

### Available Modes

1. **Vector Search (Semantic)**
   - Uses sentence-transformers + ChromaDB
   - Semantic job matching
   - Auto-filled with your resume

2. **ATS Classifier**
   - ML-based ATS prediction
   - scikit-learn classifier
   - Resume compatibility score

3. **AI Agent Orchestration**
   - LangChain ReAct framework
   - Multi-agent reasoning
   - Requires API key (user-provided)

4. **Browse All Jobs**
   - 1000+ AI/ML jobs
   - Filtering by company, salary
   - No API key required

---

## 🔍 Post-Deployment Verification

### Test Checklist

- [ ] App loads successfully
- [ ] Profile dashboard shows your resume data
- [ ] Vector search works with auto-filled resume
- [ ] ATS classifier predicts scores
- [ ] Job browsing displays 1000+ jobs
- [ ] Filters work (company, salary)
- [ ] Mobile responsive
- [ ] Custom domain resolves (if configured)

### Test URLs

**Main App:**
```
https://ai-agent-job-intelligence-phase2.streamlit.app
```

**Custom Domain (if configured):**
```
https://jobs.gozeroshot.dev
```

---

## 📈 Monitoring & Analytics

### Streamlit Cloud Dashboard

**Metrics Available:**
- Active users
- Total visitors
- App uptime
- Error logs
- Resource usage (RAM, CPU)

**Access:**
```
https://share.streamlit.io/[your-workspace]/ai-agent-job-intelligence-phase2
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"

**Cause:** Missing dependency in `requirements.txt`

**Solution:**
```bash
# Add to requirements.txt
pip freeze | grep [module-name] >> requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Add missing dependency"
git push origin main

# Streamlit Cloud auto-redeploys
```

---

### Issue: "Out of memory"

**Cause:** App exceeds 1 GB RAM limit

**Solution:**
1. Reduce dataset size in code:
```python
jobs_df = jobs_df.head(1000)  # Limit to 1000 jobs
```

2. Or upgrade to Streamlit Team ($20/month, 4 GB RAM)

---

### Issue: "App not updating"

**Cause:** Changes not pushed to GitHub

**Solution:**
```bash
git add .
git commit -m "Update app"
git push origin main

# Streamlit Cloud auto-deploys from main branch
# Or click "Reboot app" in dashboard
```

---

## 🔄 Update Workflow

### Continuous Deployment (Automatic)

**Every time you push to `main`:**
```bash
git add .
git commit -m "Your changes"
git push origin main
```

**Streamlit Cloud automatically:**
1. Detects push to `main`
2. Rebuilds app
3. Redeploys
4. Live in 2-3 minutes

**No manual deployment needed!** ✨

---

## 💰 Cost & Limits

### Free Tier (Community)

| Resource | Limit | Your Usage |
|----------|-------|------------|
| **Apps** | Unlimited public | 1 app |
| **RAM** | 1 GB per app | ~800 MB |
| **CPU** | Shared | Sufficient |
| **Storage** | 1 GB | ~50 MB (data files) |
| **Bandwidth** | Unlimited | N/A |
| **Custom domain** | ✅ Supported | Optional |

**Total Cost:** **$0/month** ✅

---

## 🎯 Distro Dojo Compliance

### 4-Platform Strategy Status

| Platform | Status | URL |
|----------|--------|-----|
| **1. Streamlit Cloud** | ✅ Ready to deploy | `ai-agent-job-intelligence-phase2.streamlit.app` |
| **2. Vercel** | ❌ Not compatible | N/A (backend app) |
| **3. Fly.io** | ✅ Deployed (compressed) | `ai-agent-job-intelligence.fly.dev` |
| **4. Local/Dev** | ✅ Working | `localhost:8501` |

**Primary Deployment:** Streamlit Cloud (free, full ML features)  
**Secondary Deployment:** Fly.io (paid, compressed version)  
**Development:** Local environment

---

## 📝 Quick Reference

### Deployment URLs

```bash
# Streamlit Cloud (to be deployed)
https://ai-agent-job-intelligence-phase2.streamlit.app

# Custom domain (optional)
https://jobs.gozeroshot.dev

# Fly.io (current, compressed)
https://ai-agent-job-intelligence.fly.dev

# Local development
http://localhost:8501
```

### Repository

```bash
# GitHub
https://github.com/anix-lynch/ai-agent-job-intelligence-phase-2

# Clone
git clone git@github.com:anix-lynch/ai-agent-job-intelligence-phase-2.git
```

### Commands

```bash
# Local run
streamlit run app.py

# Test locally
python test_resume_integration.py

# Deploy (automatic on push)
git push origin main
```

---

## ✅ Ready to Deploy!

**Everything is configured and ready:**
- ✅ Code on GitHub (`main` branch)
- ✅ Full `requirements.txt` with ML dependencies
- ✅ `.streamlit/config.toml` configured
- ✅ Resume data in `data/resume.json`
- ✅ Job data in `data/foorilla_all_jobs.csv`

**Next Step:**
1. Go to https://share.streamlit.io
2. Click "New app"
3. Select repository: `anix-lynch/ai-agent-job-intelligence-phase-2`
4. Click "Deploy!"
5. Wait 3-5 minutes
6. Done! 🎉

---

**Last Updated:** 2025-11-19  
**Version:** 1.0  
**Deployment:** Streamlit Cloud (Primary - Distro Dojo Rule)
