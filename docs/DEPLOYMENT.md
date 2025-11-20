# 🚀 Deployment Guide - Phase 2

Complete guide for deploying the AI Job Intelligence Platform with Resume MCP integration.

## 📋 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)
- ✅ Free hosting
- ✅ Auto-deploys from GitHub
- ✅ Built-in SSL/HTTPS
- ✅ Custom domain support
- ✅ No server management

### Option 2: Local Deployment
- For development and testing
- Full control over environment

### Option 3: Docker
- For production deployments
- Containerized environment

---

## 🌟 Option 1: Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account (you already have this ✅)
- Streamlit Cloud account (free)

### Step 1: Create Streamlit Cloud Account

1. Go to https://streamlit.io/cloud
2. Click "Sign up" or "Get started"
3. Sign in with your GitHub account
4. Authorize Streamlit to access your repositories

### Step 2: Deploy from GitHub

1. **Click "New app"** in Streamlit Cloud dashboard

2. **Repository Settings**:
   - Repository: `anix-lynch/ai-agent-job-intelligence-phase-2`
   - Branch: `main`
   - Main file path: `app.py`

3. **Advanced Settings** (Optional):
   - Python version: `3.11`
   - Add secrets if using API keys

4. **Click "Deploy"**

### Step 3: Wait for Deployment

- Initial deployment takes 2-5 minutes
- Streamlit will install dependencies from `requirements.txt`
- You'll see build logs in real-time

### Step 4: Access Your App

Your app will be available at:
```
https://[your-app-name].streamlit.app
```

Example: `https://ai-job-intelligence-phase2.streamlit.app`

### Step 5: Share Your App

- Copy the URL and share it
- Your resume will auto-load for anyone visiting
- Works on any device (mobile, tablet, desktop)

---

## 🔧 Option 2: Local Deployment

### For Development & Testing

```bash
# 1. Navigate to project directory
cd ai-agent-job-intelligence-phase-2

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py

# 5. Open in browser
# http://localhost:8501
```

### Using Docker Compose

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:8501
```

---

## 🐳 Option 3: Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t ai-job-intelligence-phase2 .

# Run container
docker run -p 8501:8501 ai-job-intelligence-phase2

# Access at http://localhost:8501
```

### Push to Docker Hub

```bash
# Tag image
docker tag ai-job-intelligence-phase2 yourusername/ai-job-intelligence-phase2:latest

# Push to Docker Hub
docker push yourusername/ai-job-intelligence-phase2:latest
```

---

## 🔐 Environment Variables & Secrets

### For Streamlit Cloud

If using API keys (OpenAI, DeepSeek, etc.):

1. Go to your app settings in Streamlit Cloud
2. Click "Secrets"
3. Add in TOML format:

```toml
# .streamlit/secrets.toml (for Streamlit Cloud)
OPENAI_API_KEY = "sk-..."
DEEPSEEK_API_KEY = "..."
ANTHROPIC_API_KEY = "..."
```

### For Local Development

Create `.env` file:
```bash
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
ANTHROPIC_API_KEY=...
```

---

## 📊 Deployment Checklist

### Before Deploying

- [x] Resume data in `data/resume.json`
- [x] All tests passing (`python test_resume_integration.py`)
- [x] Dependencies in `requirements.txt`
- [x] `.streamlit/config.toml` configured
- [x] Code pushed to GitHub main branch

### After Deploying

- [ ] Verify app loads successfully
- [ ] Test profile dashboard displays your resume
- [ ] Test vector search with auto-filled resume
- [ ] Test ATS classifier with your resume
- [ ] Check all navigation works
- [ ] Verify job data loads (1000+ jobs)
- [ ] Test on mobile device

---

## 🌐 Custom Domain (Optional)

### Streamlit Cloud Custom Domain

1. Go to app settings in Streamlit Cloud
2. Click "Custom domain"
3. Add your domain (e.g., `jobs.yourdomain.com`)
4. Update DNS records as instructed
5. Wait for SSL certificate provisioning

### DNS Configuration

Add CNAME record:
```
jobs.yourdomain.com -> [your-app].streamlit.app
```

---

## 📈 Monitoring & Analytics

### Streamlit Cloud Metrics

Dashboard shows:
- Number of visitors
- Active users
- App uptime
- Error logs
- Resource usage

### Custom Analytics (Optional)

Add Google Analytics to `app.py`:
```python
import streamlit.components.v1 as components

# Google Analytics
components.html("""
    <!-- Global site tag (gtag.js) - Google Analytics -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-XXXXXXXXXX');
    </script>
""", height=0)
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"

**Problem**: Missing dependencies

**Solution**:
```bash
# Update requirements.txt
pip freeze > requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### Issue: "Resume not loading"

**Problem**: `data/resume.json` not found

**Solution**:
1. Verify file exists: `ls -la data/resume.json`
2. Ensure it's committed to Git: `git add data/resume.json`
3. Push to GitHub: `git push`

### Issue: "App not updating"

**Problem**: Changes not deployed

**Solution**:
1. Commit changes: `git commit -am "Update"`
2. Push to GitHub: `git push`
3. Streamlit Cloud auto-deploys from main branch
4. Or click "Reboot app" in Streamlit Cloud

### Issue: "Out of memory"

**Problem**: Too many jobs loaded

**Solution**:
```python
# In app.py, limit dataset size
jobs_df = jobs_df.head(1000)  # Limit to 1000 jobs
```

---

## 🔄 CI/CD Pipeline (Optional)

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python test_resume_integration.py
```

---

## 📊 Performance Optimization

### 1. Caching

Already implemented:
```python
@st.cache_data
def load_jobs_data():
    # Cached job data loading

@st.cache_resource
def load_resume():
    # Cached resume loading
```

### 2. Data Sampling

For faster loading:
```python
# Sample dataset for demo
if st.sidebar.checkbox("Use sample data"):
    jobs_df = jobs_df.sample(100)
```

### 3. Lazy Loading

Load data only when needed:
```python
if search_mode == "Vector Search":
    # Only initialize vector store when needed
    if st.session_state.vector_store is None:
        st.session_state.vector_store = initialize_vector_store(jobs_df)
```

---

## 🎯 Post-Deployment Tasks

### 1. Test the Live App
```bash
# Visit your app URL
https://[your-app].streamlit.app

# Test all features:
✅ Profile loads with your resume
✅ Vector search works
✅ ATS classifier functions
✅ Job browsing works
✅ Mobile responsive
```

### 2. Share Your App

Add to your:
- Resume/CV
- LinkedIn profile
- GitHub README
- Portfolio website

Example:
```markdown
**Live Demo**: [AI Job Intelligence Platform](https://your-app.streamlit.app)
```

### 3. Collect Feedback

Add feedback form (optional):
```python
st.sidebar.markdown("---")
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Share your thoughts:")
if st.sidebar.button("Submit"):
    # Save feedback
    st.success("Thank you!")
```

---

## 📱 Mobile Optimization

The app is already mobile-responsive! Test on:
- iPhone/iOS
- Android phones
- Tablets
- Different screen sizes

Streamlit handles responsiveness automatically.

---

## 🔗 Useful Links

- **Streamlit Cloud**: https://streamlit.io/cloud
- **Streamlit Docs**: https://docs.streamlit.io
- **Your Repository**: https://github.com/anix-lynch/ai-agent-job-intelligence-phase-2
- **Resume MCP**: https://github.com/anix-lynch/resume-mcp
- **Support**: https://discuss.streamlit.io

---

## 💰 Cost Breakdown

### Streamlit Cloud (Recommended)
- **Free tier**: ✅ Unlimited public apps
- **Pro tier**: $20/month for private apps
- **Enterprise**: Custom pricing

### Vercel (Alternative)
- **Hobby**: Free for personal projects
- **Pro**: $20/month

### Total Cost: **$0/month** (using free tiers)

---

## ✅ Deployment Complete!

Once deployed, your app will:
1. ✅ Auto-load your resume from `data/resume.json`
2. ✅ Display your profile dashboard
3. ✅ Pre-fill search forms with your data
4. ✅ Provide personalized job matching
5. ✅ Work on any device (mobile, tablet, desktop)
6. ✅ Auto-update when you push to GitHub

**Your personalized job intelligence platform is now live! 🚀**

---

## 📞 Support

Need help deploying?
- **GitHub Issues**: [Report issues](https://github.com/anix-lynch/ai-agent-job-intelligence-phase-2/issues)
- **Email**: alynch@gozeroshot.dev
- **Streamlit Forum**: https://discuss.streamlit.io

---

**Last Updated**: 2025-11-19  
**Version**: Phase 2.0