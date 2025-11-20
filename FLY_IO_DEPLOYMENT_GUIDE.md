# 🚀 Fly.io Deployment Best Practices Guide

Complete guide for deploying Streamlit/Python apps to Fly.io with authentication, optimization, and troubleshooting.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Authentication Setup](#authentication-setup)
3. [Image Size Optimization](#image-size-optimization)
4. [Deployment Configuration](#deployment-configuration)
5. [Deployment Process](#deployment-process)
6. [Troubleshooting](#troubleshooting)
7. [Cost Optimization](#cost-optimization)

---

## Prerequisites

### Install Fly CLI

```bash
# macOS
brew install flyctl

# Linux
curl -L https://fly.io/install.sh | sh

# Windows
iwr https://fly.io/install.ps1 -useb | iex
```

### Verify Installation

```bash
fly version
```

---

## Authentication Setup

### Method 1: Interactive Login (Recommended for First Time)

```bash
fly auth login
```

This opens a browser window for authentication.

### Method 2: API Token (For CI/CD & Automation)

#### Step 1: Generate Token

1. Go to https://fly.io/user/personal_access_tokens
2. Click "Create Access Token"
3. Name it (e.g., "CLI Deployment Token")
4. Copy the token (starts with `FlyV1` or `fm2_`)

#### Step 2: Store Token Securely

**Option A: Environment Variable (Temporary)**
```bash
export FLY_ACCESS_TOKEN="FlyV1 your_token_here"
```

**Option B: Persistent Storage**
```bash
# Add to ~/.tokens file
echo 'FLY_ACCESS_TOKEN="FlyV1 your_token_here"' >> ~/.tokens

# Source it in your shell profile (~/.zshrc or ~/.bashrc)
echo 'source ~/.tokens' >> ~/.zshrc
```

**Option C: Shell Profile (Permanent)**
```bash
# Add to ~/.zshrc or ~/.bashrc
echo 'export FLY_ACCESS_TOKEN="FlyV1 your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

#### Step 3: Verify Authentication

```bash
fly auth whoami
```

Expected output:
```
your-email@example.com
```

---

## Image Size Optimization

### ⚠️ Critical: Fly.io Free Tier Limits

- **Maximum uncompressed image size**: 8 GB
- **Recommended compressed size**: < 500 MB

### Common Issues with ML/AI Apps

Heavy dependencies that cause size issues:
- `sentence-transformers` (~2 GB)
- `chromadb` (~1.5 GB)
- `faiss-cpu` (~500 MB)
- `langchain` with all dependencies (~1 GB)
- `torch` / `tensorflow` (~2-4 GB each)

### Solution: Create Lightweight Requirements

#### Step 1: Create `requirements-fly.txt`

```txt
# Lightweight requirements for Fly.io deployment
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.15.0
requests>=2.31.0
python-dotenv>=1.0.0
pydantic>=2.4.0
```

**Exclude heavy ML libraries:**
- ❌ `sentence-transformers`
- ❌ `chromadb`
- ❌ `faiss-cpu`
- ❌ `langchain`
- ❌ `torch`
- ❌ `tensorflow`

#### Step 2: Update Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy lightweight requirements for Fly.io
COPY requirements-fly.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Step 3: Make Code Conditional

Update your Python code to handle missing dependencies gracefully:

```python
# Conditional imports for ML features
try:
    from ml.vector_store import VectorStore
    from ml.classifier import ATSClassifier
    ML_FEATURES_AVAILABLE = True
except ImportError:
    ML_FEATURES_AVAILABLE = False
    VectorStore = None
    ATSClassifier = None

# Later in your code
if ML_FEATURES_AVAILABLE:
    # Use ML features
    vector_store = VectorStore()
else:
    # Fallback or disable feature
    st.warning("ML features not available in this deployment")
```

---

## Deployment Configuration

### Create `fly.toml`

```toml
app = "your-app-name"
primary_region = "lax"  # Choose closest region: lax, iad, fra, etc.

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8501
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
  processes = ["app"]

[[vm]]
  memory = '1gb'
  cpu_kind = 'shared'
  cpus = 1
```

### Key Configuration Options

| Option | Description | Recommendation |
|--------|-------------|----------------|
| `primary_region` | Closest data center | `lax` (LA), `iad` (Virginia), `fra` (Frankfurt) |
| `internal_port` | App port | `8501` for Streamlit, `8080` for generic |
| `auto_stop_machines` | Stop when idle | `true` for free tier |
| `min_machines_running` | Always-on instances | `0` for free tier, `1+` for production |
| `memory` | RAM allocation | `256mb` (min), `1gb` (recommended) |

---

## Deployment Process

### Initial Deployment

```bash
# 1. Authenticate
export FLY_ACCESS_TOKEN="FlyV1 your_token_here"
fly auth whoami

# 2. Navigate to project
cd /path/to/your/project

# 3. Create fly.toml (if not exists)
# Manually create the file or use fly launch (interactive)

# 4. Deploy
fly deploy

# 5. Check status
fly status

# 6. View logs
fly logs
```

### Subsequent Deployments

```bash
# Set token (if not in environment)
export FLY_ACCESS_TOKEN="FlyV1 your_token_here"

# Deploy
fly deploy

# Monitor
fly logs --follow
```

### Deployment Workflow

```bash
# Full automated deployment script
#!/bin/bash

# Set credentials
export FLY_ACCESS_TOKEN="FlyV1 your_token_here"

# Verify auth
fly auth whoami || exit 1

# Deploy
fly deploy || exit 1

# Check status
fly status

# Open in browser
fly open
```

---

## Troubleshooting

### Issue 1: "Not enough space to unpack image"

**Error:**
```
Not enough space to unpack image, possibly exceeds maximum of 8GB uncompressed
```

**Solution:**
1. Create `requirements-fly.txt` with lightweight dependencies
2. Update `Dockerfile` to use it
3. Remove heavy ML libraries
4. Redeploy

**Check image size:**
```bash
fly deploy --build-only
# Look for "image size: XXX MB" in output
```

### Issue 2: "You must be authenticated"

**Error:**
```
Error: failed retrieving current user: You must be authenticated to view this.
```

**Solution:**
```bash
# Check if token is set
echo $FLY_ACCESS_TOKEN

# If empty, set it
export FLY_ACCESS_TOKEN="FlyV1 your_full_token_here"

# Verify
fly auth whoami
```

**Note:** The token format can be:
- `FlyV1 fm2_...` (full format with prefix)
- `fm2_...` (without prefix)

Use the **complete token string** as provided by Fly.io.

### Issue 3: "ModuleNotFoundError" After Deployment

**Error:**
```
ModuleNotFoundError: No module named 'chromadb'
```

**Solution:**
Your code is importing a module not in `requirements-fly.txt`. Either:

1. **Add conditional imports** (recommended):
```python
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
```

2. **Add to requirements** (if size permits):
```txt
chromadb>=0.4.15
```

### Issue 4: App Not Starting

**Check logs:**
```bash
fly logs --follow
```

**Common issues:**
- Wrong port (must match `internal_port` in `fly.toml`)
- Missing environment variables
- Startup timeout (increase in `fly.toml`)

**Fix port:**
```dockerfile
# Ensure Streamlit uses correct port
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Issue 5: "machine getting replaced, refusing to start"

**Error:**
```
[PM07] failed to change machine state: machine getting replaced, refusing to start
```

**Solution:**
This happens during deployment. Wait for deployment to complete:
```bash
fly status
# Wait until status shows "running"
```

---

## Cost Optimization

### Free Tier Limits

- **Compute**: 3 shared-cpu-1x VMs with 256MB RAM
- **Storage**: 3GB persistent volume storage
- **Bandwidth**: 160GB outbound data transfer

### Optimization Strategies

#### 1. Auto-Stop Machines

```toml
[http_service]
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
```

**Benefit:** Machines stop when idle, restart on request (free tier friendly)

#### 2. Reduce Image Size

- Use `python:3.10-slim` instead of `python:3.10`
- Remove unnecessary dependencies
- Use `--no-cache-dir` in pip install
- Multi-stage builds for compiled languages

#### 3. Optimize Memory

```toml
[[vm]]
  memory = '256mb'  # Minimum for simple apps
  # memory = '512mb'  # For moderate apps
  # memory = '1gb'    # For heavy apps
```

#### 4. Choose Closest Region

Reduces latency and bandwidth costs:
- **US West**: `lax`, `sjc`
- **US East**: `iad`, `ewr`
- **Europe**: `fra`, `ams`, `lhr`
- **Asia**: `nrt`, `sin`, `hkg`

---

## Best Practices Summary

### ✅ Do's

1. **Use API tokens** for automation and CI/CD
2. **Create lightweight requirements** for Fly.io deployments
3. **Make imports conditional** to handle missing dependencies
4. **Set `auto_stop_machines = true`** for free tier
5. **Monitor image size** (keep under 500 MB compressed)
6. **Use health checks** in Dockerfile
7. **Store tokens securely** (environment variables, not in code)
8. **Test locally** before deploying

### ❌ Don'ts

1. **Don't commit tokens** to Git
2. **Don't include heavy ML libraries** unless necessary
3. **Don't use `min_machines_running > 0`** on free tier
4. **Don't ignore image size warnings**
5. **Don't hardcode credentials** in code
6. **Don't skip health checks**

---

## Quick Reference Commands

```bash
# Authentication
fly auth login                          # Interactive login
fly auth whoami                         # Check current user
fly auth token                          # Get current token

# Deployment
fly deploy                              # Deploy app
fly deploy --build-only                 # Build without deploying
fly deploy --remote-only                # Build remotely

# Management
fly apps list                           # List all apps
fly status                              # Check app status
fly logs                                # View logs
fly logs --follow                       # Stream logs
fly open                                # Open app in browser

# Scaling
fly scale count 1                       # Set instance count
fly scale memory 512                    # Set memory (MB)
fly scale vm shared-cpu-1x              # Set VM type

# Monitoring
fly status                              # App status
fly dashboard                           # Open web dashboard
fly monitor                             # Real-time metrics

# Cleanup
fly apps destroy <app-name>             # Delete app
fly machines list                       # List machines
fly machines stop <machine-id>          # Stop machine
```

---

## Example: Complete Deployment Workflow

```bash
#!/bin/bash
# deploy-to-fly.sh

set -e  # Exit on error

echo "🚀 Starting Fly.io deployment..."

# 1. Set credentials
export FLY_ACCESS_TOKEN="FlyV1 your_token_here"

# 2. Verify authentication
echo "📝 Verifying authentication..."
fly auth whoami || {
    echo "❌ Authentication failed"
    exit 1
}

# 3. Check app exists
echo "🔍 Checking app status..."
fly status || {
    echo "⚠️  App doesn't exist, creating..."
    fly launch --no-deploy
}

# 4. Deploy
echo "📦 Building and deploying..."
fly deploy || {
    echo "❌ Deployment failed"
    exit 1
}

# 5. Verify deployment
echo "✅ Checking deployment status..."
fly status

# 6. Show URL
APP_URL=$(fly status --json | jq -r '.Hostname')
echo "🎉 Deployment successful!"
echo "🌐 App URL: https://${APP_URL}"

# 7. Open in browser (optional)
# fly open
```

---

## Additional Resources

- **Fly.io Docs**: https://fly.io/docs/
- **Fly.io Dashboard**: https://fly.io/dashboard
- **Fly.io Pricing**: https://fly.io/docs/about/pricing/
- **Fly.io Status**: https://status.flyio.net/
- **Community Forum**: https://community.fly.io/

---

**Last Updated**: 2025-11-19  
**Version**: 1.0  
**Author**: AI Agent Deployment Guide
