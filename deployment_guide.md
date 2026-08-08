# 🌐 Sign0 Platform — Complete Deployment Knowledge Base & Guide

A comprehensive, step-by-step technical guide for deploying both the **FastAPI Backend AI Service** and the **Multi-Page Frontend Web Suite** to production cloud platforms (Vercel, Render, Railway, Hugging Face Spaces, and Docker).

---

## 🎯 Architecture Deployment Overview

```mermaid
flowchart LR
    subgraph Frontend Cloud (Vercel / Netlify / GitHub Pages)
        F1[index.html]
        F2[dictionary.html]
        F3[generator.html]
        F4[accessibility.html]
    end

    subgraph Backend Cloud (Render / Railway / HF Spaces / Docker)
        API[FastAPI Server Engine]
        MLP[asl_mlp.onnx]
        STT[asl_transformer.onnx]
        DDPM[sign_diffusion_library.json]
    end

    F1 -->|HTTPS POST| API
    F3 -->|HTTPS POST| API
```

---

## 🚀 1. Backend Deployment Guide (FastAPI + ONNX Runtime)

### Option A: Deploy on Render (Recommended — Free & Automated)
1. **Push Repo to GitHub:** Ensure all files are committed to your GitHub repository `Agniva2006/ASL_prediction`.
2. **Log into Render:** Go to [render.com](https://render.com) and click **New + $\rightarrow$ Web Service**.
3. **Connect Repository:** Select `Agniva2006/ASL_prediction`.
4. **Configuration Options:**
   - **Name:** `asl-prediction-backend`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables:**
     - `KMP_DUPLICATE_LIB_OK`: `TRUE`
5. **Deploy:** Click **Create Web Service**. Your API will be live at `https://asl-prediction-v2.onrender.com`.

---

### Option B: Deploy via Docker (Any Cloud — AWS, GCP, Azure, DigitalOcean)
1. **Build Docker Image:**
   ```bash
   docker build -t asl-prediction-backend:latest .
   ```
2. **Run Container Locally / Server:**
   ```bash
   docker run -d -p 8000:8000 --name asl-backend asl-prediction-backend:latest
   ```
3. **Verify Health Endpoint:**
   ```bash
   curl http://localhost:8000/
   ```

---

## 🌐 2. Frontend Deployment Guide (Vercel / Netlify / GitHub Pages)

### Option A: Deploy on Vercel (Recommended — Instant 1-Click Setup)
1. **Install Vercel CLI (Optional) or Log in via Web:** Go to [vercel.com](https://vercel.com).
2. **Import Project:** Click **Add New $\rightarrow$ Project**, then select `Agniva2006/ASL_prediction`.
3. **Configure Project Settings:**
   - **Framework Preset:** `Other` / `Static HTML`
   - **Root Directory:** `./` or `frontend`
4. **Deploy:** Click **Deploy**. Vercel automatically deploys all pages (`index.html`, `dictionary.html`, `generator.html`, `accessibility.html`) with SSL HTTPS enabled.

---

### Option B: Deploy on GitHub Pages
1. Navigate to your GitHub repo settings: `https://github.com/Agniva2006/ASL_prediction/settings/pages`.
2. Under **Build and deployment $\rightarrow$ Source**, select `Deploy from a branch`.
3. Select `main` branch and folder `/frontend` or `/root`.
4. Save. Your frontend will be live at `https://agniva2006.github.io/ASL_prediction/`.

---

## 🔗 3. Connecting Frontend to Production Backend

Once your backend is deployed (e.g. at `https://asl-prediction-v2.onrender.com`), update `REMOTE_API_BASE` in `frontend/script.js` and `frontend/generator.js`:

```javascript
const REMOTE_API_BASE = "https://asl-prediction-v2.onrender.com";
```

The auto-discovery mechanism will seamlessly route client calls to the live production endpoint whenever local backend servers are offline!
