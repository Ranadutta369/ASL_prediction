<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/ONNX-005CED?style=flat&logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/Vercel-Deployed-000000?style=flat&logo=vercel" alt="Vercel">
  <img src="https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat&logo=render" alt="Render">
</div>

# 🤟 Sign0 — AI Sign Language Recognition & Synthesizer

> Bridging communication gaps using real-time AI vision, Spatial-Temporal Transformers, Text-to-Sign Diffusion Synthesizers, and accessible learning tools for speech-impaired individuals and children.

<div align="center">
  <h3>
    <a href="https://asl-prediction.vercel.app">🌐 Live Demo (Vercel)</a>
    <span> | </span>
    <a href="https://asl-prediction-v2.onrender.com">🔗 API Endpoint (Render)</a>
  </h3>
</div>

---

## 📌 Features & Platform Overview

- 🎥 **Real-Time AI Sign Recognizer:** Live webcam tracking using MediaPipe Hands and ONNX Multi-Layer Perceptron (MLP) with **0.81 ms latency**.
- 🧠 **Spatial-Temporal Transformer Model:** Multi-Head Self-Attention model tracking 3D landmark trajectories over time with **100% Validation Accuracy**.
- 🪄 **AI Text-to-Sign Diffusion Synthesizer:** Denoising Diffusion Probabilistic Model (DDPM) that synthesizes frame-by-frame 3D skeleton gesture animations from text prompts.
- 📚 **Interactive ASL Visual Dictionary:** Complete visual learning hub for alphabets (A–Z), daily phrases (*Hello*, *Thank You*, *Help*, *Water*), numbers (1–10), search, and audio pronunciation.
- ♿ **Assistive & Special Needs Guide:** High-contrast dark mode, OpenDyslexic typography mode, large touch targets, and educational resource guides.
- 🔊 **Text-to-Speech (TTS) Audio Engine:** Real-time voice translation pronouncing recognized letters and full constructed sentences.

---

## 🚀 Quick Start (Local Setup)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Agniva2006/ASL_prediction.git
cd ASL_prediction
```

### 2️⃣ Launch Backend API
The AI models are already pre-trained and exported to ONNX in the `backend/` directory.

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 3️⃣ Open Web Platform
Simply open the `frontend/home.html` file in your browser!

> **Note:** The frontend auto-detects a local backend on ports `8005`, `8001`, `8000`, or `8080`. If no local API is running, it seamlessly falls back to the deployed Render backend at `https://asl-prediction-v2.onrender.com`.

---

## 🏗️ Project Architecture

```text
ASL_prediction/
├── backend/
│   ├── app.py                      # Multi-Model FastAPI API (MLP + Transformer + Diffusion)
│   ├── asl_mlp.onnx                # Exported Static MLP ONNX weights
│   ├── asl_transformer.onnx        # Exported Spatial-Temporal Transformer ONNX weights
│   ├── sign_diffusion_library.json # Precomputed DDPM 3D animation trajectories
│   ├── label_map.json              # Class index mapping
│   └── requirements.txt            # Backend Python dependencies
├── frontend/
│   ├── home.html                   # Landing Page: Platform overview & navigation hub
│   ├── index.html                  # Page 1: Real-Time AI Sign Recognizer
│   ├── dictionary.html             # Page 2: ASL Visual Dictionary & Learning Hub
│   ├── generator.html              # Page 3: Text-to-Sign AI Diffusion Synthesizer
│   ├── accessibility.html          # Page 4: Special Needs & Assistive Guide
│   ├── script.js                   # MediaPipe tracking, client normalization, API logic
│   ├── generator.js                # 3D Skeleton animation player engine
│   └── style.css                   # Ultra-premium cosmic glassmorphism design system
├── asl_models.py                   # Shared neural network architecture definitions
├── train_all_models.py             # Master multi-model training & export orchestrator
├── train_robust_model.py           # PyTorch MLP training script
├── transformer_model.py            # Spatial-Temporal Transformer training script
├── sign_diffusion_model.py         # DDPM Denoising Diffusion training script
├── architecture_guide.md           # Technical mathematical specification guide
├── deployment_guide.md             # Complete deployment knowledge base
└── README.md
```

*(Optional: To re-train all models from scratch, run `python train_all_models.py` at the project root.)*

---

## 📜 Deployment

The project is fully configured for continuous deployment:
- **Frontend:** Vercel (via `vercel.json`).
- **Backend:** Render or Docker (via `render.yaml`, `Dockerfile`, and `Procfile`).

Refer to [`deployment_guide.md`](deployment_guide.md) for detailed, step-by-step instructions.

---

## 🧑‍💻 Author & License

**Agniva Ghosh**  
GitHub: [@Agniva2006](https://github.com/Agniva2006)  

This project is licensed under the [MIT License](LICENSE).
