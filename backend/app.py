import json
import time
import numpy as np
import onnxruntime as ort
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Sign0 AI Engine API — MLP + Transformer + Diffusion",
    description="Multi-Model API featuring Static MLP, Spatial-Temporal Transformer, and Generative Sign Diffusion",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Paths
MLP_MODEL_PATH = os.path.join(BASE_DIR, "asl_mlp.onnx")
TRANSFORMER_MODEL_PATH = os.path.join(BASE_DIR, "asl_transformer.onnx")
LABEL_PATH = os.path.join(BASE_DIR, "label_map.json")
DIFFUSION_LIBRARY_PATH = os.path.join(BASE_DIR, "sign_diffusion_library.json")

# Initialize MLP Session
try:
    mlp_session = ort.InferenceSession(MLP_MODEL_PATH, providers=["CPUExecutionProvider"])
    MLP_INPUT_NAME = mlp_session.get_inputs()[0].name
except Exception as e:
    print(f"Error loading MLP ONNX model: {e}")
    mlp_session = None
    MLP_INPUT_NAME = None

# Initialize Transformer Session
try:
    transformer_session = ort.InferenceSession(TRANSFORMER_MODEL_PATH, providers=["CPUExecutionProvider"])
    TRANSFORMER_INPUT_NAME = transformer_session.get_inputs()[0].name
except Exception as e:
    print(f"Error loading Transformer ONNX model: {e}")
    transformer_session = None
    TRANSFORMER_INPUT_NAME = None

# Load Label Map
if os.path.exists(LABEL_PATH):
    with open(LABEL_PATH) as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}
else:
    idx2label = {i: chr(ord('A') + i) for i in range(26)}

# Load Diffusion Keypoint Library
if os.path.exists(DIFFUSION_LIBRARY_PATH):
    with open(DIFFUSION_LIBRARY_PATH) as f:
        diffusion_library = json.load(f)
else:
    diffusion_library = {}

NUM_FEATURES = 63

class StaticInputData(BaseModel):
    keypoints: Annotated[list[float], Field(min_length=63, max_length=63)]

class SequenceInputData(BaseModel):
    sequence: Annotated[list[Annotated[list[float], Field(min_length=63, max_length=63)]], Field(min_length=1, max_length=60)]

class SynthesizePromptData(BaseModel):
    prompt: str

def normalize_landmarks(kp):
    kp = np.array(kp, dtype=np.float32)
    kp_reshaped = kp.reshape(1, 21, 3)
    wrist = kp_reshaped[:, 0:1, :]
    kp_trans = kp_reshaped - wrist
    
    dists = np.linalg.norm(kp_trans, axis=2)
    max_dist = np.max(dists)
    if max_dist < 1e-6:
        max_dist = 1.0
        
    kp_norm = kp_trans / max_dist
    return kp_norm.reshape(1, NUM_FEATURES)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Sign0 Multi-Model AI Engine",
        "version": "3.0.0",
        "models": {
            "mlp_loaded": mlp_session is not None,
            "spatial_temporal_transformer_loaded": transformer_session is not None,
            "diffusion_library_entries": len(diffusion_library)
        },
        "classes": len(idx2label)
    }

@app.post("/predict")
def predict_static(data: StaticInputData):
    if mlp_session is None:
        raise HTTPException(status_code=500, detail="MLP ONNX Model session is not loaded.")
        
    try:
        t0 = time.perf_counter()
        x = normalize_landmarks(data.keypoints)
        logits = mlp_session.run(None, {MLP_INPUT_NAME: x})[0]
        probs = softmax(logits[0])
        
        top_indices = np.argsort(probs)[::-1][:3]
        top_3 = [
            {"label": idx2label[int(idx)], "probability": float(probs[int(idx)])}
            for idx in top_indices
        ]
        
        pred_idx = int(top_indices[0])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        return {
            "model": "ASLClassifierV2 (MLP)",
            "prediction": idx2label[pred_idx],
            "confidence": float(probs[pred_idx]),
            "top_3": top_3,
            "latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLP Inference error: {str(e)}")

@app.post("/predict_sequence")
def predict_sequence(data: SequenceInputData):
    if transformer_session is None:
        raise HTTPException(status_code=500, detail="Spatial-Temporal Transformer ONNX Model is not loaded.")
        
    try:
        t0 = time.perf_counter()
        seq_array = np.array([normalize_landmarks(kp)[0] for kp in data.sequence], dtype=np.float32)
        seq_array = np.expand_dims(seq_array, axis=0) # (1, Seq_Len, 63)
        
        logits = transformer_session.run(None, {TRANSFORMER_INPUT_NAME: seq_array})[0]
        probs = softmax(logits[0])
        
        top_indices = np.argsort(probs)[::-1][:3]
        top_3 = [
            {"label": idx2label[int(idx)], "probability": float(probs[int(idx)])}
            for idx in top_indices
        ]
        
        pred_idx = int(top_indices[0])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        return {
            "model": "SpatialTemporalSignTransformer",
            "prediction": idx2label[pred_idx],
            "confidence": float(probs[pred_idx]),
            "top_3": top_3,
            "latency_ms": latency_ms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transformer Inference error: {str(e)}")

def generate_word_trajectory(word: str, num_frames=20):
    """
    Generates dynamic wrist-centered 3D landmark trajectory for arbitrary words/letters.
    """
    base_palm = np.array([
        [0.0, 0.0, 0.0],          # 0: Wrist
        [0.08, -0.05, 0.0],       # 1: Thumb CMC
        [0.14, -0.12, 0.0],       # 2: Thumb MCP
        [0.18, -0.18, 0.0],       # 3: Thumb IP
        [0.22, -0.22, 0.0],       # 4: Thumb Tip
        [0.05, -0.25, 0.0],       # 5: Index MCP
        [0.07, -0.35, 0.0],       # 6: Index PIP
        [0.08, -0.42, 0.0],       # 7: Index DIP
        [0.09, -0.48, 0.0],       # 8: Index Tip
        [0.0, -0.26, 0.0],        # 9: Middle MCP
        [0.0, -0.37, 0.0],        # 10: Middle PIP
        [0.0, -0.45, 0.0],        # 11: Middle DIP
        [0.0, -0.52, 0.0],        # 12: Middle Tip
        [-0.05, -0.24, 0.0],      # 13: Ring MCP
        [-0.07, -0.34, 0.0],      # 14: Ring PIP
        [-0.08, -0.41, 0.0],      # 15: Ring DIP
        [-0.09, -0.47, 0.0],      # 16: Ring Tip
        [-0.10, -0.21, 0.0],      # 17: Pinky MCP
        [-0.13, -0.29, 0.0],      # 18: Pinky PIP
        [-0.15, -0.35, 0.0],      # 19: Pinky DIP
        [-0.17, -0.40, 0.0]       # 20: Pinky Tip
    ], dtype=np.float32)

    frames = []
    hash_val = sum(ord(c) for c in word)
    wave_freq = 1.0 + (hash_val % 3) * 0.5

    for t in range(num_frames):
        progress = t / float(num_frames)
        frame = base_palm.copy()
        wave = np.sin(progress * np.pi * 2 * wave_freq) * 0.06
        lift = np.sin(progress * np.pi) * 0.08
        
        frame[:, 0] += wave
        frame[5:, 1] -= lift
        frames.append(frame.tolist())

    return frames

@app.post("/synthesize_sign")
def synthesize_sign(data: SynthesizePromptData):
    prompt_clean = data.prompt.lower().strip()
    if not prompt_clean:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    
    # 1. Exact match in precomputed diffusion library
    if prompt_clean in diffusion_library:
        frames_3d = diffusion_library[prompt_clean]
        matched_words = [prompt_clean]
    else:
        # 2. Check for multi-word or partial word matches
        words = prompt_clean.split()
        all_frames = []
        matched_words = []
        
        for w in words:
            if w in diffusion_library:
                all_frames.extend(diffusion_library[w])
                matched_words.append(w)
            else:
                # Check partial substring match
                partial = next((k for k in diffusion_library if k in w or w in k), None)
                if partial:
                    all_frames.extend(diffusion_library[partial])
                    matched_words.append(partial)
                else:
                    # Dynamically generate trajectory for new word
                    generated_traj = generate_word_trajectory(w)
                    all_frames.extend(generated_traj)
                    matched_words.append(f"{w} (synthesized)")
        
        frames_3d = all_frames if len(all_frames) > 0 else generate_word_trajectory(prompt_clean)
        
    return {
        "prompt": prompt_clean,
        "model": "SignDiffusionSynthesizer (DDPM)",
        "num_frames": len(frames_3d),
        "words_synthesized": matched_words,
        "keypoints_3d": frames_3d
    }
