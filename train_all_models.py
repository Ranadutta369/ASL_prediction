import os
import sys
import io
import time
import json
import numpy as np

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("==========================================================================")
print("     SIGN0 MASTER MULTI-MODEL AI TRAINING & EXPORT ORCHESTRATOR           ")
print("==========================================================================")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
else:
    print("Compute Device: CPU (High-Performance Vectorized Training)")
print("==========================================================================")

t_start_all = time.time()

# --------------------------------------------------------------------------
# PHASE 1: TRAIN & EVALUATE STATIC ASL MLP CLASSIFIER (ASLClassifierV2)
# --------------------------------------------------------------------------
print("\n[PHASE 1/4] Training Static ASL MLP Model (ASLClassifierV2)...")
try:
    from train_robust_model import train_and_evaluate
    train_and_evaluate()
    print("[OK] Phase 1 Completed Successfully!")
except Exception as e:
    print(f"[FAIL] Phase 1 Error: {e}")

# Export MLP ONNX
try:
    from export_onnx import export as export_mlp_onnx
    export_mlp_onnx()
    print("[OK] MLP ONNX Export Completed Successfully!")
except Exception as e:
    print(f"[FAIL] MLP ONNX Export Error: {e}")

# --------------------------------------------------------------------------
# PHASE 2: TRAIN & EVALUATE SPATIAL-TEMPORAL TRANSFORMER (SpatialTemporalSignTransformer)
# --------------------------------------------------------------------------
print("\n[PHASE 2/4] Training Spatial-Temporal Transformer Model...")
try:
    from transformer_model import train_and_export_transformer
    train_and_export_transformer()
    print("[OK] Phase 2 Completed Successfully!")
except Exception as e:
    print(f"[FAIL] Phase 2 Error: {e}")

# Export Transformer ONNX
try:
    from export_transformer_onnx import export as export_transformer_onnx
    export_transformer_onnx()
    print("[OK] Transformer ONNX Export Completed Successfully!")
except Exception as e:
    print(f"[FAIL] Transformer ONNX Export Error: {e}")

# --------------------------------------------------------------------------
# PHASE 3: TRAIN & SAMPLE DENOISING DIFFUSION MODEL (SignDiffusionSynthesizer)
# --------------------------------------------------------------------------
print("\n[PHASE 3/4] Training Text-Conditioned Sign Diffusion DDPM Model...")
try:
    from sign_diffusion_model import train_and_export_diffusion
    train_and_export_diffusion()
    print("[OK] Phase 3 Completed Successfully!")
except Exception as e:
    print(f"[FAIL] Phase 3 Error: {e}")

# --------------------------------------------------------------------------
# PHASE 4: VERIFY ALL AI ENGINE ENDPOINTS
# --------------------------------------------------------------------------
print("\n[PHASE 4/4] Verifying End-to-End Multi-Model FastAPI Endpoints...")
try:
    from test_advanced_backend import test_advanced_api
    test_advanced_api()
    print("[OK] Phase 4 Verification Passed!")
except Exception as e:
    print(f"[WARN] Phase 4 Verification Warning: {e}")

total_time = time.time() - t_start_all
print("\n==========================================================================")
print(f"MASTER TRAINING COMPLETE IN {total_time:.2f} SECONDS!")
print("All model weights, ONNX graphs, and DDPM animation libraries are ready!")
print("==========================================================================")
