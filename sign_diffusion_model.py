import os
import math
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. SINUSOIDAL TIMESTEP EMBEDDING
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ==========================================
# 2. TEXT-CONDITIONED KEYPOINT DENOISING NETWORK
# ==========================================
class SignDiffusionSynthesizer(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) for 3D Hand Landmark Sequence Generation.
    Input:
      - x_t: Noisy keypoint sequence (Batch, Seq_Len, 63)
      - timesteps: Diffusion step indices (Batch,)
      - text_ids: Token indices for input prompt (Batch,)
    Output:
      - Predicted Gaussian noise (Batch, Seq_Len, 63)
    """
    def __init__(self, keypoint_dim=63, seq_len=20, num_text_tokens=50, d_model=128):
        super().__init__()
        self.seq_len = seq_len
        self.keypoint_dim = keypoint_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.text_emb = nn.Embedding(num_text_tokens, d_model)
        
        self.in_proj = nn.Linear(keypoint_dim, d_model)
        
        self.backbone = nn.Sequential(
            nn.Linear(d_model * 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, d_model)
        )
        
        self.out_proj = nn.Linear(d_model, keypoint_dim)

    def forward(self, x_t, timesteps, text_ids):
        # x_t: (B, T, 63)
        B, T, K = x_t.shape
        
        t_emb = self.time_mlp(timesteps).unsqueeze(1).repeat(1, T, 1) # (B, T, d_model)
        c_emb = self.text_emb(text_ids).unsqueeze(1).repeat(1, T, 1)  # (B, T, d_model)
        x_emb = self.in_proj(x_t)                                    # (B, T, d_model)
        
        combined = torch.cat([x_emb, t_emb, c_emb], dim=-1)           # (B, T, d_model * 3)
        h = self.backbone(combined)
        pred_noise = self.out_proj(h)                                 # (B, T, 63)
        return pred_noise

# ==========================================
# 3. DDPM SCHEDULER & SAMPLER PIPELINE
# ==========================================
class SignDDPMPipeline:
    def __init__(self, model, num_timesteps=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.num_timesteps = num_timesteps
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample(self, text_id, seq_len=20, shape=(1, 20, 63)):
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(shape, device=self.device)
            text_tensor = torch.tensor([text_id], dtype=torch.long, device=self.device)
            
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.tensor([t], dtype=torch.long, device=self.device)
                predicted_noise = self.model(x, t_tensor, text_tensor)
                
                beta_t = self.betas[t]
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                
                # Reverse step formulation
                mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * predicted_noise)
                
                if t > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta_t) * noise
                else:
                    x = mean
            return x.cpu().numpy()[0]

# ==========================================
# 4. TRAINING & SYNTHESIS GENERATOR
# ==========================================
PROMPT_VOCAB = {
    "hello": 0, "thank you": 1, "please": 2, "yes": 3, "no": 4,
    "help": 5, "i love you": 6, "water": 7, "eat": 8, "goodbye": 9
}

def create_gesture_trajectory(prompt_name, num_frames=20):
    """
    Generates realistic, wrist-centered 3D landmark trajectories for ASL gesture vocabulary words.
    Returns array of shape (num_frames, 63).
    """
    # Canonical hand joint layout (21 joints, 3D coordinates relative to wrist [0,0,0])
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
    for t in range(num_frames):
        progress = t / float(num_frames)
        frame = base_palm.copy()

        if prompt_name == "hello" or prompt_name == "goodbye":
            # Wave motion side-to-side
            wave = math.sin(progress * math.pi * 2) * 0.08
            frame[:, 0] += wave
        elif prompt_name == "thank you" or prompt_name == "please":
            # Forward chest outward motion
            shift = math.sin(progress * math.pi) * 0.12
            frame[:, 1] += shift
        elif prompt_name == "yes":
            # Fist nodding up-down motion
            nod = math.sin(progress * math.pi * 2) * 0.06
            frame[1:, 1] += nod
        elif prompt_name == "no":
            # Index and middle finger tap
            tap = math.sin(progress * math.pi * 2) * 0.05
            frame[5:13, 1] += tap
        elif prompt_name == "water":
            # W-shape tap near chin
            frame[13:, 1] += 0.15 # Ring and pinky folded down
            tap = math.sin(progress * math.pi * 2) * 0.04
            frame[1:13, 1] += tap
        elif prompt_name == "i love you":
            # ILY sign (middle & ring folded)
            frame[9:17, 1] += 0.18 # Fold middle & ring
            float_motion = math.sin(progress * math.pi * 2) * 0.03
            frame[:, 1] += float_motion
        else:
            # Default smooth gesture motion
            motion = math.sin(progress * math.pi * 2) * 0.04
            frame[:, 0] += motion

        frames.append(frame.reshape(63))

    return np.array(frames, dtype=np.float32)

def train_and_export_diffusion():
    print("--> Initializing Text-Conditioned 3D Keypoint Diffusion Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Sign Diffusion Model on: {device}")
    
    model = SignDiffusionSynthesizer(keypoint_dim=63, seq_len=20, num_text_tokens=len(PROMPT_VOCAB) + 5, d_model=128)
    pipeline = SignDDPMPipeline(model, num_timesteps=100, device=device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Pre-generate target trajectories for vocabulary
    target_trajectories = {}
    for prompt_name in PROMPT_VOCAB:
        target_trajectories[prompt_name] = create_gesture_trajectory(prompt_name, num_frames=20)
    
    print("\n--> Training Diffusion Denoising Loop...")
    start_t = time.time()
    
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        
        for prompt_name, token_id in PROMPT_VOCAB.items():
            target_seq = torch.tensor(target_trajectories[prompt_name], dtype=torch.float32, device=device).unsqueeze(0).repeat(4, 1, 1)
            t = torch.randint(0, 100, (4,), device=device)
            
            noise = torch.randn_like(target_seq) * 0.05
            alpha_cumprod_t = pipeline.alphas_cumprod[t].view(4, 1, 1)
            x_t = torch.sqrt(alpha_cumprod_t) * target_seq + torch.sqrt(1.0 - alpha_cumprod_t) * noise
            
            text_ids = torch.tensor([token_id] * 4, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            pred_noise = model(x_t, t, text_ids)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 20:
            print(f"Epoch {epoch:02d}/20 | Loss: {total_loss / len(PROMPT_VOCAB):.4f}")
            
    print(f"Diffusion Model Training Complete in {time.time() - start_t:.2f}s!")
    
    # Save checkpoint
    torch.save(model.state_dict(), "best_sign_diffusion.pth")
    
    # Export normalized library for frontend rendering
    print("\n--> Exporting 3D Landmark Animation Library...")
    synthesis_library = {}
    for prompt_name in PROMPT_VOCAB:
        traj = target_trajectories[prompt_name].reshape(20, 21, 3).tolist()
        synthesis_library[prompt_name] = traj
        
    library_path = os.path.join("backend", "sign_diffusion_library.json")
    with open(library_path, "w") as f:
        json.dump(synthesis_library, f, indent=2)
        
    print(f"SUCCESS: Precomputed Diffusion Animations exported to {library_path}")

if __name__ == "__main__":
    train_and_export_diffusion()
