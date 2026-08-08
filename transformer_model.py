import os
import sys
import math
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================
# 1. POSITIONAL ENCODING MODULE
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# ==========================================
# 2. SPATIAL-TEMPORAL SIGN TRANSFORMER
# ==========================================
class SpatialTemporalSignTransformer(nn.Module):
    """
    Multi-Head Self-Attention Spatial-Temporal Transformer for dynamic sequence sign recognition.
    Input: Keypoint trajectory tensor (Batch, Seq_Len, 63)
    Output: Logits tensor (Batch, NumClasses)
    """
    def __init__(self, input_dim=63, d_model=128, nhead=4, num_layers=3, num_classes=26, max_len=60, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, 63)
        h = self.input_projection(x) # (batch_size, seq_len, d_model)
        h = self.pos_encoder(h)
        out = self.transformer_encoder(h) # (batch_size, seq_len, d_model)
        
        # Temporal Average Pooling over sequence length
        pooled = out.mean(dim=1) # (batch_size, d_model)
        logits = self.head(pooled)
        return logits

# ==========================================
# 3. SEQUENCE SYNTHETIC & KEYPOINT DATASET
# ==========================================
class SignSequenceDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx]

def generate_sequence_dataset(num_samples_per_class=100, seq_len=16):
    """
    Generates temporal keypoint sequences simulating dynamic hand gestures for 26 classes.
    """
    X_sequences, y_labels = [], []
    
    for class_idx in range(26):
        # Base keypoints
        base = np.random.uniform(-0.4, 0.4, size=(21, 3)).astype(np.float32)
        base[0] = [0, 0, 0] # Wrist
        
        for _ in range(num_samples_per_class):
            seq = []
            # Motion trajectories
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, math.pi)
            
            for t in range(seq_len):
                factor = math.sin(t * 0.2 * freq + phase) * 0.1
                frame_kp = base.copy()
                frame_kp[1:] += factor
                seq.append(frame_kp.reshape(63))
                
            X_sequences.append(np.array(seq, dtype=np.float32))
            y_labels.append(class_idx)
            
    return np.array(X_sequences), np.array(y_labels)

# ==========================================
# 4. TRAINING & ONNX EXPORT
# ==========================================
def train_and_export_transformer():
    print("--> Generating Temporal Sequence Dataset...")
    X_seq, y = generate_sequence_dataset(num_samples_per_class=120, seq_len=16)
    print(f"Sequence Dataset Shape: X={X_seq.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_loader = DataLoader(SignSequenceDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(SignSequenceDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Spatial-Temporal Transformer on: {device}")
    
    model = SpatialTemporalSignTransformer(input_dim=63, d_model=128, nhead=4, num_layers=3, num_classes=26, max_len=30).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 25
    print("\n--> Starting Transformer Training...")
    start_t = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            correct += (logits.argmax(1) == by).sum().item()
            total += bx.size(0)
            
        train_acc = correct / total
        
        # Test eval
        model.eval()
        vcorrect, vtotal = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                vcorrect += (logits.argmax(1) == by).sum().item()
                vtotal += bx.size(0)
                
        val_acc = vcorrect / vtotal
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")
            
    print(f"\nTransformer Training Complete in {time.time() - start_t:.2f}s! Final Val Acc: {val_acc*100:.2f}%")
    
    # Save checkpoint
    torch.save(model.state_dict(), "best_transformer_model.pth")
    
    # ONNX Export
    print("--> Exporting Transformer model to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 16, 63).to(device)
    backend_onnx_path = os.path.join("backend", "asl_transformer.onnx")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            backend_onnx_path,
            input_names=["sequence_keypoints"],
            output_names=["logits"],
            dynamic_axes={
                "sequence_keypoints": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"}
            },
            opset_version=14,
            dynamo=False
        )
        print(f"SUCCESS: Transformer ONNX exported to {backend_onnx_path}")
    except Exception as e:
        print(f"ONNX export warning: {e}")

if __name__ == "__main__":
    train_and_export_transformer()
