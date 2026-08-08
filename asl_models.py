"""
Sign0 Neural Network Model Architectures
Shared module for model class definitions used by training, export, and inference.
"""
import torch.nn as nn


class ASLClassifierV2(nn.Module):
    """
    Deep MLP classifier for static ASL hand landmark recognition.
    Input: 63-dim flattened keypoints (21 joints × 3 coords)
    Output: 26-class logits (A-Z)
    """
    def __init__(self, input_dim=63, num_classes=26):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.block1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        self.head = nn.Linear(128, num_classes)
        
    def forward(self, x):
        h = self.in_proj(x)
        h = h + self.block1(h)  # Skip connection
        h2 = self.block2(h)
        return self.head(h2)
