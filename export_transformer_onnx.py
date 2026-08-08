import os
import torch
import torch.nn as nn
from transformer_model import SpatialTemporalSignTransformer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def export():
    device = torch.device("cpu")
    model = SpatialTemporalSignTransformer(input_dim=63, d_model=128, nhead=4, num_layers=3, num_classes=26, max_len=30)
    
    if os.path.exists("best_transformer_model.pth"):
        state_dict = torch.load("best_transformer_model.pth", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        
    model.eval()

    dummy_input = torch.randn(1, 16, 63, dtype=torch.float32)
    onnx_path = os.path.join("backend", "asl_transformer.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["sequence_keypoints"],
        output_names=["logits"],
        dynamic_axes={
            "sequence_keypoints": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        dynamo=False
    )
    print(f"SUCCESS: Spatial-Temporal Transformer exported to {onnx_path}")

if __name__ == "__main__":
    export()
