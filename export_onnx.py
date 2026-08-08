import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from asl_models import ASLClassifierV2

def export():
    device = torch.device("cpu")
    model = ASLClassifierV2()
    state_dict = torch.load("best_asl_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 63, dtype=torch.float32)
    onnx_path = os.path.join("backend", "asl_mlp.onnx")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["keypoints"],
            output_names=["logits"],
            dynamic_axes={
                "keypoints": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=14,
            dynamo=False
        )
        print(f"ONNX Model exported successfully to {onnx_path}")
    except Exception as e:
        print(f"ONNX export with dynamo=False failed: {e}. Trying legacy export...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["keypoints"],
            output_names=["logits"],
            opset_version=12
        )
        print(f"Legacy ONNX export successful to {onnx_path}")

if __name__ == "__main__":
    export()

