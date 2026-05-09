"""
export_model.py
───────────────
Export MobileNetV2 (pretrained) to TorchScript format and save as model.pt.

Usage:
    python3 export_model.py
"""

import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


MODEL_PATH: str = "model.pt"
DUMMY_INPUT_SHAPE: tuple[int, int, int, int] = (1, 3, 224, 224)


def load_model() -> nn.Module:
    """Load pretrained MobileNetV2 and set to eval mode."""
    print("Loading MobileNetV2 with default pretrained weights...")
    model: nn.Module = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.eval()
    print("Model loaded and set to eval mode.")
    return model


def trace_model(model: nn.Module, input_shape: tuple[int, int, int, int]) -> torch.jit.ScriptModule:
    """Trace the model with a dummy input tensor."""
    print(f"Tracing model with dummy input shape: {input_shape} ...")
    dummy_input: torch.Tensor = torch.zeros(input_shape)
    with torch.no_grad():
        traced: torch.jit.ScriptModule = torch.jit.trace(model, dummy_input)
    print("Model traced successfully.")
    return traced


def save_model(traced_model: torch.jit.ScriptModule, path: str) -> None:
    """Save TorchScript model to disk."""
    torch.jit.save(traced_model, path)
    size_bytes: int = os.path.getsize(path)
    size_mb: float = size_bytes / (1024 * 1024)
    print(f"Model saved to '{path}' ({size_mb:.2f} MB / {size_bytes} bytes)")


def main() -> None:
    model: nn.Module = load_model()
    traced_model: torch.jit.ScriptModule = trace_model(model, DUMMY_INPUT_SHAPE)
    save_model(traced_model, MODEL_PATH)
    print("Done. Run 'python3 inference.py <image_path>' to test inference.")


if __name__ == "__main__":
    main()
