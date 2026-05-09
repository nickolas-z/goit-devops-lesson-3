"""
inference.py
────────────
Run top-3 ImageNet inference on a given image or all images in a folder.

Usage:
    python3 inference.py path/to/image.jpg
    python3 inference.py --image path/to/image.jpg
    python3 inference.py path/to/folder/
"""

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms


SCRIPT_DIR: Path = Path(__file__).resolve().parent
MODEL_PATH: str = str(SCRIPT_DIR / "model.pt")
IMAGENET_CLASSES_URL: str = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")

# Minimal fallback dict (~20 classes) used when the URL is unreachable
FALLBACK_CLASSES: dict[int, str] = {
    0:   "tench",
    1:   "goldfish",
    2:   "great white shark",
    3:   "tiger shark",
    4:   "hammerhead shark",
    11:  "bullfrog",
    13:  "rock python",
    15:  "king cobra",
    18:  "box turtle",
    26:  "tree frog",
    80:  "spider monkey",
    281: "tabby cat",
    282: "tiger cat",
    283: "Persian cat",
    284: "Siamese cat",
    285: "Egyptian cat",
    291: "lion",
    292: "tiger",
    340: "zebra",
    385: "Indian elephant",
    574: "golf ball",
    717: "pickup truck",
    736: "beach wagon",
    850: "teddy bear",
    895: "warship",
    980: "volcano",
    985: "daisy",
}

###########################################################################################
# ImageNet class labels

def load_imagenet_classes() -> dict[int, str]:
    """
    Load 1000-class ImageNet labels. Priority:
    1. Local imagenet_classes.txt (same dir as script)
    2. Download from PyTorch Hub URL
    3. Embedded minimal fallback dict
    """
    local_path = SCRIPT_DIR / "imagenet_classes.txt"
    if local_path.exists():
        lines: list[str] = local_path.read_text(encoding="utf-8").splitlines()
        classes: dict[int, str] = {idx: name.strip() for idx, name in enumerate(lines)}
        print(f"[info] Loaded {len(classes)} ImageNet class labels from local file.")
        return classes

    try:
        with urllib.request.urlopen(IMAGENET_CLASSES_URL, timeout=5) as resp:
            lines = resp.read().decode("utf-8").splitlines()
        classes = {idx: name.strip() for idx, name in enumerate(lines)}
        print(f"[info] Loaded {len(classes)} ImageNet class labels from URL.")
        return classes
    except Exception as exc:
        print(f"[warn] Could not load class labels ({exc}). Using fallback dict.")
        return FALLBACK_CLASSES

###########################################################################################
# Image preprocessing

def build_transform() -> transforms.Compose:
    """Return the standard ImageNet preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image(image_path: str) -> torch.Tensor:
    """
    Open an image file and apply the preprocessing pipeline.

    Raises:
        FileNotFoundError: if the file does not exist.
        RuntimeError: if the image cannot be opened or converted.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    try:
        img: Image.Image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to open image '{image_path}': {exc}") from exc

    transform = build_transform()
    tensor: torch.Tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 224, 224)
    return tensor


###########################################################################################
# Model loading

def load_model(model_path: str) -> torch.jit.ScriptModule:
    """
    Load TorchScript model from disk.

    Raises:
        FileNotFoundError: if model file does not exist.
        RuntimeError: if the file cannot be loaded as TorchScript.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'. "
            "Run 'python3 export_model.py' first."
        )

    try:
        model: torch.jit.ScriptModule = torch.jit.load(str(path), map_location="cpu")
        model.eval()
    except Exception as exc:
        raise RuntimeError(f"Failed to load TorchScript model from '{model_path}': {exc}") from exc

    return model


###########################################################################################
# Inference

def run_inference(
    model: torch.jit.ScriptModule,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Run forward pass and return raw logits."""
    with torch.no_grad():
        logits: torch.Tensor = model(input_tensor)
    return logits


def get_top_k(
    logits: torch.Tensor,
    classes: dict[int, str],
    k: int = 3,
) -> list[tuple[int, str, float]]:
    """
    Return top-k predictions as (class_idx, class_name, probability_percent).
    """
    probabilities: torch.Tensor = torch.nn.functional.softmax(logits[0], dim=0)
    top_probs, top_indices = torch.topk(probabilities, k=k)

    results: list[tuple[int, str, float]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label: str = classes.get(idx, f"class_{idx}")
        results.append((idx, label, float(prob) * 100.0))
    return results


def print_top_predictions(predictions: list[tuple[int, str, float]]) -> None:
    """Pretty-print top-k predictions."""
    print("\nTop-3 predictions:")
    for rank, (idx, name, pct) in enumerate(predictions, start=1):
        print(f"  #{rank} {name} ({idx}): {pct:.2f}%")
    print()

###########################################################################################
# Folder / single-file resolution

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
)


def collect_images(input_path: str) -> list[Path]:
    """
    Return a list of image paths to process.
    - If input_path is a file: return [input_path].
    - If input_path is a directory: return all image files inside (non-recursive).
    """
    p = Path(input_path)
    if p.is_dir():
        images = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No image files found in directory: '{input_path}'")
        return images
    return [p]


###########################################################################################
# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ImageNet top-3 inference with a TorchScript MobileNetV2 model. "
            "Accepts a single image file or a directory of images."
        )
    )
    # Accept both positional and --image flag
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to an image file or a directory containing images (positional).",
    )
    parser.add_argument(
        "--image",
        dest="image_flag",
        default=None,
        metavar="PATH",
        help="Path to an image file or a directory containing images (--image flag).",
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        metavar="PATH",
        help=f"Path to the TorchScript model file (default: {MODEL_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = parse_args()

    # Resolve input path (positional takes precedence over --image)
    input_path: Optional[str] = args.image or args.image_flag
    if input_path is None:
        print("[error] Please provide an image path or directory.", file=sys.stderr)
        print("  Usage: python3 inference.py path/to/image.jpg", file=sys.stderr)
        print("         python3 inference.py path/to/folder/", file=sys.stderr)
        sys.exit(1)

    # Collect image paths (single file or all images in a directory)
    try:
        image_paths: list[Path] = collect_images(input_path)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    # Load class labels
    classes: dict[int, str] = load_imagenet_classes()

    # Load model
    try:
        model: torch.jit.ScriptModule = load_model(args.model)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    # Run inference for each image. Return a failure code if no input was usable.
    processed_images = 0
    for img_path in image_paths:
        if len(image_paths) > 1:
            print(f"\n=== {img_path.name} ===")
        try:
            input_tensor: torch.Tensor = load_image(str(img_path))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue

        logits: torch.Tensor = run_inference(model, input_tensor)
        predictions: list[tuple[int, str, float]] = get_top_k(logits, classes, k=3)
        print_top_predictions(predictions)
        processed_images += 1

    if processed_images == 0:
        print("[error] No images were processed successfully.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
