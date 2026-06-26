#!/usr/bin/env python3
"""
Quick smoke test: trains on 1 synthetic image for 1 epoch.
Use this to verify the training pipeline works before pulling real data.

Usage:
    cd training && uv run python scripts/smoke_test.py

    # With MPS (Apple Silicon GPU):
    cd training && uv run python scripts/smoke_test.py --device mps

    # 16-bit mixed precision (uses ~half the memory):
    cd training && uv run python scripts/smoke_test.py --device mps --precision 16-mixed
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must be set before any kraken/torch import for MPS CTC loss fallback
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Smoke test Kraken training pipeline")
    parser.add_argument("--device", default="cpu", choices=["mps", "cpu"])
    parser.add_argument("--precision", default="32", choices=["16-mixed", "32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", default=None, help="Output model path")
    parser.add_argument("--hf-repo", help="HF Hub repo to push dataset to (e.g. 'username/test-ocr')")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="smoke_test_"))
    print(f"Work dir: {tmp}")

    # Create a synthetic image with text-like content
    img = Image.new("L", (400, 60), color=255)
    for x in range(50, 350, 20):
        for y in range(15, 45):
            img.putpixel((x, y), 0)
    img_path = tmp / "synthetic.png"
    img.save(img_path)
    print(f"Synthetic image: {img_path} ({img.size})")

    ground_truth = [
        {"image": str(img_path), "baseline": [[20, 30], [380, 30]], "mask": [[0, 0], [399, 0], [399, 59], [0, 59]], "content": "Hello World"},
        {"image": str(img_path), "baseline": [[20, 30], [380, 30]], "mask": [[0, 0], [399, 0], [399, 59], [0, 59]], "content": "Kraken OCR test"},
    ]

    from training.hub import push_to_hub
    from training.prepare import make_segmentation, build_arrow_datasets
    from training.train import run_training

    if args.hf_repo:
        print(f"Pushing dataset to HF Hub: {args.hf_repo}")
        push_to_hub(ground_truth, args.hf_repo)

    print("Building Arrow datasets...")
    train_segs = make_segmentation(ground_truth)
    val_segs = make_segmentation(ground_truth)

    train_arrow = str(tmp / "train.arrow")
    val_arrow = str(tmp / "val.arrow")
    build_arrow_datasets(train_segs, val_segs, train_arrow, val_arrow)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    output = args.output or str(tmp / "smoke_model.safetensors")
    print(f"Training 1 epoch on {device}...")

    run_training(
        train_arrow=train_arrow,
        val_arrow=val_arrow,
        output_path=output,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        device=device,
        precision=args.precision,
    )

    print(f"Model saved to {output}")
    print(f"File size: {Path(output).stat().st_size} bytes")
    print("Smoke test PASSED!")


if __name__ == "__main__":
    main()
