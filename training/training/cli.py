import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any kraken/torch import for MPS CTC loss fallback
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from training.client import EscriptoriumClient, authenticate
from training.hub import push_to_hub
from training.prepare import make_segmentation, build_arrow_datasets
from training.train import run_training

logger = logging.getLogger("training.cli")


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Kraken recognition model from eScriptorium data (local, MPS-enabled)"
    )
    parser.add_argument("--url", required=True, help="eScriptorium base URL")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--document-id", type=int, required=True)
    parser.add_argument("--transcription-id", type=int, help="Transcription pk (default: first non-archived)")
    parser.add_argument("--transcription-name", help="Transcription name (default: first non-archived)")
    parser.add_argument("--model-path", help="Path to existing .safetensors model for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=5,
                        help="Stop if val CER doesn't improve for N epochs (0 = disable)")
    parser.add_argument("--precision", default="32", choices=["16-mixed", "32"])
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--output", default="icelandic_model.safetensors", help="Output model path")
    parser.add_argument("--hf-repo", help="HF Hub repo to push dataset to (e.g. 'username/icelandic-ocr')")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--temp-dir", help="Keep temp files at this path (for debugging)")
    args = parser.parse_args()

    setup_logging(args.verbose)

    logger.info("Authenticating to %s", args.url)
    token = authenticate(args.url, args.username, args.password)
    client = EscriptoriumClient(args.url, token)

    doc = client.get_document(args.document_id)
    logger.info("Document: %s (pk=%s)", doc.get("name") or "(unnamed)", doc.get("pk"))

    if args.transcription_id and args.transcription_name:
        logger.error("Specify only one of --transcription-id or --transcription-name")
        sys.exit(1)

    transcriptions = client.get_transcriptions(args.document_id)
    if args.transcription_id:
        transcription = next(
            (t for t in transcriptions if t["pk"] == args.transcription_id), None
        )
        if not transcription:
            logger.error("Transcription pk=%s not found in document", args.transcription_id)
            sys.exit(1)
    elif args.transcription_name:
        transcription = next(
            (t for t in transcriptions if t["name"] == args.transcription_name), None
        )
        if not transcription:
            logger.error("Transcription '%s' not found. Available: %s",
                         args.transcription_name, [t["name"] for t in transcriptions])
            sys.exit(1)
    else:
        transcription = next(
            (t for t in transcriptions if not t.get("archived")), None
        )
        if not transcription:
            logger.error("No active transcription found")
            sys.exit(1)
    logger.info("Using transcription: %s (pk=%s)", transcription["name"], transcription["pk"])

    parts = client.get_parts(args.document_id)
    logger.info("Found %d parts (pages)", len(parts))

    if args.temp_dir:
        work_dir = Path(args.temp_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="escriptorium_train_"))

    images_dir = work_dir / "images"
    images_dir.mkdir(exist_ok=True)

    ground_truth = []

    for idx, part in enumerate(parts):
        part_id = part["pk"]
        logger.info("[%d/%d] Part %s (order=%s)", idx + 1, len(parts), part_id, part.get("order"))

        img_url = part.get("image")
        if isinstance(img_url, dict):
            img_url = img_url.get("uri", "")
        if not img_url:
            logger.warning("  No image URL, skipping part %s", part_id)
            continue

        ext = os.path.splitext(img_url.split("?")[0])[1] or ".jpg"
        img_path = images_dir / f"part_{part_id}{ext}"
        client.download_image(img_url, img_path)

        lines = client.get_lines(args.document_id, part_id)
        lts = client.get_line_transcriptions(args.document_id, part_id, transcription["pk"])
        content_by_line = {lt["line"]: lt["content"] for lt in lts if lt.get("content")}

        for line in lines:
            content = content_by_line.get(line["pk"])
            if not content:
                continue
            if not line.get("baseline") or not line.get("mask"):
                continue
            ground_truth.append({
                "image": str(img_path),
                "baseline": line["baseline"],
                "mask": line["mask"],
                "content": content,
            })

    logger.info("Total ground truth lines: %d", len(ground_truth))

    if args.hf_repo:
        logger.info("Pushing dataset to HF Hub: %s", args.hf_repo)
        push_to_hub(ground_truth, args.hf_repo)

    if len(ground_truth) < 10:
        logger.error("Too few lines (%d). Need at least 10 to train.", len(ground_truth))
        sys.exit(1)

    import numpy as np
    np.random.default_rng(241960353267317949653744176059648850006).shuffle(ground_truth)
    partition = max(int(len(ground_truth) / 10), 1)

    logger.info("Building training dataset (%d lines)...", len(ground_truth) - partition)
    train_segs = make_segmentation(ground_truth[partition:])
    val_segs = make_segmentation(ground_truth[:partition])

    train_arrow = str(work_dir / "train.arrow")
    val_arrow = str(work_dir / "val.arrow")
    build_arrow_datasets(train_segs, val_segs, train_arrow, val_arrow)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU")
        device = "cpu"

    logger.info(
        "Starting training: device=%s precision=%s batch=%d epochs=%d",
        device, args.precision, args.batch_size, args.epochs,
    )

    run_training(
        train_arrow=train_arrow,
        val_arrow=val_arrow,
        output_path=args.output,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        device=device,
        precision=args.precision,
        model_path=args.model_path,
    )

    logger.info("Done! Model saved to %s", args.output)

    if not args.temp_dir:
        import shutil
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
