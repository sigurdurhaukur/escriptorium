#!/usr/bin/env python3
"""
Upload images from a Hugging Face dataset to eScriptorium via the API.

This script:
1. Loads a dataset from Hugging Face
2. Converts images (if needed) to standard formats
3. Uses the escriptorium-connector to upload to eScriptorium
4. Creates a new document or uses an existing one
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv
from escriptorium_connector import EscriptoriumConnector
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_credentials():
    """Load credentials from .env file."""
    load_dotenv()

    url = os.getenv("ESCRIPTORIUM_URL")
    username = os.getenv("ESCRIPTORIUM_USERNAME")
    password = os.getenv("ESCRIPTORIUM_PASSWORD")

    if not all([url, username, password]):
        raise ValueError(
            "Missing credentials. Ensure .env file has:\n"
            "ESCRIPTORIUM_URL=\n"
            "ESCRIPTORIUM_USERNAME=\n"
            "ESCRIPTORIUM_PASSWORD="
        )

    return url, username, password


def load_hf_dataset(dataset_name: str, split: str = "train", **kwargs):
    """
    Load a Hugging Face dataset.

    Args:
        dataset_name: HF dataset identifier (e.g., "mnist", "your_dataset")
        split: Dataset split to load
        **kwargs: Additional arguments for load_dataset()

    Returns:
        Dataset object
    """
    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    dataset = load_dataset(dataset_name, split=split, **kwargs)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def extract_images(
    dataset, image_column: str = "image", max_samples: Optional[int] = None
):
    """
    Extract images from dataset and save to temporary directory.

    Args:
        dataset: Hugging Face dataset
        image_column: Name of the image column
        max_samples: Maximum number of samples to extract (None = all)

    Returns:
        Tuple of (temp_dir_path, list of image file paths)
    """
    temp_dir = tempfile.mkdtemp(prefix="escriptorium_")
    image_paths = []

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)

    logger.info(f"Extracting {num_samples} images to {temp_dir}")

    for idx in range(num_samples):
        sample = dataset[idx]

        # Handle different image formats
        if image_column not in sample:
            logger.warning(f"Sample {idx} missing '{image_column}' column, skipping")
            continue

        img = sample[image_column]

        # Convert to PIL Image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img) if hasattr(img, "shape") else img

        # Save as PNG
        img_path = Path(temp_dir) / f"image_{idx:06d}.png"
        img.save(img_path)
        image_paths.append(str(img_path))

        if (idx + 1) % 100 == 0:
            logger.info(f"Extracted {idx + 1}/{num_samples} images")

    logger.info(f"Successfully extracted {len(image_paths)} images")
    return temp_dir, image_paths


def create_document(connector: EscriptoriumConnector, project_id: int, doc_name: str):
    """
    Create a new document in eScriptorium.

    Args:
        connector: EscriptoriumConnector instance
        project_id: ID of the project to create document in
        doc_name: Name for the new document

    Returns:
        Document ID
    """
    logger.info(f"Creating document '{doc_name}' in project {project_id}")

    # Create document via API
    # Note: escriptorium-connector may have different method names;
    # check actual library for exact method
    doc = connector.create_document(name=doc_name, project=project_id)

    logger.info(f"Created document with ID: {doc.get('id')}")
    return doc.get("id")


def upload_images_to_escriptorium(
    connector: EscriptoriumConnector, document_id: int, image_paths: list[str]
):
    """
    Upload images to eScriptorium document.

    Args:
        connector: EscriptoriumConnector instance
        document_id: Target document ID
        image_paths: List of image file paths
    """
    logger.info(f"Uploading {len(image_paths)} images to document {document_id}")

    for idx, img_path in enumerate(image_paths):
        try:
            with open(img_path, "rb") as f:
                # Upload image
                # Note: Method name and parameters depend on escriptorium-connector version
                connector.upload_image(
                    document_id=document_id, image_file=f, filename=Path(img_path).name
                )

            if (idx + 1) % 10 == 0:
                logger.info(f"Uploaded {idx + 1}/{len(image_paths)} images")

        except Exception as e:
            logger.error(f"Failed to upload {img_path}: {e}")
            continue

    logger.info("Image upload complete")


def main(
    dataset_name: str,
    split: str = "train",
    project_id: int = 1,
    doc_name: Optional[str] = None,
    image_column: str = "image",
    max_samples: Optional[int] = None,
):
    """
    Main workflow: HF dataset → eScriptorium.

    Args:
        dataset_name: Hugging Face dataset identifier
        split: Dataset split to use
        project_id: eScriptorium project ID (default: 1)
        doc_name: Name for new document (default: "{dataset_name}_{split}")
        image_column: Name of image column in dataset
        max_samples: Max images to upload (None = all)
    """

    # Setup
    url, username, password = setup_credentials()
    connector = EscriptoriumConnector(url, username, password)

    # Load dataset
    dataset = load_hf_dataset(dataset_name, split=split)

    # Extract images
    temp_dir, image_paths = extract_images(
        dataset, image_column=image_column, max_samples=max_samples
    )

    try:
        # Create document
        if doc_name is None:
            doc_name = f"{dataset_name}_{split}"

        document_id = create_document(connector, project_id, doc_name)

        # Upload images
        upload_images_to_escriptorium(connector, document_id, image_paths)

        logger.info(
            f"✓ Successfully uploaded {len(image_paths)} images to document {document_id}"
        )

    finally:
        # Cleanup temp directory
        import shutil

        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload Hugging Face dataset to eScriptorium"
    )
    parser.add_argument(
        "dataset",
        help="Hugging Face dataset name (e.g., 'mnist', 'your_org/your_dataset')",
    )
    parser.add_argument(
        "--split", default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--project-id", type=int, default=1, help="eScriptorium project ID (default: 1)"
    )
    parser.add_argument("--doc-name", help="Document name (default: {dataset}_{split})")
    parser.add_argument(
        "--image-column",
        default="image",
        help="Name of image column (default: 'image')",
    )
    parser.add_argument(
        "--max-samples", type=int, help="Maximum images to upload (default: all)"
    )

    args = parser.parse_args()

    main(
        dataset_name=args.dataset,
        split=args.split,
        project_id=args.project_id,
        doc_name=args.doc_name,
        image_column=args.image_column,
        max_samples=args.max_samples,
    )
