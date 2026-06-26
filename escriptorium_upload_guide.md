# Uploading Hugging Face Datasets to eScriptorium

## Overview

eScriptorium supports several input formats:
- **Images**: PNG, JPG (local files, PDF extraction, IIIF)
- **XML**: ALTO XML or PAGE XML (for segmentation/transcription import)

For Hugging Face datasets, you'll typically work with images via the API.

## Prerequisites

1. **eScriptorium instance** running and accessible via URL
2. **eScriptorium account** with API access
3. **Python 3.7+**
4. A **Hugging Face dataset** with image data

## Installation

```bash
pip install escriptorium-connector python-dotenv datasets pillow
```

## Setup

### 1. Configure Credentials

Create a `.env` file in your working directory:

```
ESCRIPTORIUM_URL=https://your-escriptorium.example.com
ESCRIPTORIUM_USERNAME=your_username
ESCRIPTORIUM_PASSWORD=your_password
```

Keep this file **secret** and add it to `.gitignore`.

### 2. Verify Credentials

Test the connection:

```python
from escriptorium_connector import EscriptoriumConnector
import os
from dotenv import load_dotenv

load_dotenv()
connector = EscriptoriumConnector(
    os.getenv('ESCRIPTORIUM_URL'),
    os.getenv('ESCRIPTORIUM_USERNAME'),
    os.getenv('ESCRIPTORIUM_PASSWORD')
)
docs = connector.get_documents()
print(f"Connected! Found {len(docs)} documents")
```

## Usage

### Basic Upload

```bash
python hf_to_escriptorium.py mnist --split train --project-id 1
```

### Advanced Options

```bash
python hf_to_escriptorium.py your_dataset/name \
    --split validation \
    --project-id 2 \
    --doc-name "My Dataset Batch" \
    --image-column "img" \
    --max-samples 500
```

**Arguments:**
- `dataset`: HF dataset identifier (required)
- `--split`: Which split to load (default: `train`)
- `--project-id`: eScriptorium project ID (default: `1`)
- `--doc-name`: Custom document name
- `--image-column`: Dataset column containing images (default: `image`)
- `--max-samples`: Upload only N samples (useful for testing)

### Custom Script Usage

If you need more control, use the script functions directly:

```python
from hf_to_escriptorium import (
    setup_credentials,
    load_hf_dataset,
    extract_images,
    create_document,
    upload_images_to_escriptorium
)
from escriptorium_connector import EscriptoriumConnector

# Setup
url, username, password = setup_credentials()
connector = EscriptoriumConnector(url, username, password)

# Load dataset
dataset = load_hf_dataset("mnist", split="train")

# Extract images (handles PIL, numpy, etc.)
temp_dir, image_paths = extract_images(dataset, max_samples=100)

# Create document
doc_id = create_document(connector, project_id=1, doc_name="MNIST Test")

# Upload
upload_images_to_escriptorium(connector, doc_id, image_paths)
```

## How It Works

1. **Load Dataset**: Downloads from Hugging Face Hub
2. **Convert Images**: Handles PIL Image objects, numpy arrays, etc.
3. **Save Locally**: Extracts to temporary PNG files
4. **Create Document**: Creates a new eScriptorium document via API
5. **Upload**: Sends images to eScriptorium (batched)
6. **Cleanup**: Removes temporary files

## API Methods Used

The script relies on these `escriptorium-connector` methods:

- `EscriptoriumConnector(url, username, password)` - Initialize connector
- `connector.create_document(name, project)` - Create new document
- `connector.upload_image(document_id, image_file, filename)` - Upload image

**Note**: Method names and signatures depend on the connector version. Check the official docs if methods change.

## Handling Different Dataset Formats

### Dataset with images in PIL format (typical)
```python
dataset = load_dataset("mnist")
# dataset["image"] returns PIL.Image objects
main("mnist", image_column="image")
```

### Dataset with images as numpy arrays
```python
# Script auto-converts numpy arrays → PIL
main("my_dataset", image_column="array_column")
```

### Dataset with nested image structures
If your dataset has a custom structure, modify `extract_images()` to handle it:

```python
def extract_images(dataset, image_column, max_samples=None):
    # ... existing code ...
    img = sample[image_column]
    if isinstance(img, dict):  # Custom structure
        img = Image.fromarray(img["data"])
    # ... rest of function
```

## Troubleshooting

### Authentication fails
- Verify credentials in `.env`
- Check eScriptorium URL is correct
- Ensure your account has API permissions

### Image extraction fails
- Check dataset has the specified `--image-column`
- Use `dataset.features` to inspect available columns
- Verify images are in supported format (PIL, numpy, etc.)

### Upload timeout
- Use `--max-samples` to upload smaller batches
- Check eScriptorium server is responsive

### Memory issues with large datasets
- Process dataset in splits
- Use `--max-samples` to limit per run
- Consider batching uploads

## Example Workflows

### Upload MNIST dataset
```bash
python hf_to_escriptorium.py mnist --split train --max-samples 1000
```

### Upload custom handwriting dataset (incrementally)
```bash
for split in train validation; do
    python hf_to_escriptorium.py \
        myorg/htr_dataset \
        --split $split \
        --doc-name "HTR Dataset - $split"
done
```

### Test with small sample first
```bash
python hf_to_escriptorium.py my_large_dataset --max-samples 50
# If successful, upload full dataset:
python hf_to_escriptorium.py my_large_dataset
```

## What Next?

After uploading images:

1. **In eScriptorium UI**:
   - Go to Documents
   - Select your uploaded document
   - Run automatic segmentation/transcription with Kraken models

2. **Via API** (for automation):
   ```python
   # Segment images
   connector.segment_document(document_id, model_id)

   # Get transcription results
   transcriptions = connector.get_document_transcriptions(document_id)
   ```

3. **Export data**:
   ```python
   # Download as XML/ALTO
   connector.export_document(document_id, format="alto")
   ```

## References

- eScriptorium Docs: https://escriptorium.readthedocs.io/
- escriptorium-connector: https://pypi.org/project/escriptorium-connector/
- Hugging Face Datasets: https://huggingface.co/docs/datasets/

## Notes for Your Icelandic OCR Project

For your Hábrók HTR work, this script will be useful for:
- Uploading your 19th-century letter dataset to eScriptorium
- Batch processing handwritten Icelandic documents
- Creating training data via eScriptorium's annotation interface
- Building active learning pipelines with eScriptorium + Kraken

Consider combining with your existing Kraken/eScriptorium setup on Hábrók.