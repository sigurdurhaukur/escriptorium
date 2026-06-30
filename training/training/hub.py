import logging

from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

logger = logging.getLogger("training.hub")


def _crop_and_adjust(image_path: str, mask: list, baseline: list):
    img = PILImage.open(image_path)
    xs = [p[0] for p in mask]
    ys = [p[1] for p in mask]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)

    cropped = img.crop((min_x, min_y, max_x + 1, max_y + 1))

    adjusted_mask = [[x - min_x, y - min_y] for x, y in mask]
    adjusted_baseline = [[x - min_x, y - min_y] for x, y in baseline]

    return cropped, adjusted_mask, adjusted_baseline


def build_hf_dataset(ground_truth: list[dict]) -> Dataset:
    images = []
    baselines = []
    masks = []

    for e in ground_truth:
        cropped, adj_mask, adj_baseline = _crop_and_adjust(
            e["image"], e["mask"], e["baseline"]
        )
        images.append(cropped)
        masks.append(adj_mask)
        baselines.append(adj_baseline)

    data = {
        "image": images,
        "text": [e["content"] for e in ground_truth],
        "baseline": baselines,
        "mask": masks,
    }
    features = Features({
        "image": Image(),
        "text": Value("string"),
        "baseline": Sequence(Sequence(Value("int32"))),
        "mask": Sequence(Sequence(Value("int32"))),
    })
    ds = Dataset.from_dict(data, features=features)
    logger.info("Built HF dataset with %d rows", len(ds))
    return ds


def push_to_hub(
    ground_truth: list[dict],
    repo_id: str,
    split: str = "train",
):
    ds = build_hf_dataset(ground_truth)
    ds.push_to_hub(repo_id, split=split)
    logger.info("Pushed dataset to %s (split=%s)", repo_id, split)
