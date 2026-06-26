import logging

from datasets import Dataset, Features, Image, Sequence, Value

logger = logging.getLogger("training.hub")


def build_hf_dataset(ground_truth: list[dict]) -> Dataset:
    data = {
        "image": [e["image"] for e in ground_truth],
        "text": [e["content"] for e in ground_truth],
        "baseline": [e["baseline"] for e in ground_truth],
        "mask": [e["mask"] for e in ground_truth],
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
