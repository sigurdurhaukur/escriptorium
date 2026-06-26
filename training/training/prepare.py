from collections import defaultdict

from kraken.containers import BaselineLine, Segmentation
from kraken.lib.arrow_dataset import build_binary_dataset


def make_segmentation(lines_data):
    by_img = defaultdict(list)
    for i, d in enumerate(lines_data):
        by_img[d["image"]].append(
            BaselineLine(
                id=str(i),
                baseline=d["baseline"],
                boundary=d["mask"],
                text=d["content"],
                language=None,
            )
        )
    return [
        Segmentation(
            text_direction="horizontal-lr",
            imagename=img,
            type="baselines",
            lines=lines,
            script_detection=False,
            language=None,
        )
        for img, lines in by_img.items()
    ]


def build_arrow_datasets(train_segs, val_segs, train_path, val_path, num_workers=0):
    build_binary_dataset(train_segs, output_file=train_path, num_workers=num_workers, format_type=None)
    build_binary_dataset(val_segs, output_file=val_path, num_workers=num_workers, format_type=None)
