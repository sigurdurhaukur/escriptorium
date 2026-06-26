import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from training.prepare import make_segmentation, build_arrow_datasets
from training.train import run_training


@pytest.fixture
def synthetic_image(tmp_path):
    img_path = tmp_path / "synthetic.png"
    img = Image.new("L", (200, 30), color=255)
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def single_line_ground_truth(synthetic_image):
    return [
        {
            "image": synthetic_image,
            "baseline": [[10, 15], [190, 15]],
            "mask": [[0, 0], [199, 0], [199, 29], [0, 29]],
            "content": "Hello World",
        }
    ]


class TestEndToEndSingleLine:
    def test_trains_one_epoch_and_produces_model(self, single_line_ground_truth, tmp_path):
        gt = single_line_ground_truth
        train_segs = make_segmentation(gt)
        val_segs = make_segmentation(gt)

        train_path = str(tmp_path / "train.arrow")
        val_path = str(tmp_path / "val.arrow")
        build_arrow_datasets(train_segs, val_segs, train_path, val_path)

        output_path = str(tmp_path / "model.safetensors")

        run_training(
            train_arrow=train_path,
            val_arrow=val_path,
            output_path=output_path,
            batch_size=1,
            max_epochs=1,
            device="cpu",
            precision="32",
        )

        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

    def test_fine_tunes_existing_model(self, single_line_ground_truth, tmp_path):
        gt = single_line_ground_truth
        train_segs = make_segmentation(gt)
        val_segs = make_segmentation(gt)

        train_path = str(tmp_path / "train.arrow")
        val_path = str(tmp_path / "val.arrow")
        build_arrow_datasets(train_segs, val_segs, train_path, val_path)

        output_path = str(tmp_path / "model.safetensors")

        run_training(
            train_arrow=train_path,
            val_arrow=val_path,
            output_path=output_path,
            batch_size=1,
            max_epochs=1,
            device="cpu",
            precision="32",
        )

        fine_tuned_path = str(tmp_path / "finetuned.safetensors")
        run_training(
            train_arrow=train_path,
            val_arrow=val_path,
            output_path=fine_tuned_path,
            model_path=output_path,
            batch_size=1,
            max_epochs=1,
            device="cpu",
            precision="32",
        )

        assert Path(fine_tuned_path).exists()
        assert Path(fine_tuned_path).stat().st_size > 0
