from pathlib import Path

import numpy as np
import pytest
from kraken.containers import BaselineLine, Segmentation

from training.prepare import make_segmentation, build_arrow_datasets


class TestMakeSegmentation:
    def test_groups_lines_by_image(self):
        lines = [
            {"image": "/img/a.jpg", "baseline": [[0, 0], [10, 0]], "mask": [[0, -5], [10, -5], [10, 5], [0, 5]], "content": "hello"},
            {"image": "/img/a.jpg", "baseline": [[0, 20], [10, 20]], "mask": [[0, 15], [10, 15], [10, 25], [0, 25]], "content": "world"},
            {"image": "/img/b.jpg", "baseline": [[0, 0], [10, 0]], "mask": [[0, -5], [10, -5], [10, 5], [0, 5]], "content": "foo"},
        ]

        segs = make_segmentation(lines)

        assert len(segs) == 2
        img_a = [s for s in segs if s.imagename == "/img/a.jpg"][0]
        assert len(img_a.lines) == 2
        assert img_a.lines[0].text == "hello"
        assert img_a.lines[1].text == "world"

        img_b = [s for s in segs if s.imagename == "/img/b.jpg"][0]
        assert len(img_b.lines) == 1
        assert img_b.lines[0].text == "foo"

    def test_assigns_sequential_ids(self):
        lines = [
            {"image": "/img/a.jpg", "baseline": [[0, 0], [10, 0]], "mask": [[0, -5], [10, -5], [10, 5], [0, 5]], "content": "a"},
            {"image": "/img/a.jpg", "baseline": [[0, 20], [10, 20]], "mask": [[0, 15], [10, 15], [10, 25], [0, 25]], "content": "b"},
        ]

        segs = make_segmentation(lines)

        assert segs[0].lines[0].id == "0"
        assert segs[0].lines[1].id == "1"

    def test_sets_default_properties(self):
        lines = [
            {"image": "/img/a.jpg", "baseline": [[0, 0], [10, 0]], "mask": [[0, -5], [10, -5], [10, 5], [0, 5]], "content": "x"},
        ]

        segs = make_segmentation(lines)

        assert len(segs) == 1
        s = segs[0]
        assert s.text_direction == "horizontal-lr"
        assert s.type == "baselines"
        assert s.script_detection is False
        assert s.language is None

    def test_stores_baseline_and_boundary(self):
        baseline = [[5, 10], [20, 10], [30, 12]]
        mask = [[0, 0], [35, 0], [35, 20], [0, 20]]
        lines = [
            {"image": "/img/a.jpg", "baseline": baseline, "mask": mask, "content": "test"},
        ]

        segs = make_segmentation(lines)

        bl = segs[0].lines[0]
        assert bl.baseline == baseline
        assert bl.boundary == mask


class TestBuildArrowDatasets:
    @pytest.fixture
    def sample_segs(self):
        return [
            Segmentation(
                text_direction="horizontal-lr",
                imagename="/img/a.jpg",
                type="baselines",
                lines=[
                    BaselineLine(id="0", baseline=[[0, 0], [10, 0]], boundary=[[0, -5], [10, -5], [10, 5], [0, 5]], text="hello", language=None),
                ],
                script_detection=False,
                language=None,
            )
        ]

    def test_creates_arrow_files(self, sample_segs, tmp_path):
        train_path = str(tmp_path / "train.arrow")
        val_path = str(tmp_path / "val.arrow")

        build_arrow_datasets(sample_segs, sample_segs, train_path, val_path)

        assert Path(train_path).exists()
        assert Path(val_path).exists()
        assert Path(train_path).stat().st_size > 0
