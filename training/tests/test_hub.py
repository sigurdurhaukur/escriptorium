import pytest
from PIL import Image

from training.hub import build_hf_dataset


@pytest.fixture
def sample_ground_truth(tmp_path):
    img = Image.new("L", (200, 30), color=255)
    img_path = tmp_path / "page.png"
    img.save(img_path)
    return [
        {"image": str(img_path), "baseline": [[10, 15], [190, 15]], "mask": [[0, 0], [199, 0], [199, 29], [0, 29]], "content": "hello"},
        {"image": str(img_path), "baseline": [[10, 15], [190, 15]], "mask": [[0, 0], [199, 0], [199, 29], [0, 29]], "content": "world"},
    ]


class TestBuildHfDataset:
    def test_returns_dataset_with_correct_rows(self, sample_ground_truth):
        ds = build_hf_dataset(sample_ground_truth)
        assert len(ds) == 2

    def test_contains_text_column(self, sample_ground_truth):
        ds = build_hf_dataset(sample_ground_truth)
        assert ds[0]["text"] == "hello"
        assert ds[1]["text"] == "world"

    def test_contains_baseline_and_mask(self, sample_ground_truth):
        ds = build_hf_dataset(sample_ground_truth)
        assert ds[0]["baseline"] == [[10, 15], [190, 15]]
        assert ds[0]["mask"] == [[0, 0], [199, 0], [199, 29], [0, 29]]

    def test_image_column_loads_pil(self, sample_ground_truth):
        ds = build_hf_dataset(sample_ground_truth)
        img = ds[0]["image"]
        assert isinstance(img, Image.Image)
        assert img.size == (200, 30)

    def test_features_schema(self, sample_ground_truth):
        ds = build_hf_dataset(sample_ground_truth)
        assert "image" in ds.features
        assert "text" in ds.features
        assert "baseline" in ds.features
        assert "mask" in ds.features

    def test_push_to_hub_called_with_correct_args(self, sample_ground_truth, monkeypatch):
        from datasets import Dataset
        pushed = []
        original_push = Dataset.push_to_hub
        def mock_push_to_hub(self, repo_id, split=None, token=None):
            pushed.append((repo_id, split, token))
        monkeypatch.setattr(Dataset, "push_to_hub", mock_push_to_hub)

        from training.hub import push_to_hub
        push_to_hub(sample_ground_truth, "test/repo")

        monkeypatch.setattr(Dataset, "push_to_hub", original_push)
        assert len(pushed) == 1
        assert pushed[0] == ("test/repo", "train", None)

    def test_empty_ground_truth(self):
        ds = build_hf_dataset([])
        assert len(ds) == 0
