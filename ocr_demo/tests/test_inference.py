"""
Tests for inference.py — run with:
    cd ocr_demo && uv run pytest tests/ -v
"""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = str(Path(__file__).parent.parent.parent / "training" / "catmus-print-fondue-ft.safetensors")


def make_image(width=400, height=200) -> bytes:
    """Synthetic white page with a black text-like horizontal band."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 50, 380, 70], fill=(0, 0, 0))
    draw.rectangle([20, 110, 300, 130], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Test 1: model loads without error ───────────────────────────────────────

def test_model_loads():
    import inference
    inference.load_models()
    assert inference._rec_model is not None


# ── Test 2: run_ocr returns expected structure ───────────────────────────────

def test_run_ocr_returns_structure():
    import inference
    inference.load_models()

    result = inference.run_ocr(make_image())

    assert "width" in result
    assert "height" in result
    assert "lines" in result
    assert "image_b64" in result
    assert "mime" in result
    assert "line_count" in result
    assert isinstance(result["lines"], list)
    assert result["line_count"] == len(result["lines"])


# ── Test 3: each line has required fields ────────────────────────────────────

def test_line_structure():
    import inference
    inference.load_models()

    result = inference.run_ocr(make_image())

    for line in result["lines"]:
        assert "text" in line
        assert "baseline" in line
        assert "boundary" in line
        assert "font_size" in line
        assert isinstance(line["font_size"], int)
        assert line["font_size"] >= 12
