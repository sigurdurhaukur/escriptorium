import os
import io
import base64
from pathlib import Path
from PIL import Image

RECOGNITION_MODEL_PATH = os.environ.get(
    "RECOGNITION_MODEL",
    str(Path(__file__).parent.parent / "training" / "catmus-print-fondue-ft.safetensors"),
)

_rec_model = None


def load_models():
    global _rec_model
    from kraken.models import load_safetensors
    from kraken.configs import RecognitionInferenceConfig

    models_list = load_safetensors(RECOGNITION_MODEL_PATH)
    rec_models = [m for m in models_list if "recognition" in m.model_type]
    if not rec_models:
        raise RuntimeError(f"No recognition model found in {RECOGNITION_MODEL_PATH}")

    model = rec_models[0]
    model.prepare_for_inference(RecognitionInferenceConfig(num_line_workers=0))
    _rec_model = model


def run_ocr(image_bytes: bytes) -> dict:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from kraken import blla

    pil_im = Image.open(io.BytesIO(image_bytes))
    img_format = pil_im.format or "PNG"
    mime = f"image/{img_format.lower()}"

    im = pil_im.convert("RGB")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        seg = blla.segment(im)

    lines = []
    for record in _rec_model.predict(im, seg):
        boundary = record.boundary or []
        baseline = record.baseline or []
        if not baseline:
            continue
        ys = [p[1] for p in boundary] if boundary else [baseline[0][1]]
        height = (max(ys) - min(ys)) if len(ys) > 1 else 20
        lines.append({
            "text": record.prediction,
            "boundary": boundary,
            "baseline": baseline,
            "font_size": max(12, int(height * 0.75)),
        })

    return {
        "width": im.width,
        "height": im.height,
        "lines": lines,
        "image_b64": base64.b64encode(image_bytes).decode(),
        "mime": mime,
        "line_count": len(lines),
    }
