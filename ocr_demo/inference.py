import io
import os
import threading
import time
from pathlib import Path
from typing import Generator

from PIL import Image

RECOGNITION_MODEL_PATH = os.environ.get(
    "RECOGNITION_MODEL",
    str(Path(__file__).parent.parent / "training" / "catmus-print-fondue-ft.safetensors"),
)

_rec_model = None


def load_models():
    global _rec_model
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=r"You will not be able to run predict\(\) on this Core ML model",
        category=RuntimeWarning,
    )
    from kraken.models import load_safetensors
    from kraken.configs import RecognitionInferenceConfig

    models_list = load_safetensors(RECOGNITION_MODEL_PATH)
    rec_models = [m for m in models_list if "recognition" in m.model_type]
    if not rec_models:
        raise RuntimeError(f"No recognition model found in {RECOGNITION_MODEL_PATH}")

    model = rec_models[0]
    model.prepare_for_inference(RecognitionInferenceConfig(num_line_workers=0))
    _rec_model = model


def scale_image(image_bytes: bytes, max_dim: int = 3000) -> bytes:
    pil_im = Image.open(io.BytesIO(image_bytes))
    fmt = pil_im.format or "PNG"
    w, h = pil_im.size
    if max(w, h) <= max_dim:
        return image_bytes
    scale = max_dim / max(w, h)
    pil_im = pil_im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if fmt == "JPEG" and pil_im.mode in ("RGBA", "LA", "P"):
        pil_im = pil_im.convert("RGB")
    buf = io.BytesIO()
    pil_im.save(buf, format=fmt)
    return buf.getvalue()


def _line_geometry(record) -> dict:
    boundary = record.boundary or []
    baseline = record.baseline or []
    ys = [p[1] for p in boundary] if boundary else [baseline[0][1]]
    height = (max(ys) - min(ys)) if len(ys) > 1 else 20
    if boundary:
        xs = [p[0] for p in boundary]
        line_width = max(xs) - min(xs)
    elif len(baseline) > 1:
        line_width = baseline[-1][0] - baseline[0][0]
    else:
        line_width = 0

    confidences = getattr(record, "confidences", None)
    avg_confidence = None
    if confidences:
        avg_confidence = round(float(sum(confidences) / len(confidences)), 4)

    return {
        "text": record.prediction,
        "boundary": boundary,
        "baseline": baseline,
        "font_size": max(12, int(height * 0.75)),
        "line_width": max(1, line_width),
        "avg_confidence": avg_confidence,
    }


def stream_ocr(
    image_bytes: bytes,
    cancel_event: threading.Event = None,
) -> Generator[dict, None, None]:
    def cancelled():
        return cancel_event is not None and cancel_event.is_set()

    if cancelled():
        return

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from kraken import blla

    pil_im = Image.open(io.BytesIO(image_bytes))
    im = pil_im.convert("RGB")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        seg = blla.segment(im)

    if cancelled():
        return

    yield {"type": "init", "width": im.width, "height": im.height}

    line_count = 0
    for record in _rec_model.predict(im, seg):
        if cancelled():
            return
        if not record.baseline:
            continue
        yield {"type": "line", **_line_geometry(record)}
        line_count += 1

    yield {"type": "done", "line_count": line_count}


def run_ocr(image_bytes: bytes) -> dict:
    import base64
    pil_im = Image.open(io.BytesIO(image_bytes))
    mime = f"image/{(pil_im.format or 'PNG').lower()}"
    image_b64 = base64.b64encode(image_bytes).decode()

    lines = []
    result = {}
    for event in stream_ocr(image_bytes):
        if event["type"] == "init":
            result["width"] = event["width"]
            result["height"] = event["height"]
        elif event["type"] == "line":
            lines.append({k: v for k, v in event.items() if k != "type"})
        elif event["type"] == "done":
            result["line_count"] = event["line_count"]
    result["lines"] = lines
    result["image_b64"] = image_b64
    result["mime"] = mime
    return result
