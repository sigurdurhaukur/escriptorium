import asyncio
import base64
import hashlib
import html as _html
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import fitz

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import inference

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger("ocr_demo")

# ── Constants ──────────────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "images"
RESULTS_DIR = DATA_DIR / "results"
META_DIR = DATA_DIR / "meta"
LOGS_DIR = DATA_DIR / "logs"
FEEDBACK_DIR = DATA_DIR / "feedback"
PDFS_DIR = DATA_DIR / "pdfs"

JOB_TIMEOUT = 120.0
MAX_PDF_PAGES = 50
PDF_PREVIEW_TTL = 1800  # 30 min

MAX_FILE_SIZE = int(os.environ.get("OCR_DEMO_MAX_FILE_SIZE", 50 * 1024 * 1024))

ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "image/webp",
    "application/pdf",
}

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}

BLOCKED_EXTENSIONS = {
    ".exe", ".bat", ".sh", ".js", ".html", ".htm",
    ".jar", ".class", ".py", ".php", ".asp", ".aspx",
    ".cgi", ".pl", ".rb", ".dll", ".zip", ".tar", ".gz",
    ".7z", ".rar",
}

# Rate limiting: max requests per IP per window
RATE_LIMIT_WINDOW = int(os.environ.get("OCR_DEMO_RATE_WINDOW", 60))
RATE_LIMIT_MAX = int(os.environ.get("OCR_DEMO_RATE_MAX", 10))


# ── Job ────────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    job_id: str
    image_hash: str
    image_bytes: bytes
    cancel_event: threading.Event
    output_q: asyncio.Queue = field(default_factory=asyncio.Queue)
    priority: int = 0


# ── Global state ───────────────────────────────────────────────────────────────

_active_loop: asyncio.AbstractEventLoop = None
_job_queue: asyncio.PriorityQueue = None
_pending_jobs: list[Job] = []
_active_jobs: dict[str, Job] = {}
_next_priority: int = 0

# Rate limiter: {ip: [timestamp, ...]}
_rate_limiter: dict[str, list[float]] = {}

# PDF preview token → {path, created_at}
_pdf_previews: dict[str, dict] = {}


# ── Disk helpers ───────────────────────────────────────────────────────────────


def _ensure_dirs():
    for d in (IMAGES_DIR, RESULTS_DIR, META_DIR, LOGS_DIR, FEEDBACK_DIR, PDFS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _image_ext(image_bytes: bytes) -> str:
    from PIL import Image
    import io
    fmt = Image.open(io.BytesIO(image_bytes)).format or "PNG"
    return "jpg" if fmt.upper() == "JPEG" else fmt.lower()


def _pdf_to_thumbnails(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count
    if page_count > MAX_PDF_PAGES:
        doc.close()
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {page_count} pages. Maximum is {MAX_PDF_PAGES}.",
        )

    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=72)  # ~200px at letter size
        img_bytes = pix.tobytes("jpeg", jpg_quality=70)
        data_url = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode()}"
        pages.append({"page": i + 1, "thumbnail": data_url})
    doc.close()
    return pages


def _extract_pdf_page(pdf_bytes: bytes, page_number: int) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_number < 1 or page_number > doc.page_count:
        doc.close()
        raise HTTPException(status_code=400, detail=f"Page {page_number} is out of range.")
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("jpeg", jpg_quality=90)
    doc.close()
    return img_bytes


def _generate_pdf_token() -> str:
    return uuid.uuid4().hex


def _save_image(image_hash: str, image_bytes: bytes, filename: str):
    ext = _image_ext(image_bytes)
    path = IMAGES_DIR / f"{image_hash}.{ext}"
    if not path.exists():
        path.write_bytes(image_bytes)
    meta_path = META_DIR / f"{image_hash}.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "filename": filename,
            "uploaded_at": time.time(),
            "donated": False,
        }))


def _load_result(image_hash: str) -> dict | None:
    path = RESULTS_DIR / f"{image_hash}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _save_result(image_hash: str, result: dict):
    (RESULTS_DIR / f"{image_hash}.json").write_text(json.dumps(result))


def _save_log(log_data: dict):
    job_id = log_data["job_id"]
    (LOGS_DIR / f"{job_id}.json").write_text(json.dumps(log_data, indent=2))


def _save_feedback(image_hash: str, feedback: dict):
    path = FEEDBACK_DIR / f"{image_hash}.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[feedback["job_id"]] = feedback
    path.write_text(json.dumps(existing, indent=2))


# ── File validation ────────────────────────────────────────────────────────────


def _validate_file(filename: str, content_type: str, size: int):
    if size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size} bytes). Maximum is {MAX_FILE_SIZE} bytes.",
        )

    ext = Path(filename).suffix.lower()
    if ext in BLOCKED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' is not allowed.",
        )

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{ext}' is not supported. "
                   f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    mime = content_type.split(";")[0].strip().lower()
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Content type '{mime}' is not supported.",
        )


# ── Rate limiter ───────────────────────────────────────────────────────────────


def _check_rate_limit(ip: str):
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    timestamps = _rate_limiter.get(ip, [])
    timestamps = [t for t in timestamps if t > window_start]
    _rate_limiter[ip] = timestamps

    if len(timestamps) >= RATE_LIMIT_MAX:
        retry_after = int(timestamps[0] + RATE_LIMIT_WINDOW - now)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    timestamps.append(now)


# ── SSE helpers ────────────────────────────────────────────────────────────────


def _sse(event: str, data: str) -> str:
    lines = "\n".join(f"data: {line}" for line in data.splitlines())
    return f"event: {event}\n{lines}\n\n"


def _build_line_svg(line: dict) -> str:
    parts = []
    if line["boundary"]:
        pts = " ".join(f"{p[0]},{p[1]}" for p in line["boundary"])
        parts.append(
            f'<polygon points="{pts}" fill="rgba(59, 130, 246, 0.06)"'
            f' stroke="rgba(59, 130, 246, 0.3)" stroke-width="1.5"/>'
        )
    if line["baseline"] and line["text"]:
        x, y = line["baseline"][0]
        text = _html.escape(line["text"])
        parts.append(
            f'<text x="{x}" y="{y}" font-size="{line["font_size"]}"'
            f' textLength="{line["line_width"]}" lengthAdjust="spacingAndGlyphs"'
            f' font-family="Georgia, serif" fill="#1e293b">{text}</text>'
        )
    return "".join(parts)


def _events_from_result(result: dict):
    yield {"type": "init", "width": result["width"], "height": result["height"]}
    for line in result["lines"]:
        yield {"type": "line", **line}
    yield {"type": "done", "line_count": len(result["lines"])}


# ── PDF cleanup ─────────────────────────────────────────────────────────────────


async def _cleanup_stale_pdfs():
    while True:
        await asyncio.sleep(300)
        now = time.time()
        stale = [t for t, p in list(_pdf_previews.items())
                 if now - p["created_at"] > PDF_PREVIEW_TTL]
        for token in stale:
            entry = _pdf_previews.pop(token, None)
            if entry:
                try:
                    entry["path"].unlink()
                except OSError:
                    pass
                logger.info("Cleaned up stale PDF preview %s", token[:12])


# ── Worker ─────────────────────────────────────────────────────────────────────


def _run_job(job: Job):
    """Runs OCR synchronously in a thread, pushes events into job.output_q."""
    lines = []
    width = height = 0
    start_ts = time.time()
    try:
        for event in inference.stream_ocr(job.image_bytes, job.cancel_event):
            _active_loop.call_soon_threadsafe(job.output_q.put_nowait, event)
            if event["type"] == "init":
                width, height = event["width"], event["height"]
            elif event["type"] == "line":
                lines.append({k: v for k, v in event.items() if k != "type"})
            elif event["type"] == "done" and not job.cancel_event.is_set():
                _save_result(job.image_hash, {
                    "width": width, "height": height, "lines": lines,
                })
    except Exception as e:
        _active_loop.call_soon_threadsafe(
            job.output_q.put_nowait, {"type": "job-error", "message": str(e)}
        )
        logger.exception("Job %s failed", job.job_id)
    finally:
        elapsed = time.time() - start_ts

        # Build detailed log entry
        log_entry = {
            "job_id": job.job_id,
            "image_hash": job.image_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "line_count": len(lines),
            "lines": lines,
            "image_width": width,
            "image_height": height,
        }
        _save_log(log_entry)
        logger.info(
            "Job %s | hash=%s | lines=%d | elapsed=%.2fs",
            job.job_id, job.image_hash[:12], len(lines), elapsed,
        )

        _active_loop.call_soon_threadsafe(job.output_q.put_nowait, None)


async def _worker():
    while True:
        try:
            _priority, job = await _job_queue.get()

            try:
                _pending_jobs.remove(job)
            except ValueError:
                pass

            if job.cancel_event.is_set():
                job.output_q.put_nowait(None)
                _job_queue.task_done()
                continue

            for i, waiting in enumerate(_pending_jobs):
                waiting.output_q.put_nowait({"type": "queue", "position": i + 1})

            loop = asyncio.get_running_loop()
            executor = loop.run_in_executor(None, _run_job, job)
            try:
                await asyncio.wait_for(executor, timeout=JOB_TIMEOUT)
            except (asyncio.TimeoutError, TimeoutError):
                job.cancel_event.set()
                job.output_q.put_nowait({"type": "job-error", "message": "Job timed out after 120 s"})
                job.output_q.put_nowait(None)
                logger.warning("Job %s timed out", job.job_id)

            _job_queue.task_done()

        except Exception as exc:
            logger.exception("Worker unhandled error: %s", exc)


# ── App ────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _active_loop, _job_queue
    _active_loop = asyncio.get_running_loop()
    _job_queue = asyncio.PriorityQueue()
    _ensure_dirs()
    inference.load_models()
    asyncio.create_task(_worker())
    asyncio.create_task(_cleanup_stale_pdfs())
    yield


app = FastAPI(title="OCR Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["points"] = lambda pts: " ".join(
    f"{p[0]},{p[1]}" for p in (pts or [])
)


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/pdf-preview")
async def pdf_preview(request: Request, file: UploadFile = File(...)):
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    raw = await file.read()
    ext = Path(file.filename or "upload.pdf").suffix.lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF.")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"PDF too large ({len(raw)} bytes). Maximum is {MAX_FILE_SIZE} bytes.",
        )

    pages = _pdf_to_thumbnails(raw)

    token = _generate_pdf_token()
    pdf_path = PDFS_DIR / f"{token}.pdf"
    pdf_path.write_bytes(raw)
    _pdf_previews[token] = {"path": pdf_path, "created_at": time.time()}

    return JSONResponse({"token": token, "pages": pages, "page_count": len(pages)})


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(None),
    preview_token: str = Form(""),
    page_number: int = Form(1),
):
    ip = request.client.host if request.client else "unknown"
    _check_rate_limit(ip)

    if preview_token:
        preview = _pdf_previews.get(preview_token)
        if not preview:
            raise HTTPException(status_code=400, detail="Preview token expired or invalid.")
        pdf_bytes = preview["path"].read_bytes()
        raw = _extract_pdf_page(pdf_bytes, page_number)
        source_name = f"page_{page_number}.jpg"
    else:
        raw = await file.read()
        _validate_file(file.filename or "upload", file.content_type or "", len(raw))
        source_name = file.filename or "upload"

    scaled = inference.scale_image(raw)
    image_hash = hashlib.sha256(scaled).hexdigest()
    _save_image(image_hash, scaled, source_name)

    job_id = str(uuid.uuid4())
    global _next_priority
    _next_priority += 1

    job = Job(
        job_id=job_id,
        image_hash=image_hash,
        image_bytes=scaled,
        cancel_event=threading.Event(),
        priority=-_next_priority,
    )
    _active_jobs[job_id] = job

    cached = _load_result(image_hash)
    if cached:
        for event in _events_from_result(cached):
            job.output_q.put_nowait(event)
        job.output_q.put_nowait(None)
        logger.info("Job %s served from cache (hash=%s)", job_id, image_hash[:12])
        return JSONResponse({"job_id": job_id, "cached": True})

    _pending_jobs.append(job)
    await _job_queue.put((job.priority, job))
    queue_pos = len(_pending_jobs)

    logger.info(
        "Job %s queued | hash=%s | filename=%s | size=%d | position=%d",
        job_id, image_hash[:12], source_name, len(raw), queue_pos,
    )

    return JSONResponse({"job_id": job_id, "cached": False, "queue_position": queue_pos})


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.cancel_event.set()
    try:
        _pending_jobs.remove(job)
        job.output_q.put_nowait(None)
        for i, waiting in enumerate(_pending_jobs):
            waiting.output_q.put_nowait({"type": "queue", "position": i + 1})
    except ValueError:
        pass
    logger.info("Job %s cancelled", job_id)
    return JSONResponse({"ok": True})


@app.post("/donate/{job_id}")
async def donate(job_id: str):
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    path = META_DIR / f"{job.image_hash}.json"
    if path.exists():
        meta = json.loads(path.read_text())
        meta["donated"] = True
        path.write_text(json.dumps(meta))
    return JSONResponse({"ok": True})


@app.post("/feedback/{job_id}")
async def feedback(job_id: str, request: Request):
    job = _active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    body = await request.json()
    rating = body.get("rating")
    if rating not in ("thumbs_up", "thumbs_down"):
        raise HTTPException(status_code=400, detail="Rating must be 'thumbs_up' or 'thumbs_down'")

    feedback_entry = {
        "job_id": job_id,
        "image_hash": job.image_hash,
        "rating": rating,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _save_feedback(job.image_hash, feedback_entry)

    logger.info("Feedback for job %s hash=%s rating=%s", job_id, job.image_hash[:12], rating)
    return JSONResponse({"ok": True})


@app.get("/stream/{job_id}")
async def stream_job(job_id: str):
    job = _active_jobs.get(job_id)
    if not job:
        return HTMLResponse("Job not found", status_code=404)

    async def generate():
        if job in _pending_jobs:
            pos = _pending_jobs.index(job) + 1
            yield _sse("queue", str(pos))

        try:
            while True:
                event = await job.output_q.get()
                if event is None:
                    break
                if event["type"] == "init":
                    svg = (
                        f'<svg id="transcription-svg"'
                        f' viewBox="0 0 {event["width"]} {event["height"]}"'
                        f' style="width:100%;height:auto;display:block;background:white;"'
                        f' overflow="hidden"></svg>'
                    )
                    yield _sse("init", svg)
                elif event["type"] == "line":
                    payload = json.dumps({
                        "svg": _build_line_svg(event),
                        "text": event["text"],
                        "avg_confidence": event.get("avg_confidence"),
                    })
                    yield _sse("line", payload)
                elif event["type"] == "done":
                    yield _sse("done", str(event["line_count"]))
                elif event["type"] == "job-error":
                    yield _sse("job-error", event["message"])
        finally:
            pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
