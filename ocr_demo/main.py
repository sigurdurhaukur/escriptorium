from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import inference

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_models()
    yield


app = FastAPI(title="OCR Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["points"] = lambda pts: " ".join(
    f"{p[0]},{p[1]}" for p in (pts or [])
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = inference.run_ocr(contents)
        return templates.TemplateResponse(request, "result.html", result)
    except Exception as e:
        return HTMLResponse(
            f'<div class="error-msg">Error processing image: {e}</div>',
            status_code=500,
        )
