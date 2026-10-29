from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..auth import require_auth
from ..state import registry

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/view", tags=["view"])


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: str = Depends(require_auth)) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {"request": request, "title": "autoclip"})


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str, user: str = Depends(require_auth)) -> HTMLResponse:
    job = registry.get(job_id)
    if job is None:
        return templates.TemplateResponse(request, "404.html", {"request": request, "job_id": job_id}, status_code=404)
    return templates.TemplateResponse(request, "job.html", {"request": request, "job": job})
