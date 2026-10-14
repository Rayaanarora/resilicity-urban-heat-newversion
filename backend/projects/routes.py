"""Project endpoints."""

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from . import db
from .service import ProjectError, create_project, generate_redesign, project_view

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


def _get_or_404(project_id: str):
    p = db.get_project(project_id)
    if p is None:
        raise HTTPException(404, f"Unknown project '{project_id}'")
    return p


@router.post("", status_code=201)
async def create(
    photo: UploadFile = File(...),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    cell_id: Optional[str] = Form(None),
    road_width_m: Optional[float] = Form(None),
    climate_zone: Optional[str] = Form(None),
    budget_lakh: Optional[float] = Form(None),
    name: Optional[str] = Form(None),
):
    content = await photo.read()
    try:
        # Segmentation and planning are CPU-bound; keep the event loop free.
        return await run_in_threadpool(
            create_project, content, photo.filename or "photo.jpg", lat, lon, cell_id or None,
            road_width_m, climate_zone, budget_lakh, name,
        )
    except ProjectError as e:
        raise HTTPException(422, str(e))


@router.get("")
def list_all():
    out = []
    for p in db.list_projects():
        v = project_view(p)
        impl = v["implementation_plan"] or {}
        out.append({
            "id": v["id"], "name": v["name"], "created_at": v["created_at"], "lat": v["lat"], "lon": v["lon"],
            "cell_id": v["cell_id"], "hotspot_rank": (v["hotspot"] or {}).get("rank"),
            "lst_pred_c": (v["hotspot"] or {}).get("lst_pred_c"), "n_accepted": len(v["accepted"]),
            "n_rejected": len(v["rejected"]), "cost_inr_low": impl.get("total_cost_inr_low"),
            "cost_inr_high": impl.get("total_cost_inr_high"), "redesign_status": v["redesign"]["status"],
            "photo_url": v["photo_url"], "data_mode": v["data_mode"],
        })
    return {"projects": out}


@router.get("/{project_id}")
def detail(project_id: str):
    return project_view(_get_or_404(project_id), include_runs=True)


@router.get("/{project_id}/photo")
def photo(project_id: str):
    return FileResponse(_get_or_404(project_id)["photo_path"])


@router.post("/{project_id}/redesign-image", status_code=202)
def request_redesign(project_id: str, background: BackgroundTasks, quality_tier: str = "fast"):
    p = _get_or_404(project_id)
    if p["redesign_status"] in ("queued", "running"):
        return {"status": p["redesign_status"]}
    db.update_project(project_id, redesign_status="queued", redesign_error=None)
    background.add_task(generate_redesign, project_id, quality_tier)
    return {"status": "queued", "poll": f"/api/v1/projects/{project_id}"}


@router.get("/{project_id}/redesign-image")
def redesign_image(project_id: str):
    p = _get_or_404(project_id)
    if p["redesign_status"] != "ready" or not p["redesign_path"] or not Path(p["redesign_path"]).exists():
        raise HTTPException(404, f"No illustrative image yet (status: {p['redesign_status']})")
    return FileResponse(p["redesign_path"], media_type="image/jpeg")


@router.delete("/{project_id}")
def delete(project_id: str):
    p = _get_or_404(project_id)
    db.delete_project(project_id)
    shutil.rmtree(Path(p["photo_path"]).parent, ignore_errors=True)
    return {"deleted": project_id}
