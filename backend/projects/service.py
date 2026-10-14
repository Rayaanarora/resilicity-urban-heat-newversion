"""Project (street workspace) service: create from a photo, persist the gated plan, generate the image on demand."""

import hashlib
import io
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from PIL import Image

from analysis_service import run_site_analysis
from buildable import cooling_band
from hotspots import get_index
from image_generation import RedesignEngine, SpatialDesignPlan, build_kontext_redesign_prompt, illustrated_zones

from . import db

logger = logging.getLogger("resilicity.projects")

MAX_PHOTO_BYTES = 20 * 1024 * 1024
LINK_RADIUS_M = 1000.0  # link a free lat/lon to the nearest hotspot cell within this distance

_engine: Optional[RedesignEngine] = None


def _redesign_engine() -> RedesignEngine:
    global _engine
    if _engine is None:
        _engine = RedesignEngine()
    return _engine


class ProjectError(ValueError):
    pass


def _hotspot_for(cell_id: Optional[str], lat: Optional[float], lon: Optional[float]) -> Optional[Dict[str, Any]]:
    idx = get_index()
    if cell_id:
        cell = idx.cell(cell_id)
        if cell is None:
            raise ProjectError(f"Unknown hotspot cell '{cell_id}'")
        return {**cell, "link": "selected", "data_mode": idx.grid.data_mode}
    if lat is not None and lon is not None:
        near = idx.nearest_cell(lat, lon)
        if near["distance_m"] <= LINK_RADIUS_M:
            return {**near, "link": "nearest", "data_mode": idx.grid.data_mode}
    return None


def create_project(
    photo: bytes,
    filename: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    cell_id: Optional[str] = None,
    road_width_m: Optional[float] = None,
    climate_zone: Optional[str] = None,
    budget_lakh: Optional[float] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Segmentation + scene understanding + gated plan. No image generation (that is on demand)."""
    if len(photo) > MAX_PHOTO_BYTES:
        raise ProjectError("Photo too large (max 20 MB)")
    try:
        image = Image.open(io.BytesIO(photo)).convert("RGB")
    except Exception as e:
        raise ProjectError(f"Invalid image: {e}")
    if cell_id is None and (lat is None or lon is None):
        raise ProjectError("Provide a hotspot cell_id or both lat and lon")

    hotspot = _hotspot_for(cell_id, lat, lon)
    if hotspot and (lat is None or lon is None):
        lat, lon = hotspot["lat"], hotspot["lon"]

    project_id = uuid.uuid4().hex[:12]
    folder = db.files_dir(project_id)
    ext = (filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg")[:5]
    photo_path = folder / f"photo.{ext}"
    photo_path.write_bytes(photo)

    started = time.time()
    site = run_site_analysis(image, road_width_m=road_width_m, climate_zone=climate_zone, budget_lakh=budget_lakh)
    plan = site.plan.dict()

    db.insert_project({
        "id": project_id,
        "name": name or (f"Street at {lat:.4f}, {lon:.4f}" if lat is not None else filename),
        "lat": lat,
        "lon": lon,
        "cell_id": hotspot["cell_id"] if hotspot else None,
        "road_width_m": road_width_m,
        "climate_zone": climate_zone,
        "budget_lakh": budget_lakh,
        "photo_path": str(photo_path),
        "photo_sha256": hashlib.sha256(photo).hexdigest(),
        "hotspot": hotspot,
        "analysis": {
            "surfaces": site.surfaces,
            "segmentation_ok": site.segmentation_ok,
            "scene_analysis": site.analysis.dict(),
            "image_size": list(image.size),
        },
        "plan": plan,
        "thermal": site.thermal,
    })
    db.log_run(project_id, "analysis", "ok", {"n_interventions": len(plan["interventions"])}, started)
    return project_view(db.get_project(project_id))


# ---------------------------------------------------------------------------------------------------------------
# Serialisation with source tags and bands
# ---------------------------------------------------------------------------------------------------------------
def _accepted_items(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    impl = plan.get("implementation_plan") or {}
    coverage = {(iv["type"], iv.get("target_zone")): iv.get("coverage", 0.5) for iv in plan.get("interventions", [])}
    items = []
    for li in impl.get("line_items", []):
        band = cooling_band(li["type"], coverage.get((li["type"], li.get("target_zone")), 0.5))
        lo, hi = li["cost_inr_low"], li["cost_inr_high"]
        items.append({
            "type": li["type"],
            "title": li["title"],
            "target_zone": li.get("target_zone"),
            "quantity": {"value": li["quantity"], "unit": li["unit"], "basis": li["basis"],
                         "source": "assumed" if li["quantity_flagged"] else "rule"},
            "cost_inr": {"low": lo, "central": round((lo + hi) / 2), "high": hi, "source": "assumed"},
            "cooling_c": {"low": band[0], "central": band[1], "high": band[2], "source": "literature"} if band else None,
            "phase": li["phase"],
            "phase_label": li["phase_label"],
            "lifespan_years": li["lifespan_years"],
            "permits": li["permits"],
            "conditions": li["conditions"],
            "maintenance": li["maintenance"],
        })
    return items


def project_view(p: Dict[str, Any], include_runs: bool = False) -> Dict[str, Any]:
    plan = p.get("plan") or {}
    metrics = plan.get("street_metrics") or {}
    hotspot = p.get("hotspot")
    view = {
        "id": p["id"],
        "name": p["name"],
        "created_at": p["created_at"],
        "updated_at": p["updated_at"],
        "lat": p["lat"],
        "lon": p["lon"],
        "cell_id": p["cell_id"],
        "inputs": {"road_width_m": p["road_width_m"], "climate_zone": p["climate_zone"], "budget_lakh": p["budget_lakh"]},
        "hotspot": hotspot,
        "data_mode": hotspot["data_mode"] if hotspot else None,
        "photo_url": f"/api/v1/projects/{p['id']}/photo",
        "street_metrics": {
            **metrics,
            "source": "measured" if metrics.get("scale_available") else "assumed",
            "road_width_source": "measured" if metrics.get("scale_source") == "user" else "assumed",
        } if metrics else None,
        "accepted": _accepted_items(plan),
        "rejected": [
            {"type": r["type"], "target_zone": r.get("target_zone"), "rule": r.get("rule"), "reason": r["reason"],
             "measured": r.get("measured", {})}
            for r in plan.get("rejected_candidates", []) if r.get("rule")
        ],
        "implementation_plan": plan.get("implementation_plan"),
        "thermal": p.get("thermal"),
        "analysis": p.get("analysis"),
        "plan": plan,
        "redesign": {
            "status": p["redesign_status"],
            "image_url": f"/api/v1/projects/{p['id']}/redesign-image" if p["redesign_status"] == "ready" else None,
            "error": p["redesign_error"],
            **(p.get("redesign_meta") or {}),
        },
        "sources": {
            "hotspot.lst_pred_c": "model",
            "street_metrics": "measured (from the photo, with the stated band)",
            "quantity": "rule",
            "cost_inr": "assumed (indicative rates)",
            "cooling_c": "literature",
        },
    }
    if include_runs:
        view["runs"] = db.list_runs(p["id"])
        view["measurements"] = db.list_measurements(p["id"])
    return view


# ---------------------------------------------------------------------------------------------------------------
# On-demand illustrative image
# ---------------------------------------------------------------------------------------------------------------
async def generate_redesign(project_id: str, quality_tier: str = "fast") -> None:
    """Background task: render the illustrative image through the cached FLUX.1 Kontext engine."""
    p = db.get_project(project_id)
    if p is None:
        return
    started = time.time()
    db.update_project(project_id, redesign_status="running", redesign_error=None)
    try:
        plan = SpatialDesignPlan(**p["plan"])
        prompt = build_kontext_redesign_prompt(plan, (p["plan"].get("scene_understanding") or {}).get("protected_objects_summary"))
        if not prompt:
            raise RuntimeError("No buildable interventions to illustrate.")
        photo = open(p["photo_path"], "rb").read()
        image = Image.open(io.BytesIO(photo)).convert("RGB")
        result = await _redesign_engine().generate(image, photo, prompt, quality_tier=quality_tier)
        if result.image is None:
            raise RuntimeError(result.error or "Image generation failed")
        out = db.files_dir(project_id) / "redesign.jpg"
        result.image.save(out, quality=92)
        meta = {"provider": result.provider, "model": result.model, "cached": result.cached,
                "illustrated_interventions": list(illustrated_zones(plan)), "quality_tier": quality_tier}
        db.update_project(project_id, redesign_status="ready", redesign_path=str(out), redesign_meta=meta)
        db.log_run(project_id, "redesign", "ok", meta, started)
    except Exception as e:
        logger.warning("Redesign for project %s failed: %s", project_id, e)
        db.update_project(project_id, redesign_status="failed", redesign_error=str(e)[:500])
        db.log_run(project_id, "redesign", "failed", {"error": str(e)[:500]}, started)
