"""Hotspot endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Response

from .service import get_index

router = APIRouter(prefix="/api/v1/hotspots", tags=["hotspots"])


def _parse_bbox(bbox: Optional[str]) -> Optional[List[float]]:
    if not bbox:
        return None
    try:
        vals = [float(v) for v in bbox.split(",")]
    except ValueError:
        raise HTTPException(422, "bbox must be 'west,south,east,north'")
    if len(vals) != 4 or vals[0] >= vals[2] or vals[1] >= vals[3]:
        raise HTTPException(422, "bbox must be 'west,south,east,north' with west < east and south < north")
    return vals


@router.get("")
def list_hotspots(
    bbox: Optional[str] = Query(None, description="west,south,east,north in degrees"),
    top: Optional[int] = Query(None, ge=1, le=100_000),
    min_pop: float = Query(0.0, ge=0),
):
    idx = get_index()
    if not bbox and not top and not min_pop:
        return Response(idx.full_json(), media_type="application/json")
    return idx.query(_parse_bbox(bbox), top, min_pop)


@router.get("/{cell_id}")
def hotspot_detail(cell_id: str):
    detail = get_index().detail(cell_id)
    if detail is None:
        raise HTTPException(404, f"Unknown cell '{cell_id}'")
    return detail
