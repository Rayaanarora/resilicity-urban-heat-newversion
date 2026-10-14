"""Work-order export endpoints (JSON, CSV bill of quantities, PDF)."""

import csv
import io
import json
from typing import Any, Dict, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from optimizer.routes import OptimizeRequest
from optimizer.service import optimize_portfolio
from projects import db
from projects.service import project_view

from .content import boq_rows, portfolio_work_order, project_work_order
from .pdf import render_pdf

router = APIRouter(tags=["exports"])
Format = Literal["json", "csv", "pdf"]


def _respond(order: Dict[str, Any], fmt: str, stem: str) -> Response:
    if fmt == "json":
        body, media = json.dumps(order, indent=2).encode("utf-8"), "application/json"
    elif fmt == "csv":
        rows = boq_rows(order)
        buf = io.StringIO()
        if rows:
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        body, media = buf.getvalue().encode("utf-8"), "text/csv"
    else:
        body, media = render_pdf(order), "application/pdf"
    return Response(body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{stem}.{fmt}"'})


@router.get("/api/v1/projects/{project_id}/export")
def export_project(project_id: str, format: Format = Query("json")):
    p = db.get_project(project_id)
    if p is None:
        raise HTTPException(404, f"Unknown project '{project_id}'")
    return _respond(project_work_order(project_view(p)), format, f"resilicity-work-order-{project_id}")


@router.post("/api/v1/portfolio/export")
def export_portfolio(req: OptimizeRequest, format: Format = Query("json")):
    try:
        result = optimize_portfolio(req.project_ids, req.budget_lakh, req.objective, req.phase_limit, req.target_person_c)
    except KeyError as e:
        raise HTTPException(404, f"Unknown project(s): {e.args[0]}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    views = {pid: project_view(db.get_project(pid)) for pid in req.project_ids}
    return _respond(portfolio_work_order(result, views), format, "resilicity-portfolio-work-order")
