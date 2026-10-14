"""Portfolio optimisation endpoint."""

from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .service import optimize_portfolio

router = APIRouter(prefix="/api/v1/portfolio", tags=["portfolio"])


class OptimizeRequest(BaseModel):
    project_ids: List[str] = Field(..., min_length=1)
    budget_lakh: float = Field(..., gt=0)
    objective: Literal["person_c", "max_cooling", "min_cost_for_target"] = "person_c"
    phase_limit: Optional[Literal["quick_win", "medium", "long_term"]] = None
    target_person_c: Optional[float] = Field(None, gt=0)


@router.post("/optimize")
def optimize(req: OptimizeRequest):
    try:
        return optimize_portfolio(req.project_ids, req.budget_lakh, req.objective, req.phase_limit, req.target_person_c)
    except KeyError as e:
        raise HTTPException(404, f"Unknown project(s): {e.args[0]}")
    except ValueError as e:
        raise HTTPException(422, str(e))
