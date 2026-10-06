"""Stage 3: vision-language urban planner.

A vision model looks at the street photo plus the Stage 1 surface mix and chooses
WHICH cooling interventions fit, where, and how aggressively (qualitative).
All numbers that feed the heat model (land-cover shifts, cooling estimates) are computed
here from the measured surface percentages, never taken from the model's text, so the
LLM cannot invent temperatures. Anything it returns is validated against a fixed catalog.
"""
import asyncio
import base64
import io
import json
import os
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

router = APIRouter()

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "claude-sonnet-5-5")
MAX_IMAGE_BYTES = 15 * 1024 * 1024

# type -> static facts. Numbers are scaled by the surface mix in build_intervention().
CATALOG: dict[str, dict[str, Any]] = {
    "tree_canopy": {"title": "Tree canopy expansion", "target": "pavement", "cost": ("med", "Medium cost")},
    "green_roof": {"title": "Green roof", "target": "roof", "cost": ("high", "High cost")},
    "cool_roof": {"title": "High-albedo cool roof coating", "target": "roof", "cost": ("low", "Low cost")},
    "cool_pavement": {"title": "Reflective pavement coating", "target": "road", "cost": ("med", "Medium cost")},
    "permeable_pave": {"title": "Permeable pavement", "target": "pavement", "cost": ("med", "Medium cost")},
    "shade_structure": {"title": "Shade structures", "target": "pavement", "cost": ("low", "Low cost")},
}
TARGET_NORMALIZATION = {"sidewalk": "pavement", "courtyard": "pavement"}
VALID_TARGETS = {"road", "roof", "pavement", "wall", "vegetation", "water", "sky", "other", "sidewalk", "courtyard"}
MAX_INTERVENTIONS = 5

SYSTEM_PROMPT = (
    "You are an urban heat-resilience planner for hot South Asian cities. You are given a street photo and the "
    "measured surface mix (percent of image area). Choose the interventions that best cool THIS site. "
    "Reply with JSON only, no prose, matching exactly:\n"
    '{"site_summary": "<=2 sentences on what drives heat here",'
    ' "interventions": [{"type": "<one of: ' + ", ".join(CATALOG) + '>",'
    ' "target_region": "<road|roof|pavement|wall|sidewalk|courtyard>",'
    ' "priority": 1,'
    ' "coverage": 0.0-1.0 (share of the applicable area to treat),'
    ' "rationale": "<=1 sentence, grounded in what is visible",'
    ' "inpaint_prompt": "<=15 words describing how the treated area should look in a photo"}]}\n'
    f"Return at most {MAX_INTERVENTIONS} interventions, most impactful first. Only propose what is physically "
    "plausible in the photo (no roof measures if no roof is visible). Do not state temperatures."
)


def _pct(surfaces: dict[str, float], *names: str) -> float:
    return sum(float(surfaces.get(n, 0.0)) for n in names)


def build_intervention(raw: dict[str, Any], surfaces: dict[str, float], index: int) -> dict[str, Any] | None:
    """Validate one model-proposed item and attach deterministic numbers. Returns None if unusable."""
    t = raw.get("type")
    if t not in CATALOG:
        return None
    try:
        coverage = min(1.0, max(0.1, float(raw.get("coverage", 0.6))))
    except (TypeError, ValueError):
        coverage = 0.6

    roof, hard = _pct(surfaces, "roof"), _pct(surfaces, "road", "pavement")
    meta = CATALOG[t]
    shift: dict[str, float] | None = None
    literature: float | None = None

    if t == "tree_canopy":
        if hard + _pct(surfaces, "wall") < 5:
            return None
        s = round(min(coverage * 0.10, 0.5 * hard / 100), 3)
        if s <= 0:
            return None
        shift, impact = {"f_built": -s, "f_tree": s}, 24 * s
    elif t == "green_roof":
        if roof < 2:
            return None
        s = round(coverage * 0.5 * roof / 100, 3)
        shift, impact = {"f_built": -s, "f_grass": s}, min(2.0, 20 * s)
    elif t == "cool_roof":
        if roof < 2:
            return None
        literature, impact = round(1.5 * coverage, 1), 0.8 * coverage
    elif t in ("cool_pavement", "permeable_pave"):
        if hard < 5:
            return None
        literature, impact = round((0.9 if t == "cool_pavement" else 0.7) * coverage, 1), (0.6 if t == "cool_pavement" else 0.5) * coverage
    else:  # shade_structure
        literature, impact = round(1.0 * coverage, 1), 0.5 + 0.5 * coverage

    target = raw.get("target_region")
    target = target if target in VALID_TARGETS else meta["target"]
    target = TARGET_NORMALIZATION.get(target, target)
    try:
        priority = max(1, int(raw.get("priority", index + 1)))
    except (TypeError, ValueError):
        priority = index + 1

    out: dict[str, Any] = {
        "id": f"p-{t}-{index}",
        "type": t,
        "title": meta["title"],
        "targetRegion": target,
        "priority": priority,
        "coverage": round(coverage, 2),
        "estCostTier": meta["cost"][0],
        "estCostText": meta["cost"][1],
        "coolingImpact": round(max(0.2, impact), 1),
        "description": str(raw.get("rationale") or meta["title"])[:240],
        "promptTemplate": str(raw.get("inpaint_prompt") or meta["title"])[:160],
        "defaultEnabled": priority <= 2,
    }
    if shift:
        out["landCoverShift"] = shift
    if literature is not None:
        out["literatureCoolingC"] = literature
    return out


def parse_plan(text: str, surfaces: dict[str, float]) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("planner reply contained no JSON object")
    data = json.loads(text[start : end + 1])
    items = data.get("interventions")
    if not isinstance(items, list):
        raise ValueError("planner reply missing 'interventions' list")

    built, seen = [], set()
    for i, raw in enumerate(items[:MAX_INTERVENTIONS]):
        if not isinstance(raw, dict) or raw.get("type") in seen:
            continue
        item = build_intervention(raw, surfaces, i)
        if item:
            seen.add(item["type"])
            built.append(item)
    if not built:
        raise ValueError("planner proposed no applicable interventions")
    built.sort(key=lambda x: x["priority"])
    for rank, item in enumerate(built, 1):
        item["priority"] = rank
        item["defaultEnabled"] = rank <= 2
    return {"siteSummary": str(data.get("site_summary", ""))[:400], "interventions": built}


def _prepare_image(content: bytes) -> str:
    img = Image.open(io.BytesIO(content)).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call_model(image_b64: str, surfaces: dict[str, float]) -> str:
    import anthropic  # imported lazily so the rest of the API works without the package

    client = anthropic.Anthropic(timeout=45.0)
    mix = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(surfaces.items(), key=lambda kv: -kv[1]) if v > 0)
    msg = client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": f"Measured surface mix: {mix}. Plan the cooling interventions."},
            ],
        }],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def planner_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@router.post("/api/v1/plan")
async def plan(file: UploadFile = File(...), surfaces: str = Form(...)):
    if not planner_available():
        raise HTTPException(503, "Planner unavailable: ANTHROPIC_API_KEY is not set on the server")
    try:
        surf = {str(k): float(v) for k, v in json.loads(surfaces).items()}
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(422, "'surfaces' must be a JSON object of class -> percent")

    content = await file.read()
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image file too large (max 15 MB)")
    try:
        image_b64 = _prepare_image(content)
    except Exception as e:
        raise HTTPException(400, f"Invalid or unreadable image file: {e}")

    try:
        result = parse_plan(await asyncio.to_thread(_call_model, image_b64, surf), surf)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(502, f"Planner returned an unusable plan: {e}")
    except Exception as e:
        raise HTTPException(502, f"Planner request failed: {e}")
    return {"source": "vlm", "model": PLANNER_MODEL, **result}
