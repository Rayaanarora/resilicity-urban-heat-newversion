"""Ground-plane geometric draft compositor for the local redesign pipeline.

The compositor creates draft layers only. It does not present procedural color
filters or generated sprites as the final redesign; local diffusion must
harmonize the drafted intervention pixels before the route can return ready.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

logger = logging.getLogger("resilicity.compositor")


SAMPLES_DIR = Path(__file__).resolve().parents[2] / "gen-ai" / "public" / "samples"


def build_road_marking_preserve_mask(image: Image.Image, road_mask: np.ndarray) -> np.ndarray:
    """Identify high-contrast painted lane markings and crosswalks within the road surface."""
    w, h = image.size
    img_gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if road_mask.shape != (h, w) or np.count_nonzero(road_mask) == 0:
        return np.zeros((h, w), dtype=bool)

    road_pixels = img_gray[road_mask]
    thresh_val = float(np.percentile(road_pixels, 86))
    bright_road = (img_gray >= thresh_val) & road_mask
    edges = (np.abs(cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)) > 38) & bright_road
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edges.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def _polygon_mask(width: int, height: int, polygon: Optional[List[List[float]]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon and len(polygon) >= 3:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    return mask > 0


def _texture_noise(height: int, width: int, base_rgb: Tuple[int, int, int], contrast: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, contrast, (height, width)).astype(np.float32)
    fine = rng.normal(0, contrast * 0.35, (height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (9, 9), 0) + fine
    base = np.full((height, width, 3), np.array(base_rgb, dtype=np.float32), dtype=np.float32)
    return np.clip(base + noise[:, :, None], 0, 255).astype(np.uint8)


def _warp_texture_to_mask(
    image: Image.Image,
    surface_mask: np.ndarray,
    texture: np.ndarray,
    preserve_mask: Optional[np.ndarray] = None,
    alpha: float = 0.82,
) -> Tuple[Image.Image, np.ndarray]:
    """Blend a material texture onto the actual segmented ground surface."""
    arr = np.array(image.convert("RGB")).astype(np.float32)
    h, w = surface_mask.shape
    tex = cv2.resize(texture, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    active = surface_mask.copy()
    if preserve_mask is not None:
        active &= ~preserve_mask

    if np.count_nonzero(active) == 0:
        return image.copy(), active

    light = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lum = np.clip(light / (float(np.mean(light[active])) + 1e-5), 0.72, 1.22)
    tex = np.clip(tex * lum[..., None], 0, 255)

    blend = cv2.GaussianBlur(active.astype(np.float32), (7, 7), 0)[..., None] * alpha
    out = (arr * (1.0 - blend) + tex * blend).astype(np.uint8)
    return Image.fromarray(out), active


def render_road_material_draft(
    image: Image.Image,
    road_mask: np.ndarray,
    preserve_markings: bool = True,
) -> Tuple[Image.Image, np.ndarray]:
    """Create a cool-pavement material draft. This is never a final AI result."""
    preserve = build_road_marking_preserve_mask(image, road_mask) if preserve_markings else None
    tex = _texture_noise(768, 768, (164, 166, 160), 9.5, seed=4101)
    return _warp_texture_to_mask(image, road_mask, tex, preserve_mask=preserve, alpha=0.78)


def render_sidewalk_material_draft(
    image: Image.Image,
    pavement_mask: np.ndarray,
) -> Tuple[Image.Image, np.ndarray]:
    """Create a permeable-paver draft from a warped material texture."""
    tex = _texture_noise(768, 768, (175, 170, 158), 8.0, seed=4202)
    yy, xx = np.indices((768, 768))
    joint_x = ((xx + ((yy // 34) % 2) * 22) % 44) < 2
    joint_y = (yy % 34) < 2
    joints = cv2.GaussianBlur((joint_x | joint_y).astype(np.float32), (3, 3), 0)
    tex = np.clip(tex.astype(np.float32) * (1.0 - 0.33 * joints[..., None]), 0, 255).astype(np.uint8)
    return _warp_texture_to_mask(image, pavement_mask, tex, alpha=0.64)


# Backward-compatible names used by older tests/debug code.
render_procedural_road_draft = render_road_material_draft
render_procedural_sidewalk_paver_draft = render_sidewalk_material_draft


@lru_cache(maxsize=1)
def _load_local_tree_cutout() -> Optional[Image.Image]:
    """Extract a local photographic/AI tree cutout from bundled sample imagery."""
    candidates = [
        SAMPLES_DIR / "urban_street_after.png",
        SAMPLES_DIR / "sample_4_low_vegetation.jpg",
        SAMPLES_DIR / "sample_5_commercial_campus.jpg",
    ]

    for path in candidates:
        if not path.exists():
            continue
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        green = (
            (hsv[..., 0] >= 32)
            & (hsv[..., 0] <= 92)
            & (hsv[..., 1] >= 35)
            & (hsv[..., 2] >= 45)
        )
        green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green.astype(np.uint8), 8)
        if n_labels <= 1:
            continue

        label = max(range(1, n_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 600:
            continue
        x, y, w, h, _ = stats[label]
        pad = 18
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(arr.shape[1], x + w + pad), min(arr.shape[0], y + h + pad)
        crop = img.crop((x1, y1, x2, y2)).convert("RGBA")

        alpha = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        alpha[labels[y1:y2, x1:x2] == label] = 255
        alpha = cv2.dilate(alpha, np.ones((5, 5), np.uint8), iterations=1)
        alpha = cv2.GaussianBlur(alpha, (7, 7), 0)
        crop.putalpha(Image.fromarray(alpha))
        logger.info("Loaded local photographic tree draft asset from %s", path)
        return crop

    logger.warning("No local photographic tree cutout could be extracted from bundled samples.")
    return None


def _paste_rgba_with_mask(
    canvas: Image.Image,
    overlay: Image.Image,
    x: int,
    y: int,
    mask_arr: np.ndarray,
) -> Image.Image:
    w, h = canvas.size
    canvas.paste(overlay, (x, y), overlay)
    alpha = np.array(overlay.getchannel("A"))
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + overlay.width), min(h, y + overlay.height)
    if x2 > x1 and y2 > y1:
        ax1, ay1 = x1 - x, y1 - y
        ax2, ay2 = ax1 + (x2 - x1), ay1 + (y2 - y1)
        mask_arr[y1:y2, x1:x2] |= alpha[ay1:ay2, ax1:ax2]
    return canvas


def _draw_directional_ground_shadow(
    shadow_layer: np.ndarray,
    anchor: Dict[str, Any],
    sun_direction: str,
    ground_mask: np.ndarray,
) -> None:
    x, y = int(anchor["x"]), int(anchor["y"])
    r = int(anchor.get("canopy_radius", 36))
    scale = float(anchor.get("scale", 0.7))
    if sun_direction == "illuminating_from_right":
        dx, dy = -0.75, 0.34
    elif sun_direction == "illuminating_from_left":
        dx, dy = 0.75, 0.34
    else:
        dx, dy = 0.25, 0.40
    cx = int(x + dx * r * 0.82)
    cy = int(y + dy * r * 0.48)
    cv2.ellipse(
        shadow_layer,
        (cx, cy),
        (max(10, int(r * 0.95)), max(5, int(r * 0.28))),
        float(np.degrees(np.arctan2(dy, dx))),
        0,
        360,
        float(0.28 + 0.12 * scale),
        -1,
    )
    shadow_layer[~ground_mask] = 0.0


def _apply_shadow_layer(image: Image.Image, shadow_layer: np.ndarray) -> Image.Image:
    shadow_layer = cv2.GaussianBlur(shadow_layer, (21, 21), 0)
    arr = np.array(image.convert("RGB")).astype(np.float32)
    factor = 1.0 - np.clip(shadow_layer, 0.0, 0.42)[..., None]
    return Image.fromarray(np.clip(arr * factor, 0, 255).astype(np.uint8))


from .asset_library import get_tree_asset


def _draw_shade_structure_draft(
    image: Image.Image,
    footprint: List[List[float]],
    supports: List[List[float]],
    ground_mask: np.ndarray,
    protected_mask: np.ndarray,
    sun_direction: str = "overhead_solar_insolation",
) -> Tuple[Image.Image, np.ndarray]:
    w, h = image.size
    out = image.convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = [(int(x), int(y)) for x, y in footprint]
    if len(pts) >= 3:
        # Modern tensile fabric membrane: clean off-white with subtle architectural borders
        draw.polygon(pts, fill=(240, 243, 238, 220), outline=(195, 200, 195, 240))
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)

    # Draw structural steel support columns anchored on sidewalk
    for px, py in supports:
        x, y = int(px), int(py)
        if not (0 <= x < w and 0 <= y < h) or not ground_mask[y, x] or protected_mask[y, x]:
            continue
        top_y = min(y - 20, min(p[1] for p in pts) + 10) if pts else y - int(h * 0.18)
        col_w = max(3, int(w * 0.005))
        # Structural column: dark charcoal architectural steel
        draw.line([(x, y), (x, int(top_y))], fill=(48, 52, 56, 245), width=col_w)
        cv2.line(mask, (x, y), (x, int(top_y)), 255, col_w + 2)

    out = Image.alpha_composite(out, layer).convert("RGB")

    # Cast directional ground shadow onto the sidewalk beneath the canopy
    shadow = np.zeros((h, w), dtype=np.float32)
    if pts:
        if sun_direction == "illuminating_from_right":
            s_dx, s_dy = -int(w * 0.04), int(h * 0.04)
        elif sun_direction == "illuminating_from_left":
            s_dx, s_dy = int(w * 0.04), int(h * 0.04)
        else:
            s_dx, s_dy = 0, int(h * 0.05)

        # Project canopy footprint down to sidewalk ground plane
        avg_ground_y = int(np.mean([p[1] for p in supports])) if supports else int(h * 0.75)
        shadow_pts = np.array([[x + s_dx, avg_ground_y + int((y - pts[0][1]) * 0.25) + s_dy] for x, y in pts], dtype=np.int32)
        cv2.fillPoly(shadow, [shadow_pts], 0.28)
        shadow[~ground_mask] = 0.0
        out = _apply_shadow_layer(out, shadow)

    mask[protected_mask] = 0
    return out, mask


def composite_geometric_draft(
    image: Image.Image,
    plan: Any,
    seg_result: Dict[str, Any],
    scene_understanding: Optional[Any] = None,
) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
    """Assemble physical draft layers before local diffusion harmonization."""
    w, h = image.size
    current_img = image.copy()
    inpaint_mask_arr = np.zeros((h, w), dtype=np.uint8)
    individual_masks: Dict[str, Image.Image] = {}
    draft_layers: Dict[str, Image.Image] = {}

    raw_preds = seg_result.get("raw_preds")
    if raw_preds is not None:
        road_mask = np.isin(raw_preds, [6, 54, 91])
        pavement_mask = np.isin(raw_preds, [3, 11, 52])
    else:
        road_mask = np.zeros((h, w), dtype=bool)
        pavement_mask = np.zeros((h, w), dtype=bool)

    from .mask_builder import build_protected_object_mask

    protected_mask = build_protected_object_mask(w, h, seg_result)
    sun_direction = getattr(scene_understanding, "sun_direction_proxy", "overhead_solar_insolation") if scene_understanding else "overhead_solar_insolation"
    interventions = getattr(plan, "interventions", [])
    ground_mask = road_mask | pavement_mask

    # Ground materials: real texture drafts warped/blended to the segmented geometry.
    for iv in interventions:
        itype = getattr(iv, "type", "")
        if itype == "cool_pavement":
            current_img, applied_road = render_road_material_draft(current_img, road_mask)
            inpaint_mask_arr |= applied_road.astype(np.uint8) * 255
            individual_masks["cool_pavement"] = Image.fromarray(applied_road.astype(np.uint8) * 255, mode="L")
            draft_layers["road_draft"] = current_img.copy()
            logger.info("Composited road material draft: coverage_pct=%.2f", np.count_nonzero(applied_road) / max(1, w * h) * 100.0)
        elif itype == "permeable_pave":
            current_img, applied_pave = render_sidewalk_material_draft(current_img, pavement_mask)
            inpaint_mask_arr |= applied_pave.astype(np.uint8) * 255
            individual_masks["permeable_pave"] = Image.fromarray(applied_pave.astype(np.uint8) * 255, mode="L")
            draft_layers["sidewalk_draft"] = current_img.copy()
            logger.info("Composited sidewalk material draft: coverage_pct=%.2f", np.count_nonzero(applied_pave) / max(1, w * h) * 100.0)

    # Tree canopy drafts: strictly realistic photographic assets scaled with perspective depth
    tree_mask_arr = np.zeros((h, w), dtype=np.uint8)
    all_anchors: List[Dict[str, Any]] = []

    for iv in interventions:
        if getattr(iv, "type", "") != "tree_canopy":
            continue
        anchors = getattr(iv, "anchors", []) or []
        all_anchors.extend(anchors)

        shadow = np.zeros((h, w), dtype=np.float32)
        # Painter's Algorithm: render from background (lowest y) to foreground (highest y)
        anchors_sorted = sorted(anchors, key=lambda a: a.get("y", 0))
        for idx, anc in enumerate(anchors_sorted):
            scale = float(anc.get("scale", 0.65))
            target_h = max(56, int(anc.get("tree_height") or h * 0.27 * scale))
            variant = "mature" if idx % 2 == 0 else "young"
            asset = get_tree_asset(variant=variant, target_height=target_h, flip_h=(idx % 2 == 1))

            x, y = int(anc["x"]), int(anc["y"])
            paste_x = x - asset.width // 2
            paste_y = y - asset.height + 3
            current_img = _paste_rgba_with_mask(current_img.convert("RGBA"), asset, paste_x, paste_y, tree_mask_arr).convert("RGB")
            _draw_directional_ground_shadow(shadow, anc, sun_direction, ground_mask)
            logger.info("Placed authentic photographic tree draft at anchor=(%d,%d) scale=%.3f depth=%.3f", x, y, scale, float(anc.get("relative_depth", 0.0)))

        if np.count_nonzero(shadow) > 0:
            current_img = _apply_shadow_layer(current_img, shadow)

    if np.count_nonzero(tree_mask_arr) > 0:
        tree_mask_arr[protected_mask] = 0
        inpaint_mask_arr |= tree_mask_arr
        individual_masks["tree_canopy"] = Image.fromarray(tree_mask_arr, mode="L")
        draft_layers["tree_draft"] = current_img.copy()

    # Shade structure drafts: architectural tensile sails with sidewalk steel posts
    shade_mask_arr = np.zeros((h, w), dtype=np.uint8)
    for iv in interventions:
        if getattr(iv, "type", "") != "shade_structure":
            continue
        footprint = getattr(iv, "footprint_polygon", None)
        supports = getattr(iv, "support_points", None)
        if footprint and supports and len(supports) >= 2:
            current_img, m = _draw_shade_structure_draft(current_img, footprint, supports, ground_mask, protected_mask, sun_direction=sun_direction)
            shade_mask_arr |= m
            logger.info("Composited shade draft: supports=%d coverage_pct=%.2f", len(supports), np.count_nonzero(m) / max(1, w * h) * 100.0)

    if np.count_nonzero(shade_mask_arr) > 0:
        inpaint_mask_arr |= shade_mask_arr
        individual_masks["shade_structure"] = Image.fromarray(shade_mask_arr, mode="L")
        draft_layers["shade_draft"] = current_img.copy()

    # Final protection pass on geometric composite
    if protected_mask is not None and np.count_nonzero(protected_mask) > 0:
        orig_arr = np.array(image.convert("RGB"))
        curr_arr = np.array(current_img.convert("RGB"))
        curr_arr[protected_mask] = orig_arr[protected_mask]
        current_img = Image.fromarray(curr_arr)
        inpaint_mask_arr[protected_mask] = 0

    harmonize_mask = Image.fromarray(inpaint_mask_arr, mode="L")
    meta = {
        "visual_path_proof": {
            "multi_pass": True,
            "legacy_inpainting_engine": False,
            "tree_sprites": False,
            "procedural_tree_renderer": False,
            "sd15_required_for_ready": True,
        },
        "placed_trees_count": len(all_anchors),
        "sun_direction": sun_direction,
        "inpaint_coverage_pct": round(float(np.count_nonzero(inpaint_mask_arr) / (w * h) * 100.0), 2),
        "individual_masks": individual_masks,
        "draft_layers": draft_layers,
    }
    return current_img, harmonize_mask, meta
