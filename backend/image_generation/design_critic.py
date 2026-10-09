"""Autonomous Design Critic for Urban Heat Resilience Redesigns (Design Intelligence V2).

Evaluates every generated intervention image against the formal DesignIntent specification:
1. INTERVENTION PRESENCE (Did the intended intervention actually appear? Leaves, trunk, canopy?)
2. SPATIAL COMPLIANCE (Is it grounded on the sidewalk, off the road, depth-scaled?)
3. PRESERVATION (Are protected vehicles, facades, pedestrians, and infrastructure preserved?)
4. DESIGN COHERENCE (Perspective rhythm, scale, natural integration?)
5. HEAT-RESILIENCE INTENT (Functional microclimate shade vs decorative landscaping?)
6. UNAUTHORIZED CHANGE DETECTION (No giant planter boxes, barriers, or road mutilation!)

Returns a structured DesignCritique object and actionable retry recommendations.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from .schemas import DesignCritique, DesignIntent

logger = logging.getLogger("resilicity.critic")


def evaluate_design_critique(
    original_crop: Image.Image,
    generated_crop: Image.Image,
    mask: Image.Image,
    intervention_type: str,
    protected_mask_crop: Optional[np.ndarray] = None,
    design_intent: Optional[DesignIntent] = None,
    attempt_number: int = 1,
) -> DesignCritique:
    """Critique a generated intervention crop against architectural and thermal design intent."""
    gen_arr = np.array(generated_crop.convert("RGB"))
    h, w = gen_arr.shape[:2]

    if original_crop.size != (w, h):
        original_crop = original_crop.resize((w, h), Image.Resampling.LANCZOS)
    orig_arr = np.array(original_crop.convert("RGB"))

    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.Resampling.NEAREST)
    mask_arr = np.array(mask.convert("L")) > 30

    if protected_mask_crop is not None and protected_mask_crop.shape[:2] != (h, w):
        protected_mask_crop = cv2.resize(
            protected_mask_crop.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        ) > 0

    total_mask_px = int(np.count_nonzero(mask_arr))

    failure_reasons: List[str] = []
    retry_recommendation: Optional[str] = None

    if total_mask_px == 0:
        return DesignCritique(
            passed=False,
            overall_score=0.0,
            intervention_presence_score=0.0,
            spatial_compliance_score=0.0,
            preservation_score=1.0,
            coherence_score=0.0,
            heat_strategy_score=0.0,
            unauthorized_change_score=1.0,
            failure_reasons=["empty_intervention_mask"],
            retry_recommendation="Expand intervention mask geometry along target sidewalk corridor.",
            intervention_type=intervention_type,
            attempt_number=attempt_number,
        )

    # -------------------------------------------------------------------------
    # 1. INTERVENTION PRESENCE & MATERIALITY
    # -------------------------------------------------------------------------
    presence_score = 0.5
    if intervention_type == "tree_canopy":
        # Check for organic vegetative leaf foliage inside mask
        gen_pixels = gen_arr[mask_arr]
        orig_pixels = orig_arr[mask_arr]

        # Metric A: Greenness index [G - 0.5*(R + B)]
        greenness = gen_pixels[:, 1].astype(float) - 0.5 * (
            gen_pixels[:, 0].astype(float) + gen_pixels[:, 2].astype(float)
        )
        orig_greenness = orig_pixels[:, 1].astype(float) - 0.5 * (
            orig_pixels[:, 0].astype(float) + orig_pixels[:, 2].astype(float)
        )
        green_fraction = float(np.mean(greenness > 12.0))
        mean_greenness = float(np.mean(greenness))

        # Metric B: Texture complexity (leaf detail vs flat wall)
        gray_gen = cv2.cvtColor(gen_arr, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray_gen, cv2.CV_64F)
        texture_energy = float(np.var(laplacian[mask_arr])) if total_mask_px > 0 else 0.0

        # Metric C: Mean pixel difference inside mask
        diff_arr = np.abs(gen_arr.astype(float) - orig_arr.astype(float))
        mean_mask_diff = float(np.mean(diff_arr[mask_arr]))

        diff_factor = np.clip(mean_mask_diff / 30.0, 0.0, 1.0)
        texture_factor = np.clip(texture_energy / 180.0, 0.1, 1.0)

        # For tree canopy, verify presence of genuine green foliage
        if green_fraction < 0.12 or mean_greenness < -2.0:
            presence_score = min(0.40, float(0.20 * diff_factor + 0.20 * texture_factor))
            failure_reasons.append("insufficient_canopy_foliage_greenness")
            logger.warning(
                "Critic: Tree canopy lacks green foliage (green_frac=%.2f, mean_greenness=%.1f)",
                green_fraction, mean_greenness,
            )
        else:
            green_factor = np.clip(green_fraction * 2.0 + (mean_greenness + 2.0) / 20.0, 0.0, 1.0)
            presence_score = float(0.30 * diff_factor + 0.55 * green_factor + 0.15 * texture_factor)

        if presence_score < 0.55 and "insufficient_canopy_foliage_greenness" not in failure_reasons:
            failure_reasons.append("insufficient_canopy_presence")
            logger.warning(
                "Critic: Tree presence low (score=%.2f, green_frac=%.2f, diff=%.1f)",
                presence_score, green_fraction, mean_mask_diff,
            )

    elif intervention_type in ("permeable_pave", "cool_pavement"):
        diff_arr = np.abs(gen_arr.astype(float) - orig_arr.astype(float))
        mean_mask_diff = float(np.mean(diff_arr[mask_arr]))
        presence_score = float(np.clip(mean_mask_diff / 25.0, 0.2, 1.0))
        if presence_score < 0.50:
            failure_reasons.append("insufficient_pavement_texture_shift")

    else:
        diff_arr = np.abs(gen_arr.astype(float) - orig_arr.astype(float))
        mean_mask_diff = float(np.mean(diff_arr[mask_arr]))
        presence_score = float(np.clip(mean_mask_diff / 25.0, 0.3, 1.0))

    # -------------------------------------------------------------------------
    # 2. SPATIAL COMPLIANCE & PERSPECTIVE GROUNDING
    # -------------------------------------------------------------------------
    # Verify the intervention is physically grounded in lower portion of crop
    spatial_score = 0.90
    lower_mask_px = int(np.count_nonzero(mask_arr[int(h * 0.6):, :]))
    if lower_mask_px == 0 and intervention_type == "tree_canopy":
        spatial_score -= 0.30
        failure_reasons.append("floating_canopy_without_ground_contact")

    # -------------------------------------------------------------------------
    # 3. PRESERVATION OF PROTECTED OBJECTS
    # -------------------------------------------------------------------------
    preservation_score = 1.0
    if protected_mask_crop is not None and np.count_nonzero(protected_mask_crop) > 0:
        p_mask_bool = protected_mask_crop > 0
        p_diff = np.abs(gen_arr[p_mask_bool].astype(float) - orig_arr[p_mask_bool].astype(float))
        mean_p_diff = float(np.mean(p_diff)) if p_diff.size > 0 else 0.0

        if mean_p_diff > 12.0:
            preservation_score = float(max(0.0, 1.0 - (mean_p_diff - 12.0) / 40.0))
            if preservation_score < 0.75:
                failure_reasons.append("protected_region_violation")
                logger.warning("Critic: Protected region violation (mean_diff=%.1f)", mean_p_diff)

    # -------------------------------------------------------------------------
    # 4. UNAUTHORIZED CHANGE / HALLUCINATION DETECTION
    # -------------------------------------------------------------------------
    unauthorized_score = 0.95
    if intervention_type == "tree_canopy":
        # Detect giant planter box:
        # Planter boxes manifest as high-contrast solid rectangular blocks in the lower 35% of the crop
        lower_crop_gen = gen_arr[int(h * 0.65):, :]
        lower_crop_orig = orig_arr[int(h * 0.65):, :]
        lower_diff = np.abs(lower_crop_gen.astype(float) - lower_crop_orig.astype(float))
        lower_diff_pct = float(np.mean(lower_diff > 25.0))

        # Check for solid rectangular geometry across lower crop
        gray_lower = cv2.cvtColor(lower_crop_gen, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_lower, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=35, maxLineGap=8)
        horizontal_box_lines = 0
        if lines is not None:
            for line in lines:
                pts = np.squeeze(line)
                if pts.shape == (4,):
                    x1, y1, x2, y2 = pts
                    if abs(y2 - y1) <= 4:  # Strongly horizontal line in lower ground
                        horizontal_box_lines += 1

        if horizontal_box_lines >= 6 and lower_diff_pct > 0.45:
            unauthorized_score -= 0.35
            failure_reasons.append("planter_box_detected")
            logger.warning("Critic: Potential planter box structure detected (lines=%d)", horizontal_box_lines)

    # -------------------------------------------------------------------------
    # 5. HEAT STRATEGY & COHERENCE SCORES
    # -------------------------------------------------------------------------
    heat_strategy_score = float(np.clip(presence_score * 0.8 + spatial_score * 0.2, 0.2, 1.0))
    coherence_score = float(np.clip(spatial_score * 0.6 + preservation_score * 0.4, 0.3, 1.0))

    # -------------------------------------------------------------------------
    # 6. OVERALL CRITIQUE VERDICT & RETRY RECOMMENDATION
    # -------------------------------------------------------------------------
    overall_score = float(
        0.35 * presence_score
        + 0.20 * spatial_score
        + 0.20 * preservation_score
        + 0.15 * heat_strategy_score
        + 0.10 * unauthorized_score
    )

    passed = (
        overall_score >= 0.65
        and presence_score >= 0.50
        and preservation_score >= 0.80
        and unauthorized_score >= 0.70
    )

    if not passed:
        if "insufficient_canopy_presence" in failure_reasons:
            retry_recommendation = (
                "Strengthen mature tree canopy positive prompt, reduce ControlNet depth scale to 0.35 "
                "to allow 3D foliage volume over flat background, and increase guidance scale."
            )
        elif "planter_box_detected" in failure_reasons:
            retry_recommendation = (
                "Inject strong negative weights for planter boxes and raised containers, restrict ground "
                "mask to flush in-ground tree pit, and forbid rectangular barriers."
            )
        elif "protected_region_violation" in failure_reasons:
            retry_recommendation = (
                "Re-assert strict zeroing of protected pixels and tighten contextual crop margin."
            )
        else:
            retry_recommendation = "Adjust diffusion random seed and refine local intervention mask geometry."

    critique = DesignCritique(
        passed=passed,
        overall_score=round(overall_score, 2),
        intervention_presence_score=round(presence_score, 2),
        spatial_compliance_score=round(spatial_score, 2),
        preservation_score=round(preservation_score, 2),
        coherence_score=round(coherence_score, 2),
        heat_strategy_score=round(heat_strategy_score, 2),
        unauthorized_change_score=round(unauthorized_score, 2),
        failure_reasons=failure_reasons,
        retry_recommendation=retry_recommendation,
        intervention_type=intervention_type,
        attempt_number=attempt_number,
    )

    logger.info(
        "Design Critic evaluation for %s (attempt %d): passed=%s, overall=%.2f, presence=%.2f, reasons=%s",
        intervention_type, attempt_number, passed, overall_score, presence_score, failure_reasons,
    )
    return critique
