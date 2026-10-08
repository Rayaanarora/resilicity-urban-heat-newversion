"""Autonomous Multi-Pass SDXL Inpainting Pipeline.

Implements Parts L, M, N, O, P:
- Autonomous sequential inpainting passes:
    PASS 1: tree_canopy and green infrastructure
    PASS 2: shade_structure
    PASS 3: permeable_pave + cool_pavement (material transformations)
    PASS 4: cool_roof / green_roof (if applicable)
- Sequential image passing: accepted output of Pass k becomes input to Pass k+1.
- Cumulative preservation mask: accepted newly generated regions are protected from subsequent passes.
- Per-pass dedicated prompts and tailored strengths (structural: 0.95-1.0, material: 0.70-0.85).
- Per-pass validation with automatic retry up to 3 attempts. Deterministic seeds.
- Reverts to previous accepted image on pass failure, preventing corruption.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from .mask_builder import build_pass_mask, build_protected_object_mask
from .prompts import build_pass_sdxl_prompt
from .schemas import SpatialDesignPlan, ValidationReport
from .sdxl_provider import LocalSDXLInpaintingProvider
from .validation import validate_image_output, validate_pass_output

logger = logging.getLogger("resilicity.multi_pass")


async def run_autonomous_multi_pass_redesign(
    image: Image.Image,
    plan: SpatialDesignPlan,
    seg_result: Dict[str, Any],
    sdxl: LocalSDXLInpaintingProvider,
    scene_understanding: Optional[Any] = None,
    quality_tier: str = "fast",
    save_debug: bool = True,
    debug_dir: Optional[Path] = None,
    base_seed: int = 42,
) -> Tuple[Optional[Image.Image], Optional[ValidationReport], Dict[str, Any]]:
    """Execute autonomous sequential multi-pass SDXL inpainting.

    Returns:
        (final_image, validation_report, multi_pass_metadata)
    """
    w, h = image.size
    total_px = w * h
    out_dir = debug_dir or (Path(__file__).resolve().parent.parent / "data" / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Group interventions into sequential passes (Part L)
    # Ordering: Trees -> Shade -> Road/Pavement -> Roof
    pass_ordering = ["tree_canopy", "shade_structure", "permeable_pave", "cool_pavement", "green_roof", "cool_roof"]
    ordered_interventions = []

    for expected_type in pass_ordering:
        for iv in plan.interventions:
            if iv.type == expected_type and iv not in ordered_interventions:
                ordered_interventions.append(iv)

    if not ordered_interventions:
        logger.warning("No interventions in plan, adding default tree canopy")
        ordered_interventions = plan.interventions[:1]

    logger.info("=" * 65)
    logger.info("STARTING AUTONOMOUS MULTI-PASS SDXL INPAINTING")
    logger.info("Planned sequential passes: %s", [f"{i.type} ({getattr(i, 'target_zone', 'zone')})" for i in ordered_interventions])
    logger.info("=" * 65)

    current_image = image.copy()
    cumulative_accepted_mask = np.zeros((h, w), dtype=bool)
    protected_mask = build_protected_object_mask(w, h, seg_result)

    pass_results: List[Dict[str, Any]] = []
    max_attempts_per_pass = 3

    for pass_idx, intervention in enumerate(ordered_interventions, start=1):
        itype = intervention.type
        tzone = getattr(intervention, "target_zone", None)
        title = getattr(intervention, "title", itype)
        is_structural = itype in ("tree_canopy", "shade_structure")

        # Per-pass parameters (Part N)
        base_strength = 0.999 if is_structural else 0.78
        base_steps = 12 if quality_tier == "fast" else 18
        base_guidance = 7.5

        logger.info("-" * 55)
        logger.info("PASS %d/%d: %s on %s (structural=%s)", pass_idx, len(ordered_interventions), title, tzone or "surface", is_structural)

        pass_accepted = False
        accepted_pass_img = None
        accepted_pass_mask = None
        pass_metrics = {}

        for attempt in range(max_attempts_per_pass):
            # Deterministic variation per attempt
            current_seed = base_seed + pass_idx * 100 + attempt * 17
            attempt_strength = min(1.0, base_strength + (0.001 if is_structural else attempt * 0.04))
            attempt_guidance = base_guidance + (attempt * 1.0)
            attempt_steps = base_steps + (attempt * 2)

            # Generate intervention-specific localized mask, protecting previous accepted regions
            pass_mask_img, mask_meta = build_pass_mask(
                intervention_type=itype,
                image=image,
                seg_result=seg_result,
                scene_understanding=scene_understanding,
                target_zone=tzone,
                expansion_level=attempt,
                accepted_regions_mask=cumulative_accepted_mask,
            )

            mask_px = mask_meta["covered_pixels"]
            if mask_px < 100:
                logger.info("Pass %d [%s] mask has negligible area (%d px). Skipping pass.", pass_idx, itype, mask_px)
                pass_accepted = True
                break

            prompt = build_pass_sdxl_prompt(itype, attempt=attempt)

            logger.info(
                "Pass %d Attempt %d/%d: seed=%d strength=%.3f steps=%d mask_px=%d (%.1f%%)",
                pass_idx, attempt + 1, max_attempts_per_pass, current_seed, attempt_strength, attempt_steps,
                mask_px, mask_meta["coverage_percentage"],
            )
            logger.info("Prompt: %s", prompt)

            # Execute SDXL inpainting on current accepted image
            gen_img, gen_err = await sdxl.edit(
                current_image,
                prompt=prompt,
                quality_tier=quality_tier,
                mask_image=pass_mask_img,
                seed=current_seed,
                strength=attempt_strength,
                guidance_scale=attempt_guidance,
                steps=attempt_steps,
                max_retries=1,
            )

            if gen_img is None:
                logger.warning("Pass %d attempt %d generation failed: %s", pass_idx, attempt + 1, gen_err)
                continue

            # Per-pass validation (Part O)
            is_valid, metrics, warn_msg = validate_pass_output(
                gen_image=gen_img,
                base_image=current_image,
                pass_mask=pass_mask_img,
                protected_mask=protected_mask,
                intervention_type=itype,
                min_masked_diff=10.0 if is_structural else 7.0,
            )

            if is_valid:
                logger.info("PASS %d ACCEPTED on attempt %d: masked_diff=%.1f", pass_idx, attempt + 1, metrics.get("masked_diff", 0))
                pass_accepted = True
                accepted_pass_img = gen_img
                accepted_pass_mask = np.array(pass_mask_img) > 30
                pass_metrics = metrics
                break
            else:
                logger.warning("Pass %d attempt %d rejected: %s (escalating prompt & mask)", pass_idx, attempt + 1, warn_msg)

        if pass_accepted and accepted_pass_img is not None:
            # Update current image and lock in cumulative mask to protect these pixels
            current_image = accepted_pass_img
            if accepted_pass_mask is not None:
                cumulative_accepted_mask |= accepted_pass_mask

            # Save per-pass debug image
            if save_debug:
                pass_fname = f"pass_{pass_idx:02d}_{itype}.png"
                current_image.save(out_dir / pass_fname)
                pass_mask_img.save(out_dir / f"mask_{pass_idx:02d}_{itype}.png")
                logger.info("Saved pass artifact to %s", pass_fname)

            pass_results.append({
                "pass_index": pass_idx,
                "type": itype,
                "status": "accepted",
                "metrics": pass_metrics,
            })
        else:
            logger.warning("Pass %d [%s] failed all %d attempts. Reverting to previous image state.", pass_idx, itype, max_attempts_per_pass)
            pass_results.append({
                "pass_index": pass_idx,
                "type": itype,
                "status": "reverted",
            })

    # Overall final validation comparing final image against original photograph
    cum_mask_pil = Image.fromarray((cumulative_accepted_mask * 255).astype(np.uint8), mode="L")
    is_valid, final_report, normalized_final = validate_image_output(
        current_image,
        image.size,
        orig_img=image,
        mask_img=cum_mask_pil,
        min_diff_mean=5.0,
        min_masked_diff=10.0,
        min_pct_changed=3.5,
    )

    metadata = {
        "passes_executed": len(pass_results),
        "passes_accepted": sum(1 for p in pass_results if p["status"] == "accepted"),
        "pass_breakdown": pass_results,
        "cumulative_mask_coverage": round(float(np.count_nonzero(cumulative_accepted_mask) / total_px * 100.0), 2),
    }

    if save_debug:
        current_image.save(out_dir / "final_redesign.png")
        cum_mask_pil.save(out_dir / "debug_cumulative_accepted_mask.png")

    logger.info("=" * 65)
    logger.info("MULTI-PASS REDESIGN COMPLETED: %d/%d passes accepted, final diff_mean=%.1f", metadata["passes_accepted"], len(pass_results), final_report.diff_mean or 0)
    logger.info("=" * 65)

    return normalized_final, final_report, metadata
