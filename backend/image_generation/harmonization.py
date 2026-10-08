"""Crop-Based Diffusion Harmonization Engine.

Implements Requirements 6, 7, 8, 10, 15:
- Takes the geometric draft composite containing placed assets and material drafts.
- Identifies intervention-specific bounding boxes and extracts contextual crops with 48-64px padding.
- Runs local SD 1.5 inpainting at native 512x512 with moderate denoising strength (0.45 - 0.62)
  to harmonize lighting, integrate textures, refine foliage/bark, and feather-blend edges.
- Prompt explicitly instructs: "Preserve the existing geometric composition. Refine and photorealistically
  integrate the already-placed urban design element in the masked region."
- Blends refined crop back into full-resolution street photograph with soft organic feathering.
- Ensures identity preservation and retains the geometric composite draft on any failure.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageFilter

from .sdxl_provider import LocalSDXLInpaintingProvider

logger = logging.getLogger("resilicity.harmonization")


def extract_contextual_crop(
    image: Image.Image,
    mask: Image.Image,
    padding: int = 56,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    """Extract bounding box with contextual padding around active mask pixels."""
    w, h = image.size
    mask_arr = np.array(mask.convert("L"))
    active_y, active_x = np.where(mask_arr > 20)

    if active_y.size == 0:
        # Fallback to center region
        return image, mask, (0, 0, w, h)

    min_x, max_x = int(np.min(active_x)), int(np.max(active_x))
    min_y, max_y = int(np.min(active_y)), int(np.max(active_y))

    # Add contextual padding around active object
    crop_x1 = max(0, min_x - padding)
    crop_y1 = max(0, min_y - padding)
    crop_x2 = min(w, max_x + padding)
    crop_y2 = min(h, max_y + padding)

    # Ensure crop has minimum dimensions (at least 128x128)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    if crop_w < 128:
        extra_w = (128 - crop_w) // 2
        crop_x1 = max(0, crop_x1 - extra_w)
        crop_x2 = min(w, crop_x2 + extra_w)
    if crop_h < 128:
        extra_h = (128 - crop_h) // 2
        crop_y1 = max(0, crop_y1 - extra_h)
        crop_y2 = min(h, crop_y2 + extra_h)

    bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
    cropped_img = image.crop(bbox)
    cropped_mask = mask.crop(bbox)
    return cropped_img, cropped_mask, bbox


def feather_blend_crop_back(
    base_image: Image.Image,
    harmonized_crop: Image.Image,
    crop_mask: Image.Image,
    bbox: Tuple[int, int, int, int],
    feather_radius: float = 3.0,
) -> Image.Image:
    """Seamlessly feather-blend harmonized crop back into full-resolution canvas."""
    crop_x1, crop_y1, crop_x2, crop_y2 = bbox
    target_w = crop_x2 - crop_x1
    target_h = crop_y2 - crop_y1

    # Resize harmonized crop back to exact patch size if needed
    if harmonized_crop.size != (target_w, target_h):
        harmonized_crop = harmonized_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # Soft feather alpha blend mask
    soft_mask = crop_mask.resize((target_w, target_h), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )

    base_patch = base_image.crop(bbox)
    blended_patch = Image.composite(harmonized_crop, base_patch, soft_mask)

    result_image = base_image.copy()
    result_image.paste(blended_patch, (crop_x1, crop_y1))
    return result_image


async def harmonize_intervention_crop(
    composite_image: Image.Image,
    intervention_mask: Image.Image,
    intervention_type: str,
    sd_provider: LocalSDXLInpaintingProvider,
    base_seed: int = 42,
    save_debug_crops: bool = True,
    debug_dir: Optional[Path] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Harmonize placed geometric objects via crop-based local diffusion inpainting.

    Requirements 6, 7, 8:
    - Crops around intervention with contextual padding.
    - Uses moderate strength (0.52 - 0.62) to fuse textures and illumination without altering geometry.
    - Prompts model to photorealistically refine the already placed elements.
    """
    w, h = composite_image.size
    cropped_img, cropped_mask, bbox = extract_contextual_crop(composite_image, intervention_mask, padding=56)

    # Targeted harmonization prompts (Requirement 8)
    if intervention_type == "tree_canopy":
        prompt = (
            "Preserve existing geometric composition and street architecture. "
            "Photorealistically harmonize and integrate the green street tree: "
            "natural leaf foliage, authentic tree bark texture, realistic branching, "
            "soft ambient daylight, natural cast shadows, 8k photographic quality"
        )
        neg_prompt = "cartoon, illustration, 3d render, blurry, distorted, deformed trunk, floating leaves"
        strength = 0.58
        steps = 20

    elif intervention_type == "shade_structure":
        prompt = (
            "Preserve existing geometric composition. Photorealistically integrate "
            "the modern tensile fabric pedestrian shade canopy: crisp architectural fabric, "
            "slender steel support columns, natural ambient daylight, soft realistic cast shadow"
        )
        neg_prompt = "cartoon, illustration, blurry, warped, distorted supports"
        strength = 0.54
        steps = 18

    elif intervention_type == "cool_pavement":
        prompt = (
            "Preserve lane markings, crosswalks, vehicles, and curbs. "
            "Photorealistically refine light-gray solar-reflective cool pavement asphalt, "
            "authentic road surface texture, natural daylight"
        )
        neg_prompt = "cartoon, blurry, painted lines erased, deformed road"
        strength = 0.44
        steps = 16

    elif intervention_type == "permeable_pave":
        prompt = (
            "Preserve curb lines and storefront doors. Refine interlocking modular "
            "permeable stone pavers along pedestrian walkway, crisp drainage joints, "
            "natural architectural paving texture, sharp focus"
        )
        neg_prompt = "cartoon, blurry, cracked, flat texture"
        strength = 0.46
        steps = 16

    else:
        prompt = "Photorealistically harmonize and integrate urban design infrastructure, natural lighting, sharp focus"
        neg_prompt = "cartoon, blurry, low resolution"
        strength = 0.50
        steps = 18

    # Save pre-crop debug artifact
    if save_debug_crops and debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cropped_img.save(debug_dir / f"{intervention_type}_crop_before.png")
        if intervention_type == "tree_canopy":
            cropped_img.save(debug_dir / "tree_diffusion_input.png")

    # Run inpainting on the crop
    gen_crop, err = await sd_provider.edit(
        image=cropped_img,
        prompt=prompt,
        negative_prompt=neg_prompt,
        mask_image=cropped_mask,
        seed=base_seed,
        strength=strength,
        guidance_scale=7.5,
        steps=steps,
        max_retries=1,
    )

    if gen_crop is None:
        logger.warning("Harmonization failed for %s (%s). Retaining draft composite patch.", intervention_type, err)
        return composite_image, {"status": "retained_draft", "error": err}

    # Save post-crop debug artifact
    if save_debug_crops and debug_dir:
        gen_crop.save(debug_dir / f"{intervention_type}_crop_after.png")
        if intervention_type == "tree_canopy":
            gen_crop.save(debug_dir / "tree_diffusion_output.png")
        elif intervention_type == "shade_structure":
            gen_crop.save(debug_dir / "shade_output.png")

    # Feather-blend harmonized crop back into full image canvas
    blended_full = feather_blend_crop_back(
        base_image=composite_image,
        harmonized_crop=gen_crop,
        crop_mask=cropped_mask,
        bbox=bbox,
        feather_radius=3.5,
    )

    meta = {
        "status": "harmonized",
        "bbox": bbox,
        "strength": strength,
        "steps": steps,
        "seed": base_seed,
    }
    return blended_full, meta
