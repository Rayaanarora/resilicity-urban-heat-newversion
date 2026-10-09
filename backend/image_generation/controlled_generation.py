"""Crop-Based Controlled Generative Inpainting Pipeline with Depth ControlNet.

Fulfills Requirements 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18:
- The diffusion model directly GENERATES the interventions from scratch.
- Zero pasted PNG cutouts. Zero flat vector polygons. Zero sprite stamps.
- Crop-based inpainting with generous environmental context (curb, road, facade, sky, ground).
- Depth conditioning passed to ControlNet controls 3D perspective and scale.
- Explicit planner geometry defines contextual intervention masks.
- Saves 01..07 debug artifacts per intervention pass.
- Restores protected pixels bit-perfectly.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

from .depth_util import preprocess_depth_for_controlnet
from .design_critic import evaluate_design_critique
from .schemas import DesignCritique, DesignIntent
from .sd15_controlnet_provider import LocalSD15ControlNetInpaintingProvider

logger = logging.getLogger("resilicity.controlled_gen")


def extract_contextual_crop(
    image: Image.Image,
    mask: Image.Image,
    depth_map: np.ndarray,
    padding: int = 72,
    min_size: int = 256,
) -> Tuple[Image.Image, Image.Image, np.ndarray, Tuple[int, int, int, int]]:
    """Extract bounding box with generous environmental context around active mask pixels."""
    w, h = image.size
    mask_arr = np.array(mask.convert("L"))
    active_y, active_x = np.where(mask_arr > 20)

    if active_y.size == 0:
        # Fallback to center region
        return image.copy(), mask.copy(), depth_map.copy(), (0, 0, w, h)

    min_x, max_x = int(np.min(active_x)), int(np.max(active_x))
    min_y, max_y = int(np.min(active_y)), int(np.max(active_y))

    # Add generous contextual padding (ground, curb, facade, sky)
    crop_x1 = max(0, min_x - padding)
    crop_y1 = max(0, min_y - padding)
    crop_x2 = min(w, max_x + padding)
    crop_y2 = min(h, max_y + padding)

    # Ensure crop has minimum dimensions for diffusion context
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    if crop_w < min_size:
        extra_w = (min_size - crop_w) // 2
        crop_x1 = max(0, crop_x1 - extra_w)
        crop_x2 = min(w, crop_x2 + extra_w)
    if crop_h < min_size:
        extra_h = (min_size - crop_h) // 2
        crop_y1 = max(0, crop_y1 - extra_h)
        crop_y2 = min(h, crop_y2 + extra_h)

    bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
    cropped_img = image.crop(bbox)
    cropped_mask = mask.crop(bbox)
    cropped_depth = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_img, cropped_mask, cropped_depth, bbox


def feather_blend_crop_back(
    base_image: Image.Image,
    generated_crop: Image.Image,
    crop_mask: Image.Image,
    bbox: Tuple[int, int, int, int],
    feather_radius: float = 2.5,
) -> Image.Image:
    """Seamlessly feather-blend generated crop back into the full-resolution street photograph."""
    crop_x1, crop_y1, crop_x2, crop_y2 = bbox
    target_w = crop_x2 - crop_x1
    target_h = crop_y2 - crop_y1

    if generated_crop.size != (target_w, target_h):
        generated_crop = generated_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)

    soft_mask = crop_mask.resize((target_w, target_h), Image.Resampling.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )

    base_patch = base_image.crop(bbox)
    blended_patch = Image.composite(generated_crop, base_patch, soft_mask)

    result_image = base_image.copy()
    result_image.paste(blended_patch, (crop_x1, crop_y1))
    return result_image


def build_intervention_prompt(
    intervention_type: str,
    custom_desc: Optional[str] = None,
    design_intent: Optional[DesignIntent] = None,
) -> Tuple[str, str]:
    """Synthesize photorealistic design-specific positive and negative prompts.

    Integrates DesignIntent hard negative rules and preservation boundaries.
    """
    hard_negs = ""
    if design_intent and design_intent.hard_negative_rules:
        hard_negs = ", " + ", ".join(design_intent.hard_negative_rules)

    if intervention_type == "tree_canopy":
        prompt = (
            "photorealistic lush mature roadside shade tree with dense vibrant green leafy canopy, "
            "spreading branches, realistic textured tree bark trunk rooted in sidewalk ground planting pit, "
            "dappled sunlight filtering through green leaves, casting natural cooling pedestrian shade, 8k uhd street photograph"
        )
        negative_prompt = (
            "cartoon, illustration, 3d render, CGI, metal pole, lamppost, bollard, signpost, artificial tree, "
            "cutout, sticker, bare tree, winter, autumn, dead tree, distorted, blurry, thatched roof, straw umbrella, "
            "woven basket, wicker, sculpture, statue, plastic foliage, giant egg"
            + hard_negs
        )
    elif intervention_type == "shade_structure":
        prompt = (
            "photorealistic contemporary pedestrian shade canopy integrated into this existing urban sidewalk, "
            "realistic slender structural steel supports anchored to the pavement, modern tensile fabric canopy, "
            "correct perspective, realistic architectural materials, realistic ground shadows, physically plausible "
            "construction, naturally integrated into the street scene"
        )
        negative_prompt = (
            "cartoon, illustration, 3d render, flat colored polygon, floating canopy, broken supports, "
            "impossible geometry, distorted perspective, neon colors, CGI, blurry, low resolution"
            + hard_negs
        )
    elif intervention_type == "cool_pavement":
        prompt = (
            "photorealistic high-albedo solar-reflective light-gray road pavement coating, clean modern architectural "
            "road surface, realistic fine stone texture, subtle surface roughness, preserving lane markings and curb edges, "
            "realistic street lighting and perspective, seamless urban roadway integration"
        )
        negative_prompt = (
            "cartoon, flat color paint, blue paint, plastic floor, smeared texture, blurry, altered lane markings, "
            "distorted vehicles, damaged curbs, unrealistic saturation"
            + hard_negs
        )
    elif intervention_type == "permeable_pave":
        prompt = (
            "photorealistic modular interlocking permeable concrete stone pavers on pedestrian sidewalk, "
            "warm light-gray architectural paving stones with neat drainage joints, porous stone paver texture, "
            "physically grounded, realistic perspective, seamlessly integrated into the street scene"
        )
        negative_prompt = (
            "cartoon, flat texture, indoor tiles, linoleum, blurry, distorted pavement, plastic, saturated colors"
            + hard_negs
        )
    else:
        prompt = custom_desc or (
            "photorealistic urban green infrastructure seamlessly integrated into the existing street photograph, "
            "physically plausible materials, realistic lighting and perspective"
        )
        negative_prompt = (
            "cartoon, illustration, 3d render, floating objects, distorted perspective, blurry, low resolution"
            + hard_negs
        )

    return prompt, negative_prompt


async def generate_single_tree_localized(
    base_image: Image.Image,
    anchor: Dict[str, Any],
    depth_map: np.ndarray,
    protected_mask: Optional[np.ndarray],
    sd_provider: LocalSD15ControlNetInpaintingProvider,
    design_intent: Optional[DesignIntent] = None,
    tree_idx: int = 1,
    base_seed: int = 42,
    debug_dir: Optional[Path] = None,
    max_retries: int = 2,
    save_debug: bool = True,
) -> Tuple[Image.Image, Dict[str, Any], DesignCritique]:
    """Execute localized tree-by-tree generation pass with Design Critic and bounded retry.

    Requirements 9, 10, 11, 12, 13 (Design Intelligence V2):
    - Dedicated localized crop specifically sized for this individual tree.
    - Contextual mask covering flush ground pit, trunk corridor, spreading canopy crown, and ground shadow.
    - Sidewalk surrounding the tree remains strictly preserved.
    - Zero planter box tolerance: flush pit only.
    - Moderated ControlNet depth scale (0.40) allows 3D volume synthesis over flat background.
    - Evaluated by Design Critic with intelligent bounded retry.
    """
    w, h = base_image.size
    t0 = time.time()
    out_dir = debug_dir or (Path(__file__).resolve().parent.parent / "data" / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Anchor geometry
    ax = int(round(anchor.get("x", w * 0.3)))
    ay = int(round(anchor.get("y", h * 0.75)))
    canopy_rx = int(round(anchor.get("canopy_radius", 55)))
    canopy_ry = int(round(anchor.get("canopy_height", 65)))
    canopy_cy = int(round(anchor.get("canopy_center_y", ay - int(canopy_ry * 1.5))))
    trunk_w = max(8, int(round(anchor.get("trunk_width", 14))))
    pit_dict = anchor.get("planting_pit", {})
    pit_w = max(18, int(round(pit_dict.get("width", trunk_w * 3.5))))
    pit_d = max(10, int(round(pit_dict.get("depth", trunk_w * 1.8))))

    # 2. Localized square contextual crop bounding box
    tree_total_h = ay - (canopy_cy - canopy_ry)
    crop_size = min(w, h, max(460, int(round(tree_total_h * 1.35))))
    crop_cx = ax
    crop_cy = (ay + canopy_cy) // 2
    crop_x1 = max(0, min(w - crop_size, crop_cx - crop_size // 2))
    crop_y1 = max(0, min(h - crop_size, crop_cy - crop_size // 2))
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
    cropped_img = base_image.crop(crop_bbox)
    actual_cw, actual_ch = cropped_img.size

    # 3. Localized intervention mask
    scale = 512.0 / float(crop_size)
    d_ax = int((ax - crop_x1) * scale)
    d_ay = int((ay - crop_y1) * scale)
    d_cy = int((canopy_cy - crop_y1) * scale)
    d_rx = int(canopy_rx * scale)
    d_ry = int(canopy_ry * scale)
    d_tw = max(10, int(trunk_w * scale))

    mask_512 = np.zeros((512, 512), dtype=np.uint8)
    # A. Flush in-ground porous planting pit
    cv2.ellipse(mask_512, (d_ax, d_ay), (max(20, int(d_tw * 1.8)), max(10, int(d_tw * 0.9))), 0, 0, 360, 255, -1)
    # B. Trunk corridor connecting ground pit to canopy
    cv2.rectangle(mask_512, (d_ax - d_tw // 2, d_cy), (d_ax + d_tw // 2, d_ay), 255, -1)
    # C. Natural spreading leafy canopy crown
    cv2.ellipse(mask_512, (d_ax, d_cy), (d_rx, d_ry), 0, 0, 360, 255, -1)
    # D. Cast shadow on ground
    cv2.ellipse(mask_512, (d_ax + int(d_tw * 1.5), d_ay + 10), (int(d_rx * 0.5), int(d_tw * 1.2)), 0, 0, 360, 180, -1)

    # Strictly protect scene obstacles (living pedestrians and active vehicles)
    if protected_mask is not None:
        p_crop = protected_mask[crop_y1:crop_y2, crop_x1:crop_x2]
        if np.count_nonzero(p_crop) > 0:
            p_crop_512 = cv2.resize(p_crop.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST) > 0
            mask_512[p_crop_512] = 0

    mask_512 = cv2.GaussianBlur(mask_512, (9, 9), 0)
    mask_512 = np.where(mask_512 > 30, 255, 0).astype(np.uint8)
    tree_mask_pil = Image.fromarray(mask_512, mode="L")

    # 4. Depth conditioning: proven volumetric structural sculpting
    depth_cond_arr = np.zeros((512, 512), dtype=np.uint8)
    for y in range(512):
        depth_cond_arr[y, :] = int(120 + 80 * (y / 512.0))

    y_idx, x_idx = np.indices((512, 512))
    dist_sq = ((x_idx - d_ax) / max(1.0, float(d_rx))) ** 2 + ((y_idx - d_cy) / max(1.0, float(d_ry))) ** 2
    canopy_vol = np.clip(1.0 - dist_sq, 0.0, 1.0)
    canopy_layer = (210 * (canopy_vol ** 0.5)).astype(np.uint8)
    depth_cond_arr = np.maximum(depth_cond_arr, canopy_layer)
    depth_cond_arr[d_cy:d_ay, max(0, d_ax - d_tw // 2):min(512, d_ax + d_tw // 2)] = np.maximum(
        depth_cond_arr[d_cy:d_ay, max(0, d_ax - d_tw // 2):min(512, d_ax + d_tw // 2)], 180
    )
    depth_cond_arr = cv2.GaussianBlur(depth_cond_arr, (11, 11), 0)
    depth_cond = Image.fromarray(depth_cond_arr, mode="L").convert("RGB")

    # Save debug inputs for this tree
    tree_prefix = f"tree_{tree_idx:02d}"
    tree_mask_pil.save(out_dir / f"{tree_prefix}_mask.png")
    depth_cond.save(out_dir / f"{tree_prefix}_depth.png")
    cropped_img.save(out_dir / f"{tree_prefix}_input.png")

    # 5. Diffusion Parameters & Intelligent Bounded Retry Loop
    cur_prompt, cur_neg_prompt = build_intervention_prompt("tree_canopy", design_intent=design_intent)
    cur_cnet_scale = 0.45  # Proven optimal scale for organic foliage volume and bark synthesis
    cur_guidance = 8.0
    cur_steps = 24

    best_crop = None
    best_critique = None
    best_score = -1.0

    for attempt in range(max_retries):
        cur_seed = base_seed + (attempt * 109)
        logger.info(
            "Generating %s (attempt %d/%d): seed=%d, cnet_scale=%.2f, guidance=%.1f",
            tree_prefix, attempt + 1, max_retries, cur_seed, cur_cnet_scale, cur_guidance,
        )

        gen_crop, err_msg, trace_info = await sd_provider.generate_intervention(
            image=cropped_img,
            mask=tree_mask_pil,
            depth_condition=depth_cond,
            prompt=cur_prompt,
            negative_prompt=cur_neg_prompt,
            seed=cur_seed,
            strength=1.0,
            controlnet_conditioning_scale=cur_cnet_scale,
            steps=cur_steps,
            guidance_scale=cur_guidance,
            max_retries=1,
        )

        if gen_crop is None:
            logger.error("Single tree generation failed on attempt %d: %s", attempt + 1, err_msg)
            continue

        # Evaluate generated result with Design Critic
        p_crop_arr = protected_mask[crop_y1:crop_y2, crop_x1:crop_x2] if protected_mask is not None else None
        critique = evaluate_design_critique(
            original_crop=cropped_img,
            generated_crop=gen_crop,
            mask=tree_mask_pil,
            intervention_type="tree_canopy",
            protected_mask_crop=p_crop_arr,
            design_intent=design_intent,
            attempt_number=attempt + 1,
        )

        if critique.overall_score > best_score:
            best_score = critique.overall_score
            best_crop = gen_crop
            best_critique = critique

        if critique.passed:
            logger.info("Design Critic PASSED for %s on attempt %d (score=%.2f)", tree_prefix, attempt + 1, critique.overall_score)
            break
        else:
            logger.warning("Design Critic FAILED for %s on attempt %d: %s", tree_prefix, attempt + 1, critique.failure_reasons)
            # Apply intelligent retry parameter adaptations
            if "insufficient_canopy_presence" in critique.failure_reasons:
                cur_prompt = (
                    "photorealistic mature roadside shade tree with dense vibrant green leafy canopy, "
                    "realistic natural tree bark trunk rooted in sidewalk ground pit, lush leafy foliage providing shade, "
                    "correct perspective and scale, physically grounded, realistic sunlight and cast shadow, 8k uhd"
                )
                cur_cnet_scale = 0.32  # Grant more volumetric freedom over flat background
                cur_guidance = 8.5
            if "planter_box_detected" in critique.failure_reasons:
                cur_neg_prompt += ", planter box, giant planter, raised planter, concrete container, rectangular box, raised bed"

    if best_crop is None:
        best_crop = cropped_img
        best_critique = DesignCritique(
            passed=False,
            overall_score=0.0,
            failure_reasons=["generation_failed"],
            intervention_type="tree_canopy",
            attempt_number=max_retries,
        )

    # Save debug output & critique for this tree
    best_crop.save(out_dir / f"{tree_prefix}_output.png")
    import json
    with open(out_dir / f"{tree_prefix}_critique.json", "w", encoding="utf-8") as f:
        json.dump(best_critique.dict(), f, indent=2)

    # 6. Recomposite generated crop back into base street image
    recomposited_image = feather_blend_crop_back(
        base_image=base_image,
        generated_crop=best_crop,
        crop_mask=tree_mask_pil,
        bbox=crop_bbox,
        feather_radius=2.5,
    )
    meta = {
        "tree_idx": tree_idx,
        "crop_bbox": list(crop_bbox),
        "anchor": [ax, ay],
        "scale": anchor.get("scale", 0.6),
        "critique": best_critique.dict(),
        "execution_time_s": round(time.time() - t0, 2),
        "success": best_critique.passed,
    }

    return recomposited_image, meta, best_critique


async def generate_controlled_intervention(
    base_image: Image.Image,
    intervention_mask: Image.Image,
    depth_map: np.ndarray,
    intervention_type: str,
    sd_provider: LocalSD15ControlNetInpaintingProvider,
    base_seed: int = 42,
    pass_number: int = 1,
    save_debug: bool = True,
    debug_dir: Optional[Path] = None,
    protected_mask: Optional[np.ndarray] = None,
    design_intent: Optional[DesignIntent] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Execute complete controlled generation pipeline for a single intervention.

    Steps:
    1. Extract contextual crop with padding from original street photograph.
    2. Crop corresponding depth map and intervention mask.
    3. Preprocess depth crop into ControlNet conditioning format.
    4. Save 01..05 input debug artifacts.
    5. Run SD 1.5 inpainting + Depth ControlNet.
    6. Save 06 generation output.
    7. Recomposite generated crop into original street photograph coordinates.
    8. Bit-perfectly restore protected pixels.
    9. Save 07 recomposited debug artifact.
    """
    w, h = base_image.size
    t0 = time.time()
    out_dir = debug_dir or (Path(__file__).resolve().parent.parent / "data" / "debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Contextual Crop
    cropped_img, cropped_mask, cropped_depth, crop_bbox = extract_contextual_crop(
        image=base_image,
        mask=intervention_mask,
        depth_map=depth_map,
        padding=72,
        min_size=256,
    )

    crop_w = crop_bbox[2] - crop_bbox[0]
    crop_h = crop_bbox[3] - crop_bbox[1]

    # 2. Preprocess Depth Conditioning
    depth_cond = preprocess_depth_for_controlnet(
        cropped_depth,
        target_size=(512, 512),
        invert_if_depth=True,
    )

    # 3. Build Prompts
    prompt, neg_prompt = build_intervention_prompt(intervention_type, design_intent=design_intent)

    # 4. Save 01..05 Debug Artifacts for this intervention pass
    prefix = f"pass_{pass_number:02d}_{intervention_type}"
    if save_debug:
        # 01: original crop
        cropped_img.resize((512, 512), Image.Resampling.LANCZOS).save(out_dir / f"{prefix}_01_original_crop.png")
        if pass_number == 1 or intervention_type == "tree_canopy":
            cropped_img.resize((512, 512), Image.Resampling.LANCZOS).save(out_dir / "01_original_crop.png")
            cropped_img.save(out_dir / "09_tree_crop_input.png")

        # 02: depth condition
        depth_cond.save(out_dir / f"{prefix}_02_depth_condition.png")
        if pass_number == 1 or intervention_type == "tree_canopy":
            depth_cond.save(out_dir / "02_depth_condition.png")

        # 03: intervention mask
        cropped_mask.resize((512, 512), Image.Resampling.NEAREST).save(out_dir / f"{prefix}_03_intervention_mask.png")
        if pass_number == 1 or intervention_type == "tree_canopy":
            cropped_mask.resize((512, 512), Image.Resampling.NEAREST).save(out_dir / "03_intervention_mask.png")
            cropped_mask.save(out_dir / "10_tree_crop_mask.png")

        # 04: control input
        depth_cond.save(out_dir / f"{prefix}_04_control_input.png")
        if pass_number == 1 or intervention_type == "tree_canopy":
            depth_cond.save(out_dir / "04_control_input.png")

        # 05: generation input (visual overlay of mask on crop)
        vis_input = cropped_img.resize((512, 512), Image.Resampling.LANCZOS).convert("RGBA")
        mask_vis = cropped_mask.resize((512, 512), Image.Resampling.NEAREST)
        tint = Image.new("RGBA", (512, 512), (0, 220, 100, 90))
        vis_input = Image.composite(tint, vis_input, mask_vis)
        vis_input.convert("RGB").save(out_dir / f"{prefix}_05_generation_input.png")
        if pass_number == 1 or intervention_type == "tree_canopy":
            vis_input.convert("RGB").save(out_dir / "05_generation_input.png")

    # 5. Run Generative Inpainting with Depth ControlNet
    # Trees and shade structures are generated freshly inside mask (strength 1.0)
    # Pavements use high strength (0.85 - 0.95) to create realistic texture variation
    cnet_scale = 0.82 if intervention_type in ("tree_canopy", "shade_structure") else 0.70
    strength = 1.0 if intervention_type in ("tree_canopy", "shade_structure") else 0.88
    steps = 24 if intervention_type == "tree_canopy" else 20

    logger.info(
        "Launching ControlNet generation pass %d (%s): bbox=%s seed=%d strength=%.2f cnet_scale=%.2f",
        pass_number, intervention_type, crop_bbox, base_seed, strength, cnet_scale,
    )

    generated_crop_512, gen_err, gen_trace = await sd_provider.generate_intervention(
        image=cropped_img,
        mask=cropped_mask,
        depth_condition=depth_cond,
        prompt=prompt,
        negative_prompt=neg_prompt,
        seed=base_seed,
        strength=strength,
        controlnet_conditioning_scale=cnet_scale,
        steps=steps,
        guidance_scale=7.5,
        max_retries=2,
    )

    if generated_crop_512 is None:
        logger.error("ControlNet generation failed on pass %d: %s", pass_number, gen_err)
        return base_image, {
            "error": gen_err,
            "crop_bbox": list(crop_bbox),
            "pass": pass_number,
            "intervention": intervention_type,
            "success": False,
        }

    # 6. Save 06 Generation Output
    if save_debug:
        generated_crop_512.save(out_dir / f"{prefix}_06_generation_output.png")
        if pass_number == 1 or intervention_type == "tree_canopy":
            generated_crop_512.save(out_dir / "06_generation_output.png")
            generated_crop_512.save(out_dir / "11_tree_crop_output.png")
        elif intervention_type == "shade_structure":
            generated_crop_512.save(out_dir / "15_shade_crop_output.png")

    # 7. Recomposite Generated Crop into Full Street Image
    recomposited_image = feather_blend_crop_back(
        base_image=base_image,
        generated_crop=generated_crop_512,
        crop_mask=cropped_mask,
        bbox=crop_bbox,
        feather_radius=2.5,
    )

    # 8. Restore Protected Pixels Bit-Perfect if inside crop bbox
    if protected_mask is not None:
        p_crop = protected_mask[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
        if np.count_nonzero(p_crop) > 0:
            orig_arr = np.array(base_image.convert("RGB"))
            rec_arr = np.array(recomposited_image.convert("RGB"))
            rec_arr[protected_mask] = orig_arr[protected_mask]
            recomposited_image = Image.fromarray(rec_arr)
            logger.info("Restored protected pixels in recomposited pass %d", pass_number)

    # 9. Save 07 Recomposited
    if save_debug:
        recomposited_image.save(out_dir / f"{prefix}_07_recomposited.png")
        if pass_number == 1 or intervention_type == "tree_canopy":
            recomposited_image.save(out_dir / "07_recomposited.png")

    meta = {
        "pass": pass_number,
        "intervention": intervention_type,
        "crop_bbox": list(crop_bbox),
        "seed": gen_trace.get("seed", base_seed),
        "steps": steps,
        "strength": strength,
        "controlnet_scale": cnet_scale,
        "masked_diff": gen_trace.get("best_masked_diff", 0.0),
        "peak_vram_gb": gen_trace.get("peak_vram_gb", 0.0),
        "execution_time_s": gen_trace.get("execution_time_s", round(time.time() - t0, 2)),
        "success": True,
        "error": None,
    }

    return recomposited_image, meta
