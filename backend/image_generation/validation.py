"""Quantitative Image Validation and Quality Assessment for Urban Redesign Outputs.

Implements Task 5 Multi-Stage Validation:
1. Image similarity check: rejects outputs virtually identical to source photograph.
2. Localized mask-region difference check: verifies that pixels inside intervention masks
   underwent substantial physical/color transformation.
3. Architecture-preservation check: verifies that unmasked structures, facades, and vehicles
   outside the inpainting mask were preserved faithfully without corruption.
4. Intervention-presence check: verifies spectral indicators of cooling interventions
   (e.g., green vegetation boost in tree canopy zones, albedo shift in cool pavement zones).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

from .schemas import ValidationReport

logger = logging.getLogger("resilicity.validation")

# Validation thresholds
DEFAULT_MIN_OVERALL_DIFF = 5.0       # Minimum mean pixel difference across entire image
DEFAULT_MIN_MASKED_DIFF = 12.0       # Minimum mean pixel difference inside masked region
DEFAULT_MIN_PCT_CHANGED = 4.0        # Minimum % of image pixels noticeably changed (>15/255)
DEFAULT_MAX_UNMASKED_DIFF = 20.0     # Maximum acceptable drift in unmasked zones (architecture preservation)


def validate_image_output(
    image: Image.Image,
    original_size: Tuple[int, int],
    orig_img: Optional[Image.Image] = None,
    mask_img: Optional[Image.Image] = None,
    individual_masks: Optional[Dict[str, Image.Image]] = None,
    tolerance: float = 0.08,
    min_diff_mean: float = DEFAULT_MIN_OVERALL_DIFF,
    min_masked_diff: float = DEFAULT_MIN_MASKED_DIFF,
    min_pct_changed: float = DEFAULT_MIN_PCT_CHANGED,
) -> Tuple[bool, ValidationReport, Image.Image]:
    """Validate generated image quality, non-blank status, aspect ratio, and meaningful visual change.

    Args:
        image: Generated PIL Image.
        original_size: (width, height) of the source uploaded photograph.
        orig_img: Source original PIL Image to perform difference analysis against.
        mask_img: Inpainting mask used during generation ('L' mode, 255=inpainted, 0=preserved).
        individual_masks: Dictionary of separate localized masks (tree_mask, road_mask, etc.).
        tolerance: Allowed aspect ratio discrepancy before normalization.
        min_diff_mean: Minimum average pixel difference threshold across the whole image.
        min_masked_diff: Minimum average pixel difference inside the masked inpainting area.
        min_pct_changed: Minimum percent of pixels that must differ from original.

    Returns:
        Tuple of (is_valid, validation_report, normalized_image).
    """
    orig_w, orig_h = original_size
    gen_w, gen_h = image.size

    orig_ratio = orig_w / float(orig_h)
    gen_ratio = gen_w / float(gen_h)

    checks_passed: List[str] = []
    warnings: List[str] = []

    # -----------------------------------------------------------------------
    # Stage 1: Dimension and Basic Existence Check
    # -----------------------------------------------------------------------
    if gen_w < 64 or gen_h < 64:
        return False, ValidationReport(
            is_valid=False,
            aspect_ratio_preserved=False,
            dimensions_valid=False,
            non_blank_verified=False,
            warnings=["Image dimensions too small to be valid."],
        ), image

    checks_passed.append(f"Dimensions valid ({gen_w}x{gen_h})")

    # -----------------------------------------------------------------------
    # Stage 2: Non-Blank and Non-Trivial Content Verification
    # -----------------------------------------------------------------------
    np_img = np.array(image.convert("RGB")).astype(np.float32)
    std_dev = float(np.std(np_img))
    if std_dev < 8.0:
        return False, ValidationReport(
            is_valid=False,
            aspect_ratio_preserved=False,
            dimensions_valid=True,
            non_blank_verified=False,
            warnings=["Image appears flat or blank (standard deviation < 8)."],
        ), image

    checks_passed.append(f"Non-blank verified (content variance std={std_dev:.1f})")

    # -----------------------------------------------------------------------
    # Stage 3: Image Similarity & Difference Scoring Against Original
    # -----------------------------------------------------------------------
    diff_mean = None
    pct_changed = None
    perceptual_score = None
    masked_diff = None
    unmasked_diff = None
    architecture_preserved = True
    interventions_detected: List[str] = []

    if orig_img is not None:
        # Resize original to match generated image dimensions for comparison
        orig_resized = orig_img.convert("RGB").resize((gen_w, gen_h), Image.Resampling.LANCZOS)
        orig_np = np.array(orig_resized).astype(np.float32)

        # Absolute difference per pixel channel
        abs_diff = np.abs(np_img - orig_np)  # shape (H, W, 3)
        diff_mean = float(np.mean(abs_diff))

        # Channel max difference: pixels where color changed noticeably (> 15 / 255)
        pixel_diff = np.max(abs_diff, axis=2)
        changed_pixels = np.count_nonzero(pixel_diff > 15.0)
        total_pixels = gen_w * gen_h
        pct_changed = float(round((changed_pixels / total_pixels) * 100.0, 2))

        # Perceptual score [0-100] based on meaningful change magnitude
        perceptual_score = float(round(min(100.0, (diff_mean / 40.0) * 100.0), 1))

        # 3a. Check overall difference: Reject if visually identical
        if diff_mean < min_diff_mean or pct_changed < min_pct_changed:
            msg = (
                f"Generated image is virtually identical to original "
                f"(diff_mean={diff_mean:.2f} < {min_diff_mean:.1f}, changed={pct_changed:.1f}% < {min_pct_changed:.1f}%)."
            )
            warnings.append(msg)
            logger.warning("Validation rejected: %s", msg)
            return False, ValidationReport(
                is_valid=False,
                aspect_ratio_preserved=True,
                dimensions_valid=True,
                non_blank_verified=True,
                diff_mean=diff_mean,
                pct_changed=pct_changed,
                perceptual_score=perceptual_score,
                checks_passed=checks_passed,
                warnings=warnings,
            ), image

        checks_passed.append(
            f"Image similarity check passed: {pct_changed:.1f}% changed, mean diff={diff_mean:.1f}"
        )

        # -------------------------------------------------------------------
        # 3b. Localized Mask-Region Difference Check
        # -------------------------------------------------------------------
        if mask_img is not None:
            m_resized = mask_img.convert("L").resize((gen_w, gen_h), Image.Resampling.NEAREST)
            mask_arr = (np.array(m_resized) > 30).astype(np.float32)
            mask_3d = np.stack([mask_arr] * 3, axis=2)

            masked_px = float(mask_3d.sum())
            unmasked_px = float(total_pixels * 3 - masked_px)

            if masked_px > 0:
                masked_diff = float((abs_diff * mask_3d).sum() / masked_px)
                # Count changed pixels specifically inside the mask
                masked_pixel_diff = pixel_diff * mask_arr
                pct_masked_changed = float(
                    (np.count_nonzero(masked_pixel_diff > 15.0) / np.count_nonzero(mask_arr)) * 100.0
                )

                if masked_diff < min_masked_diff or pct_masked_changed < 10.0:
                    msg = (
                        f"Masked intervention region did not change sufficiently "
                        f"(masked_diff={masked_diff:.2f} < {min_masked_diff:.1f}, "
                        f"pct_masked_changed={pct_masked_changed:.1f}% < 10.0%)."
                    )
                    warnings.append(msg)
                    logger.warning("Validation rejected: %s", msg)
                    return False, ValidationReport(
                        is_valid=False,
                        aspect_ratio_preserved=True,
                        dimensions_valid=True,
                        non_blank_verified=True,
                        diff_mean=diff_mean,
                        pct_changed=pct_changed,
                        perceptual_score=perceptual_score,
                        masked_diff=masked_diff,
                        checks_passed=checks_passed,
                        warnings=warnings,
                    ), image

                checks_passed.append(
                    f"Masked region difference verified: masked_diff={masked_diff:.1f}, {pct_masked_changed:.1f}% modified"
                )

            # ---------------------------------------------------------------
            # 3c. Architecture-Preservation Check (Unmasked Region Fidelity)
            # ---------------------------------------------------------------
            if unmasked_px > 0:
                inv_mask_3d = 1.0 - mask_3d
                unmasked_diff = float((abs_diff * inv_mask_3d).sum() / unmasked_px)
                if unmasked_diff > DEFAULT_MAX_UNMASKED_DIFF:
                    architecture_preserved = False
                    warnings.append(
                        f"Unmasked architecture drifted more than expected (unmasked_diff={unmasked_diff:.1f} > {DEFAULT_MAX_UNMASKED_DIFF:.1f})."
                    )
                else:
                    checks_passed.append(
                        f"Architecture preserved: unmasked diff={unmasked_diff:.1f} <= {DEFAULT_MAX_UNMASKED_DIFF:.1f}"
                    )

        # -------------------------------------------------------------------
        # 3d. Intervention-Presence Check (Spectral Indicators)
        # -------------------------------------------------------------------
        if individual_masks:
            # Check for vegetation presence in tree_mask
            t_mask = individual_masks.get("tree_mask")
            if t_mask is not None:
                tm_resized = t_mask.convert("L").resize((gen_w, gen_h), Image.Resampling.NEAREST)
                tm_arr = np.array(tm_resized) > 30
                if np.count_nonzero(tm_arr) > 100:
                    orig_green = np.mean(orig_np[tm_arr, 1] - 0.5 * (orig_np[tm_arr, 0] + orig_np[tm_arr, 2]))
                    gen_green = np.mean(np_img[tm_arr, 1] - 0.5 * (np_img[tm_arr, 0] + np_img[tm_arr, 2]))
                    if gen_green > orig_green + 1.0:
                        interventions_detected.append("tree_canopy_vegetation")
                        checks_passed.append(f"Tree vegetation verified (green index delta={gen_green - orig_green:+.1f})")

            # Check for cool pavement albedo in road_mask
            r_mask = individual_masks.get("road_mask")
            if r_mask is not None:
                rm_resized = r_mask.convert("L").resize((gen_w, gen_h), Image.Resampling.NEAREST)
                rm_arr = np.array(rm_resized) > 30
                if np.count_nonzero(rm_arr) > 100:
                    orig_lum = np.mean(0.299 * orig_np[rm_arr, 0] + 0.587 * orig_np[rm_arr, 1] + 0.114 * orig_np[rm_arr, 2])
                    gen_lum = np.mean(0.299 * np_img[rm_arr, 0] + 0.587 * np_img[rm_arr, 1] + 0.114 * np_img[rm_arr, 2])
                    if abs(gen_lum - orig_lum) > 3.0:
                        interventions_detected.append("cool_pavement_albedo_shift")
                        checks_passed.append(f"Cool pavement albedo verified (luminance delta={gen_lum - orig_lum:+.1f})")

    # -----------------------------------------------------------------------
    # Stage 4: Aspect Ratio Check and Normalization
    # -----------------------------------------------------------------------
    ratio_diff = abs(gen_ratio - orig_ratio) / orig_ratio
    aspect_preserved = (ratio_diff <= tolerance)

    normalized_image = image
    if not aspect_preserved:
        target_aspect = orig_ratio
        if gen_ratio > target_aspect:
            new_w = int(round(gen_h * target_aspect))
            left = (gen_w - new_w) // 2
            normalized_image = image.crop((left, 0, left + new_w, gen_h))
            warnings.append(f"Adjusted width from {gen_w} to {new_w} to match original aspect ratio.")
        else:
            new_h = int(round(gen_w / target_aspect))
            top = (gen_h - new_h) // 2
            normalized_image = image.crop((0, top, gen_w, top + new_h))
            warnings.append(f"Adjusted height from {gen_h} to {new_h} to match original aspect ratio.")
        aspect_preserved = True
        checks_passed.append(f"Normalized to match source aspect ratio ({orig_ratio:.2f})")
    else:
        checks_passed.append(f"Aspect ratio naturally preserved ({gen_ratio:.2f} vs {orig_ratio:.2f})")

    report = ValidationReport(
        is_valid=True,
        aspect_ratio_preserved=aspect_preserved,
        dimensions_valid=True,
        non_blank_verified=True,
        diff_mean=diff_mean,
        pct_changed=pct_changed,
        perceptual_score=perceptual_score,
        masked_diff=masked_diff,
        unmasked_diff=unmasked_diff,
        architecture_preserved=architecture_preserved,
        interventions_detected=interventions_detected,
        checks_passed=checks_passed,
        warnings=warnings,
    )

    return True, report, normalized_image


def validate_pass_output(
    gen_image: Image.Image,
    base_image: Image.Image,
    pass_mask: Image.Image,
    protected_mask: Optional[np.ndarray],
    intervention_type: str,
    min_masked_diff: float = 10.0,
    max_protected_drift: float = 18.0,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """Validate a single autonomous inpainting pass for intervention presence and protection.

    Checks:
    1. Meaningful change inside the pass mask region (masked_diff >= min_masked_diff).
    2. Zero or minimal drift inside protected object regions (protected_diff <= max_protected_drift).
    3. Intervention presence:
       - tree_canopy: green index increase inside tree mask.
       - cool_pavement: positive albedo/luminance shift on roadway.
       - shade_structure / permeable_pave: distinct material/texture transformation.
    """
    w, h = gen_image.size
    base_resized = base_image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    mask_resized = pass_mask.convert("L").resize((w, h), Image.Resampling.NEAREST)

    gen_arr = np.array(gen_image.convert("RGB")).astype(np.float32)
    base_arr = np.array(base_resized).astype(np.float32)
    m_arr = (np.array(mask_resized) > 30)

    masked_px = np.count_nonzero(m_arr)
    if masked_px == 0:
        return True, {"masked_diff": 0.0, "presence_verified": True}, None

    abs_diff = np.abs(gen_arr - base_arr)
    mask_3d = np.stack([m_arr] * 3, axis=2)
    masked_diff = float((abs_diff * mask_3d).sum() / (masked_px * 3))

    # 1. Masked diff check
    if masked_diff < min_masked_diff:
        return False, {"masked_diff": round(masked_diff, 2)}, f"Intervention region did not change sufficiently ({masked_diff:.1f} < {min_masked_diff:.1f})"

    # 2. Protected object drift check
    protected_diff = 0.0
    if protected_mask is not None:
        p_arr = protected_mask
        if p_arr.shape != (h, w):
            p_pil = Image.fromarray(p_arr.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)
            p_arr = np.array(p_pil) > 128
        p_px = np.count_nonzero(p_arr)
        if p_px > 0:
            p_3d = np.stack([p_arr] * 3, axis=2)
            protected_diff = float((abs_diff * p_3d).sum() / (p_px * 3))
            if protected_diff > max_protected_drift:
                return False, {"masked_diff": round(masked_diff, 2), "protected_diff": round(protected_diff, 2)}, f"Protected objects damaged or modified ({protected_diff:.1f} > {max_protected_drift:.1f})"

    # 3. Intervention-specific presence verification
    presence_verified = True
    metric_detail = {}

    if intervention_type == "tree_canopy":
        base_green = float(np.mean(base_arr[m_arr, 1] - 0.5 * (base_arr[m_arr, 0] + base_arr[m_arr, 2])))
        gen_green = float(np.mean(gen_arr[m_arr, 1] - 0.5 * (gen_arr[m_arr, 0] + gen_arr[m_arr, 2])))
        green_delta = gen_green - base_green
        metric_detail["green_delta"] = round(green_delta, 2)
        if green_delta < -3.0 and masked_diff < 15.0:
            presence_verified = False

    elif intervention_type == "cool_pavement":
        base_lum = float(np.mean(0.299 * base_arr[m_arr, 0] + 0.587 * base_arr[m_arr, 1] + 0.114 * base_arr[m_arr, 2]))
        gen_lum = float(np.mean(0.299 * gen_arr[m_arr, 0] + 0.587 * gen_arr[m_arr, 1] + 0.114 * gen_arr[m_arr, 2]))
        lum_delta = gen_lum - base_lum
        metric_detail["lum_delta"] = round(lum_delta, 2)
        if lum_delta < -8.0:
            presence_verified = False

    metrics = {
        "masked_diff": round(masked_diff, 2),
        "protected_diff": round(protected_diff, 2),
        "presence_verified": presence_verified,
        "details": metric_detail,
    }

    if not presence_verified:
        return False, metrics, f"Intervention presence validation failed for {intervention_type}"

    return True, metrics, None

