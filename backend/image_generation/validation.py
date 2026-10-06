"""Image validation, difference scoring, and aspect ratio preservation for Generative AI outputs."""

from typing import Optional, Tuple
import numpy as np
from PIL import Image

from .schemas import ValidationReport


def validate_image_output(
    image: Image.Image,
    original_size: Tuple[int, int],
    orig_img: Optional[Image.Image] = None,
    tolerance: float = 0.08,
    min_diff_mean: float = 4.0,
    min_pct_changed: float = 1.5,
) -> Tuple[bool, ValidationReport, Image.Image]:
    """Validate generated image quality, non-blank status, aspect ratio, and meaningful visual change.

    Args:
        image: Generated PIL Image.
        original_size: (width, height) of the source uploaded photograph.
        orig_img: Optional source original PIL Image to perform difference analysis against.
        tolerance: Allowed aspect ratio discrepancy before normalization.
        min_diff_mean: Minimum average pixel difference threshold to prevent unchanged image passes.
        min_pct_changed: Minimum percent of pixels that must differ from original.

    Returns:
        Tuple of (is_valid, validation_report, normalized_image).
    """
    orig_w, orig_h = original_size
    gen_w, gen_h = image.size

    orig_ratio = orig_w / float(orig_h)
    gen_ratio = gen_w / float(gen_h)

    checks_passed = []
    warnings = []

    # 1. Dimension and existence check
    if gen_w < 64 or gen_h < 64:
        return False, ValidationReport(
            is_valid=False,
            aspect_ratio_preserved=False,
            dimensions_valid=False,
            non_blank_verified=False,
            warnings=["Image dimensions too small to be valid."],
        ), image

    checks_passed.append(f"Dimensions valid ({gen_w}x{gen_h})")

    # 2. Non-blank and non-trivial content verification
    np_img = np.array(image.convert("RGB")).astype(np.float32)
    std_dev = float(np.std(np_img))
    if std_dev < 8.0:
        return False, ValidationReport(
            is_valid=False,
            aspect_ratio_preserved=False,
            dimensions_valid=True,
            non_blank_verified=False,
            warnings=["Image appears to be flat or blank (standard deviation < 8)."],
        ), image

    checks_passed.append(f"Non-blank verified (content variance std={std_dev:.1f})")

    # 3. Difference and Change Detection against Original Photograph
    diff_mean = None
    pct_changed = None
    perceptual_score = None

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

        # Verification: Reject image if it is effectively identical to the original
        if diff_mean < min_diff_mean or pct_changed < min_pct_changed:
            warnings.append(
                f"Generated image is virtually identical to original (diff_mean={diff_mean:.2f}, changed={pct_changed:.1f}%)."
            )
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
            f"Visibly redesigned: {pct_changed:.1f}% area transformed (mean diff={diff_mean:.1f})"
        )

    # 4. Aspect ratio check and normalization
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
        checks_passed=checks_passed,
        warnings=warnings,
    )

    return True, report, normalized_image
