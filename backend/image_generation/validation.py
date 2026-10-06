"""Image validation and aspect ratio preservation for Generative AI outputs."""

import io
import math
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from .schemas import ValidationReport


def validate_image_output(
    image: Image.Image,
    original_size: Tuple[int, int],
    tolerance: float = 0.08,
) -> Tuple[bool, ValidationReport, Image.Image]:
    """Validate generated image quality, non-blank status, and aspect ratio.
    
    Args:
        image: Generated PIL Image.
        original_size: (width, height) of the source uploaded photograph.
        tolerance: Allowed aspect ratio discrepancy before normalization.
        
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
    np_img = np.array(image.convert("RGB"))
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

    # 3. Aspect ratio check and normalization
    ratio_diff = abs(gen_ratio - orig_ratio) / orig_ratio
    aspect_preserved = (ratio_diff <= tolerance)

    normalized_image = image
    if not aspect_preserved:
        # Normalize without distortion by center cropping to match original aspect ratio exactly
        target_aspect = orig_ratio
        if gen_ratio > target_aspect:
            # Generated image is wider than original: crop sides
            new_w = int(round(gen_h * target_aspect))
            left = (gen_w - new_w) // 2
            normalized_image = image.crop((left, 0, left + new_w, gen_h))
            warnings.append(f"Adjusted width from {gen_w} to {new_w} to match original aspect ratio.")
        else:
            # Generated image is taller than original: crop top/bottom
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
        checks_passed=checks_passed,
        warnings=warnings,
    )

    return True, report, normalized_image
