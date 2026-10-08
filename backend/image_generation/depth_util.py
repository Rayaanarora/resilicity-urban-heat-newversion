"""Depth Preprocessing for ControlNet Conditioning.

Fulfills Requirement 16:
- Normalizes depth values to [0.0, 1.0].
- Converts to expected ControlNet Depth format (MiDaS disparity: closer objects are brighter/255, farther objects are darker/0).
- Preserves relative ordering and depth gradients.
- Resizes consistently with image crops using high-quality interpolation.
- Ensures correct RGB 3-channel uint8 PIL Image format.
- Logs min, max, mean statistics.
- Saves debug depth-conditioning image: depth_condition.png for each crop.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger("resilicity.depth_util")


def preprocess_depth_for_controlnet(
    depth_input: Union[np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = None,
    save_debug_path: Optional[Path] = None,
    invert_if_depth: bool = True,
) -> Image.Image:
    """Convert depth estimation array or image into ControlNet Depth conditioning format.

    Args:
        depth_input: 2D numpy array [H, W] or PIL Image of depth map.
        target_size: Optional (width, height) to resize to.
        save_debug_path: Optional Path to save 'depth_condition.png'.
        invert_if_depth: If True, detects if 0=near and inverts so near=255 (MiDaS disparity).

    Returns:
        PIL.Image.Image in RGB mode with uint8 values in [0, 255].
    """
    if isinstance(depth_input, Image.Image):
        depth_arr = np.array(depth_input.convert("L")).astype(np.float32) / 255.0
    else:
        depth_arr = np.array(depth_input, dtype=np.float32)

    if depth_arr.ndim == 3:
        depth_arr = depth_arr[:, :, 0]

    h, w = depth_arr.shape
    d_min, d_max = float(np.min(depth_arr)), float(np.max(depth_arr))
    d_mean = float(np.mean(depth_arr))

    # Normalize to [0.0, 1.0]
    if d_max > d_min:
        norm = (depth_arr - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(depth_arr)

    # In street scenes, the lower portion of the image is near the camera (foreground street/sidewalk).
    # The upper portion is the sky / vanishing point (far).
    # ControlNet Depth (trained on MiDaS / DPT) expects near objects to be BRIGHT (255) and far to be DARK (0).
    if invert_if_depth:
        bot_mean = float(np.mean(norm[int(h * 0.75):, :]))
        top_mean = float(np.mean(norm[:int(h * 0.25), :]))
        if bot_mean < top_mean:
            # Foreground is darker than background -> invert so foreground is bright
            norm = 1.0 - norm

    # Log min, max, mean
    disp_min = float(np.min(norm)) * 255.0
    disp_max = float(np.max(norm)) * 255.0
    disp_mean = float(np.mean(norm)) * 255.0
    logger.debug(
        "Depth preprocessing for ControlNet: raw_range=[%.3f, %.3f, mean=%.3f] -> disparity_range=[%.1f, %.1f, mean=%.1f]",
        d_min, d_max, d_mean, disp_min, disp_max, disp_mean,
    )

    uint8_disp = (norm * 255.0).clip(0, 255).astype(np.uint8)
    disp_pil = Image.fromarray(uint8_disp, mode="L")

    if target_size is not None and (disp_pil.width, disp_pil.height) != target_size:
        disp_pil = disp_pil.resize(target_size, Image.Resampling.LANCZOS)

    # Convert to 3-channel RGB for ControlNet
    depth_rgb = Image.merge("RGB", (disp_pil, disp_pil, disp_pil))

    if save_debug_path is not None:
        save_debug_path.parent.mkdir(parents=True, exist_ok=True)
        depth_rgb.save(save_debug_path)
        logger.debug("Saved depth-conditioning debug artifact: %s", save_debug_path)

    return depth_rgb
