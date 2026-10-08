"""Monocular Depth Estimation for Relative Spatial Scene Understanding.

Uses the lightweight Depth-Anything-V2-Small model (cached locally) for relative depth geometry.
Depth is normalized to [0.0, 1.0] where 0.0 is nearest to camera and 1.0 is farthest (horizon/infinity).
Runs in inference_mode, keeping VRAM minimal, with a robust geometric perspective fallback.
"""

import logging
import os
from typing import Optional, Tuple
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("resilicity.depth")

_DEPTH_PIPE = None


def get_depth_pipeline():
    """Lazily load lightweight depth pipeline on CPU or GPU if enabled."""
    global _DEPTH_PIPE
    if _DEPTH_PIPE is not None:
        return _DEPTH_PIPE

    if os.environ.get("USE_DEPTH_ANYTHING", "false").lower() != "true":
        return None

    try:
        from transformers import pipeline
        device = 0 if (torch.cuda.is_available() and os.environ.get("DEPTH_ON_GPU", "false").lower() == "true") else -1
        logger.info("Initializing Depth-Anything-V2 pipeline (device=%s)...", device)
        _DEPTH_PIPE = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )
        logger.info("Depth-Anything-V2-Small initialized successfully.")
    except Exception as e:
        logger.warning("Could not initialize Depth-Anything pipeline: %s. Using geometric fallback.", e)
        _DEPTH_PIPE = None

    return _DEPTH_PIPE


def estimate_relative_depth(image: Image.Image) -> np.ndarray:
    """Estimate relative depth map for a PIL Image.
    
    Returns:
        np.ndarray: 2D float32 array of shape (height, width) with values in [0.0, 1.0],
                    where 0.0 is nearest foreground and 1.0 is farthest background/sky.
    """
    w, h = image.size
    pipe = get_depth_pipeline()

    if pipe is not None:
        try:
            # Resize image to fast working resolution for depth inference (e.g. max 518)
            scale = min(518 / max(w, h), 1.0)
            dw, dh = int(w * scale), int(h * scale)
            small_img = image.resize((dw, dh), Image.Resampling.BILINEAR)

            depth_out = pipe(small_img)["depth"]
            # depth_out is a PIL Image with predicted relative depth
            depth_arr = np.array(depth_out).astype(np.float32)

            # In Depth-Anything, larger values represent closer depth (disparity-like) or distance.
            # Normalize to [0.0, 1.0] where 0.0 is nearest to camera, 1.0 is farthest.
            d_min, d_max = float(np.min(depth_arr)), float(np.max(depth_arr))
            if d_max > d_min:
                norm_depth = (depth_arr - d_min) / (d_max - d_min)
            else:
                norm_depth = np.zeros_like(depth_arr)

            # If upper pixels (usually sky) have low values, invert so that 0=near (bottom), 1=far (top/sky)
            top_mean = float(np.mean(norm_depth[:int(dh * 0.2), :]))
            bottom_mean = float(np.mean(norm_depth[int(dh * 0.8):, :]))
            if top_mean < bottom_mean:
                norm_depth = 1.0 - norm_depth

            # Resize to original dimensions
            norm_depth_pil = Image.fromarray((norm_depth * 255.0).astype(np.uint8)).resize(
                (w, h), Image.Resampling.BILINEAR
            )
            return (np.array(norm_depth_pil).astype(np.float32) / 255.0).clip(0.0, 1.0)

        except Exception as e:
            logger.warning("Depth inference failed: %s. Using geometric perspective fallback.", e)

    # Clean geometric perspective fallback (calibrated ground plane + vertical distance gradient)
    return compute_geometric_depth_fallback(w, h)


def compute_geometric_depth_fallback(width: int, height: int, horizon_ratio: float = 0.45) -> np.ndarray:
    """Create a physically plausible relative depth gradient based on perspective convergence."""
    y_coords = np.linspace(0, 1, height)[:, None]
    # Distances scale inversely from bottom (y=1, near=0.0) up to horizon (y=horizon_ratio, far=1.0)
    horizon_y = horizon_ratio
    depth = np.where(
        y_coords >= horizon_y,
        (1.0 - (y_coords - horizon_y) / (1.0 - horizon_y)),  # 0 at bottom, 1 at horizon
        1.0,  # 1.0 for sky / upper region
    ).astype(np.float32)
    # Broadcast to full image width
    return np.repeat(depth, width, axis=1).clip(0.0, 1.0)
