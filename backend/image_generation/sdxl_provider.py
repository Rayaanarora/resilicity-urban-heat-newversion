"""Backwards compatibility shim redirecting to sd15_inpaint_provider.py.

The local generative model is Stable Diffusion 1.5 inpainting
(stable-diffusion-v1-5/stable-diffusion-inpainting), not SDXL.
"""

from .sd15_inpaint_provider import (
    LocalSD15InpaintingProvider,
    LocalSDXLInpaintingProvider,
    _compute_masked_diff,
    _compute_working_dimensions,
)

__all__ = [
    "LocalSD15InpaintingProvider",
    "LocalSDXLInpaintingProvider",
    "_compute_masked_diff",
    "_compute_working_dimensions",
]
