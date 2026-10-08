"""Image generation and editing subsystem for ResiliCity."""

from .base import ImageEditingProvider
from .gemini_provider import GeminiImageEditingProvider
from .sdxl_provider import LocalSDXLInpaintingProvider
from .mask_builder import build_inpainting_mask, build_pass_mask, build_protected_object_mask
from .prompts import (
    build_redesign_prompt,
    build_refinement_prompt,
    build_sdxl_inpainting_prompt,
    build_pass_sdxl_prompt,
)
from .multi_pass import run_autonomous_multi_pass_redesign
from .schemas import (
    DesignProfile,
    SpatialInterventionSpec,
    SpatialDesignPlan,
    SceneAnalysis,
    VisualizationOutput,
    ValidationReport,
    UnifiedRedesignResponse,
    RefinementIntent,
    RefinementResponse,
)
from .intent_parser import parse_refinement_intent
from .validation import validate_image_output, validate_pass_output

__all__ = [
    "ImageEditingProvider",
    "GeminiImageEditingProvider",
    "LocalSDXLInpaintingProvider",
    "build_inpainting_mask",
    "build_pass_mask",
    "build_protected_object_mask",
    "build_redesign_prompt",
    "build_refinement_prompt",
    "build_sdxl_inpainting_prompt",
    "build_pass_sdxl_prompt",
    "run_autonomous_multi_pass_redesign",
    "DesignProfile",
    "SpatialInterventionSpec",
    "SpatialDesignPlan",
    "SceneAnalysis",
    "VisualizationOutput",
    "ValidationReport",
    "UnifiedRedesignResponse",
    "RefinementIntent",
    "RefinementResponse",
    "parse_refinement_intent",
    "validate_image_output",
    "validate_pass_output",
]
