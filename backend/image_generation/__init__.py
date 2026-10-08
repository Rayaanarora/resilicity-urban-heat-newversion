"""Image generation and editing subsystem for ResiliCity centered on ControlNet Inpainting."""

from .base import ImageEditingProvider
from .gemini_provider import GeminiImageEditingProvider
from .sd15_controlnet_provider import (
    LocalSD15ControlNetInpaintingProvider,
    LocalSD15InpaintingProvider,
    LocalSDXLInpaintingProvider,
)
from .depth_util import preprocess_depth_for_controlnet
from .controlled_generation import (
    generate_controlled_intervention,
    build_intervention_prompt,
    extract_contextual_crop,
    feather_blend_crop_back,
)
from .mask_builder import (
    build_inpainting_mask,
    build_pass_mask,
    build_protected_object_mask,
    build_mask_from_intervention_geometry,
)
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
from .layout_engine import populate_intervention_explicit_geometry

__all__ = [
    "ImageEditingProvider",
    "GeminiImageEditingProvider",
    "LocalSD15ControlNetInpaintingProvider",
    "LocalSD15InpaintingProvider",
    "LocalSDXLInpaintingProvider",
    "preprocess_depth_for_controlnet",
    "generate_controlled_intervention",
    "build_intervention_prompt",
    "extract_contextual_crop",
    "feather_blend_crop_back",
    "build_inpainting_mask",
    "build_pass_mask",
    "build_protected_object_mask",
    "build_mask_from_intervention_geometry",
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
    "populate_intervention_explicit_geometry",
]
