"""Image generation subsystem for ResiliCity, built on instruction-based editing (FLUX.1 Kontext)."""

from .base import ImageEditingProvider
from .gemini_provider import GeminiImageEditingProvider
from .kontext_provider import FalKontextProvider, HuggingFaceSpaceKontextProvider
from .redesign_engine import GenerationResult, RedesignEngine
from .prompts import (
    build_kontext_redesign_prompt,
    build_kontext_refinement_prompt,
    build_redesign_prompt,
    build_refinement_prompt,
    illustrated_zones,
)
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
from .validation import validate_image_output

__all__ = [
    "ImageEditingProvider",
    "GeminiImageEditingProvider",
    "FalKontextProvider",
    "HuggingFaceSpaceKontextProvider",
    "GenerationResult",
    "RedesignEngine",
    "build_kontext_redesign_prompt",
    "build_kontext_refinement_prompt",
    "build_redesign_prompt",
    "build_refinement_prompt",
    "illustrated_zones",
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
]
