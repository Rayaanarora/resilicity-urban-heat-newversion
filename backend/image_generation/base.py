"""Abstract base class for generative image editing providers."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from PIL import Image

from .schemas import SpatialDesignPlan, VisualizationOutput


class ImageEditingProvider(ABC):
    """Abstract interface for image editing models (e.g. Gemini 3.1 Flash / Gemini 3 Pro)."""

    @abstractmethod
    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        quality_tier: str = "fast",
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Perform multimodal image editing on an original photograph.
        
        Args:
            image: Source PIL Image.
            prompt: Rich spatial redesign instructions.
            negative_prompt: Elements to strictly avoid.
            model_name: Model identifier (e.g. gemini-3.1-flash-image).
            quality_tier: 'fast' or 'final'.
            
        Returns:
            Tuple of (redesigned_image, error_message).
        """
        pass
