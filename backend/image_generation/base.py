"""Abstract base class for generative image editing providers."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from PIL import Image


class ImageEditingProvider(ABC):
    """Abstract interface for image editing models (e.g. Local SDXL Inpainting / Gemini)."""

    @abstractmethod
    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        quality_tier: str = "fast",
        mask_image: Optional[Image.Image] = None,
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Perform multimodal image editing on an original photograph.
        
        Args:
            image: Source PIL Image.
            prompt: Rich spatial redesign instructions.
            negative_prompt: Elements to strictly avoid.
            model_name: Model identifier or path.
            quality_tier: 'fast' or 'final'.
            mask_image: Optional binary PIL Image mask (white = area to inpaint, black = preserve).
            
        Returns:
            Tuple of (redesigned_image, error_message).
        """
        pass
