"""Local SDXL Inpainting Provider implementation for autonomous urban redesign."""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image

import torch

from .base import ImageEditingProvider

logger = logging.getLogger("resilicity.sdxl")

DEFAULT_MODEL_PATH = r"D:\huggingface_cache\sdxl-inpainting"
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, 3d render, fantasy city, "
    "floating tree, deformed tree, duplicated objects, "
    "tree in road, tree inside building, unrealistic jungle, "
    "segmentation mask, colored overlay, text, watermark, "
    "blurry, low resolution, distorted architecture"
)


def _compute_working_dimensions(width: int, height: int, max_dim: int = 512) -> Tuple[int, int]:
    """Compute aspect-preserving dimensions rounded to multiples of 8 for SDXL."""
    scale = min(max_dim / max(width, height), 1.0)
    w_scaled = int(round((width * scale) / 8.0) * 8)
    h_scaled = int(round((height * scale) / 8.0) * 8)
    return max(w_scaled, 64), max(h_scaled, 64)


class LocalSDXLInpaintingProvider(ImageEditingProvider):
    """Local SDXL Inpainting provider running on NVIDIA RTX GPU."""

    _instance: Optional["LocalSDXLInpaintingProvider"] = None

    def __init__(self):
        self.model_path = os.environ.get("SDXL_MODEL_PATH", DEFAULT_MODEL_PATH).strip()
        self.steps = int(os.environ.get("SDXL_STEPS", "20"))
        self.guidance_scale = float(os.environ.get("SDXL_GUIDANCE_SCALE", "7.5"))
        self.strength = float(os.environ.get("SDXL_STRENGTH", "0.99"))
        self.max_dim = int(os.environ.get("SDXL_MAX_DIM", "512"))
        self.use_cpu_offload = os.environ.get("SDXL_CPU_OFFLOAD", "true").lower() in ("true", "1", "yes")

        self.cuda_available = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.cuda_available else "CPU"
        self.vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if self.cuda_available
            else 0.0
        )

        self.pipe = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "LocalSDXLInpaintingProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_pipeline(self) -> None:
        """Load and configure the SDXL inpainting pipeline with VRAM optimizations."""
        if self.is_loaded and self.pipe is not None:
            return

        if not self.cuda_available:
            self.load_error = "CUDA is not available. GPU acceleration required."
            logger.error(self.load_error)
            return

        try:
            logger.info("Loading local SDXL inpainting model from: %s", self.model_path)
            t0 = time.time()
            from diffusers import AutoPipelineForInpainting

            pipe = AutoPipelineForInpainting.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=True,
            )

            # Essential for 6GB VRAM (RTX 3050 Laptop)
            if self.use_cpu_offload:
                pipe.enable_model_cpu_offload()

            try:
                pipe.enable_vae_slicing()
            except Exception as e:
                logger.warning("Could not enable VAE slicing: %s", e)

            try:
                pipe.enable_vae_tiling()
            except Exception as e:
                logger.warning("Could not enable VAE tiling: %s", e)

            self.pipe = pipe
            self.is_loaded = True
            self.load_error = None
            logger.info("SDXL model loaded in %.2f seconds on %s", time.time() - t0, self.gpu_name)
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load SDXL pipeline: {e}"
            self.pipe = None
            logger.error(self.load_error, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Expose generator status for health checks."""
        return {
            "available": self.is_loaded or Path(self.model_path).exists(),
            "loaded": self.is_loaded,
            "model": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            "model_path": self.model_path,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "error": self.load_error,
        }

    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        quality_tier: str = "fast",
        mask_image: Optional[Image.Image] = None,
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Run local SDXL inpainting with spatial mask and identity preservation."""
        if not self.cuda_available:
            return None, "CUDA GPU is not available for local generation."

        async with self._lock:
            loop = asyncio.get_running_loop()

            def _sync_generate() -> Tuple[Optional[Image.Image], Optional[str]]:
                if not self.is_loaded:
                    self.load_pipeline()
                    if not self.is_loaded or self.pipe is None:
                        return None, self.load_error or "SDXL pipeline failed to load."

                orig_w, orig_h = image.size

                # Working resolution: 512 for fast, 768 for final tier if requested
                max_dim = self.max_dim if quality_tier == "fast" else min(768, self.max_dim + 256)
                work_w, work_h = _compute_working_dimensions(orig_w, orig_h, max_dim=max_dim)

                input_img = image.convert("RGB").resize((work_w, work_h), Image.Resampling.LANCZOS)

                # Prepare inpainting mask
                if mask_image is not None:
                    work_mask = mask_image.convert("L").resize((work_w, work_h), Image.Resampling.NEAREST)
                else:
                    # Fallback mask: roadside planting corridor (lower 40% height on left/right edges)
                    from PIL import ImageDraw
                    work_mask = Image.new("L", (work_w, work_h), 0)
                    draw = ImageDraw.Draw(work_mask)
                    draw.rectangle([0, int(work_h * 0.5), int(work_w * 0.35), work_h], fill=255)
                    draw.rectangle([int(work_w * 0.65), int(work_h * 0.5), work_w, work_h], fill=255)

                neg_prompt = (negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT)

                # Inference steps based on quality tier
                steps = self.steps if quality_tier == "fast" else min(28, self.steps + 6)

                logger.info(
                    "Local SDXL Inpainting start: res=%dx%d steps=%d scale=%.1f device=%s",
                    work_w,
                    work_h,
                    steps,
                    self.guidance_scale,
                    self.gpu_name,
                )

                if self.cuda_available:
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                generator = torch.Generator(device="cuda").manual_seed(42)

                try:
                    with torch.inference_mode():
                        result = self.pipe(
                            prompt=prompt,
                            negative_prompt=neg_prompt,
                            image=input_img,
                            mask_image=work_mask,
                            strength=self.strength,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=steps,
                            generator=generator,
                        ).images[0]

                    elapsed = time.time() - t0
                    peak_vram = (
                        torch.cuda.max_memory_allocated() / (1024 ** 3)
                        if self.cuda_available
                        else 0.0
                    )
                    logger.info(
                        "Local SDXL Inpainting finished in %.2fs (Peak VRAM: %.2f GB)",
                        elapsed,
                        peak_vram,
                    )

                    # Upscale back to source photograph dimensions with high fidelity
                    final_img = result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                    return final_img, None

                except torch.cuda.OutOfMemoryError:
                    if self.cuda_available:
                        torch.cuda.empty_cache()
                    logger.error("CUDA OOM error during SDXL inpainting generation.")
                    return None, "CUDA Out of Memory on RTX 3050. Reduce working resolution or batch."
                except Exception as e:
                    if self.cuda_available:
                        torch.cuda.empty_cache()
                    logger.error("SDXL generation error: %s", e, exc_info=True)
                    return None, f"Local SDXL generation error: {e}"

            return await loop.run_in_executor(None, _sync_generate)
