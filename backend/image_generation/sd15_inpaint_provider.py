"""Local Stable Diffusion 1.5 Inpainting Provider for autonomous urban redesign.

Runs on NVIDIA RTX GPU (RTX 3050 6 GB safe: FP16, VAE slicing, memory efficient).
Uses local weights (D:\\huggingface_cache\\sd15-inpaint or stable-diffusion-v1-5/stable-diffusion-inpainting).
Zero external network requests, zero cloud API costs.
"""

import asyncio
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageFilter

import torch
import numpy as np

from .base import ImageEditingProvider

logger = logging.getLogger("resilicity.sd15")

DEFAULT_MODEL_PATH = r"D:\huggingface_cache\sd15-inpaint"
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, 3d render, blurry, distorted, "
    "low resolution, bad architecture, floating objects, fantasy"
)

# Minimum mean pixel difference to consider the generation as visibly changed
MIN_MASKED_DIFF = 8.0
MIN_OVERALL_DIFF = 4.0


def _compute_working_dimensions(width: int, height: int, max_dim: int = 1024) -> Tuple[int, int]:
    """Compute aspect-preserving dimensions rounded to multiples of 8."""
    scale = min(max_dim / max(width, height), 1.0)
    w_scaled = int(round((width * scale) / 8.0) * 8)
    h_scaled = int(round((height * scale) / 8.0) * 8)
    return max(w_scaled, 64), max(h_scaled, 64)


def _compute_masked_diff(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Tuple[float, float]:
    """Compute mean pixel difference overall and within the masked region only."""
    w, h = generated.size
    orig_resized = original.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    mask_resized = mask.convert("L").resize((w, h), Image.Resampling.NEAREST)

    orig_arr = np.array(orig_resized).astype(np.float32)
    gen_arr = np.array(generated.convert("RGB")).astype(np.float32)
    mask_arr = (np.array(mask_resized) > 30).astype(np.float32)

    abs_diff = np.abs(gen_arr - orig_arr)
    overall_diff = float(np.mean(abs_diff))

    # Masked-region difference
    mask_3d = np.stack([mask_arr] * 3, axis=2)
    masked_pixels = mask_3d.sum()
    if masked_pixels > 0:
        masked_diff = float((abs_diff * mask_3d).sum() / masked_pixels)
    else:
        masked_diff = overall_diff

    return overall_diff, masked_diff


class LocalSD15InpaintingProvider(ImageEditingProvider):
    """Local SD 1.5 inpainting provider running on NVIDIA RTX GPU with padding_mask_crop."""

    _instance: Optional["LocalSD15InpaintingProvider"] = None

    def __init__(self):
        self.model_path = os.environ.get("SD15_MODEL_PATH", os.environ.get("SDXL_MODEL_PATH", DEFAULT_MODEL_PATH)).strip()
        self.steps = int(os.environ.get("SD15_STEPS", os.environ.get("SDXL_STEPS", "20")))
        self.guidance_scale = float(os.environ.get("SD15_GUIDANCE_SCALE", os.environ.get("SDXL_GUIDANCE_SCALE", "7.5")))
        self.strength = float(os.environ.get("SD15_STRENGTH", os.environ.get("SDXL_STRENGTH", "1.0")))
        self.max_dim = int(os.environ.get("SD15_MAX_DIM", os.environ.get("SDXL_MAX_DIM", "1024")))
        self.use_cpu_offload = os.environ.get("SD15_CPU_OFFLOAD", "false").lower() in ("true", "1", "yes")

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
    def get_instance(cls) -> "LocalSD15InpaintingProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_pipeline(self) -> None:
        """Load and configure the SD 1.5 inpainting pipeline directly on CUDA."""
        if self.is_loaded and self.pipe is not None:
            return

        if not self.cuda_available:
            self.load_error = "CUDA is not available. GPU acceleration required."
            logger.error(self.load_error)
            return

        try:
            logger.info("Loading local SD 1.5 inpainting model from: %s", self.model_path)
            t0 = time.time()

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

            from diffusers import StableDiffusionInpaintPipeline

            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    local_files_only=True,
                    safety_checker=None,
                ).to("cuda")
            except Exception:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    safety_checker=None,
                ).to("cuda")

            if hasattr(pipe, "enable_vae_slicing"):
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                try:
                    pipe.vae.enable_slicing()
                except Exception:
                    pass

            self.pipe = pipe
            self.is_loaded = True
            self.load_error = None
            logger.info("SD 1.5 inpainting model loaded in %.2f seconds on %s", time.time() - t0, self.gpu_name)
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load SD 1.5 pipeline: {e}"
            self.pipe = None
            logger.error(self.load_error, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Expose generator status for health checks."""
        return {
            "available": self.is_loaded or Path(self.model_path).exists(),
            "loaded": self.is_loaded,
            "model": "stable-diffusion-v1-5/stable-diffusion-inpainting",
            "model_path": self.model_path,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "error": self.load_error,
        }

    def _run_inference(
        self,
        input_img: Image.Image,
        work_mask: Image.Image,
        prompt: str,
        neg_prompt: str,
        steps: int,
        strength: float,
        guidance_scale: float,
        seed: int,
        work_w: int,
        work_h: int,
    ) -> Image.Image:
        """Run a single SD inpainting pass with padding_mask_crop. Returns the image at working resolution."""
        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=input_img,
                mask_image=work_mask,
                height=512,
                width=512,
                padding_mask_crop=48,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

        if result.size != (work_w, work_h):
            result = result.resize((work_w, work_h), Image.Resampling.LANCZOS)

        # Composite newly synthesized content strictly inside the inpainting mask.
        if work_mask is not None:
            mask_blend = work_mask.convert("L").filter(ImageFilter.GaussianBlur(radius=1.2))
            result = Image.composite(result, input_img, mask_blend)

        return result

    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        model_name: Optional[str] = None,
        quality_tier: str = "fast",
        mask_image: Optional[Image.Image] = None,
        seed: Optional[int] = None,
        strength: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        steps: Optional[int] = None,
        max_retries: int = 1,
    ) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Run local SD 1.5 inpainting with spatial mask, with optional dynamic params and retry."""
        if not self.cuda_available:
            return None, "CUDA GPU is not available for local generation."

        async with self._lock:
            loop = asyncio.get_running_loop()

            def _sync_generate() -> Tuple[Optional[Image.Image], Optional[str]]:
                if not self.is_loaded:
                    self.load_pipeline()
                    if not self.is_loaded or self.pipe is None:
                        return None, self.load_error or "SD 1.5 pipeline failed to load."

                orig_w, orig_h = image.size

                # Working resolution: keep full street resolution up to 1024 since padding_mask_crop handles 512x512 inpainting
                work_w, work_h = _compute_working_dimensions(orig_w, orig_h, max_dim=self.max_dim)

                input_img = image.convert("RGB").resize((work_w, work_h), Image.Resampling.LANCZOS)

                # Prepare inpainting mask — ensure BINARY (0 or 255)
                if mask_image is not None:
                    work_mask = mask_image.convert("L").resize((work_w, work_h), Image.Resampling.NEAREST)
                    mask_arr = np.array(work_mask)
                    mask_arr = np.where(mask_arr > 30, 255, 0).astype(np.uint8)
                    work_mask = Image.fromarray(mask_arr, mode="L")
                else:
                    from PIL import ImageDraw
                    work_mask = Image.new("L", (work_w, work_h), 0)
                    draw = ImageDraw.Draw(work_mask)
                    draw.rectangle([0, int(work_h * 0.4), int(work_w * 0.30), work_h], fill=255)
                    draw.rectangle([int(work_w * 0.70), int(work_h * 0.4), work_w, work_h], fill=255)
                    draw.rectangle([int(work_w * 0.10), int(work_h * 0.55), int(work_w * 0.90), work_h], fill=255)

                neg_prompt = (negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT)

                # Inference steps based on quality tier or caller override
                base_steps = steps or self.steps
                base_strength = strength if strength is not None else self.strength
                base_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

                logger.info(
                    "Local SD 1.5 Inpainting start: res=%dx%d steps=%d strength=%.2f scale=%.1f device=%s",
                    work_w, work_h, base_steps, base_strength, base_guidance, self.gpu_name,
                )

                if self.cuda_available:
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.time()
                best_result = None
                best_diff = 0.0

                for attempt in range(max_retries):
                    current_seed = seed if (seed is not None and attempt == 0) else (42 if attempt == 0 else random.randint(1, 999999))
                    attempt_guidance = base_guidance + (attempt * 1.5)
                    attempt_strength = min(1.0, base_strength + (attempt * 0.003))
                    attempt_steps = base_steps + (attempt * 2)

                    logger.info(
                        "SD 1.5 attempt %d/%d: seed=%d guidance=%.1f strength=%.3f steps=%d",
                        attempt + 1, max_retries, current_seed, attempt_guidance, attempt_strength, attempt_steps,
                    )

                    try:
                        result = self._run_inference(
                            input_img=input_img,
                            work_mask=work_mask,
                            prompt=prompt,
                            neg_prompt=neg_prompt,
                            steps=attempt_steps,
                            strength=attempt_strength,
                            guidance_scale=attempt_guidance,
                            seed=current_seed,
                            work_w=work_w,
                            work_h=work_h,
                        )

                        # Check if the result is visibly different from input
                        overall_diff, masked_diff = _compute_masked_diff(input_img, result, work_mask)

                        logger.info(
                            "SD 1.5 attempt %d result: overall_diff=%.2f, masked_diff=%.2f",
                            attempt + 1, overall_diff, masked_diff,
                        )

                        # Keep the best result
                        if masked_diff > best_diff:
                            best_diff = masked_diff
                            best_result = result

                        # Accept if the masked region changed enough
                        if masked_diff >= MIN_MASKED_DIFF:
                            logger.info(
                                "SD 1.5 accepted on attempt %d (masked_diff=%.2f, overall=%.2f)",
                                attempt + 1, masked_diff, overall_diff,
                            )
                            break

                        logger.warning(
                            "SD 1.5 attempt %d rejected: masked_diff=%.2f < %.2f (will retry)",
                            attempt + 1, masked_diff, MIN_MASKED_DIFF,
                        )

                    except torch.cuda.OutOfMemoryError:
                        if self.cuda_available:
                            torch.cuda.empty_cache()
                        logger.error("CUDA OOM on attempt %d", attempt + 1)
                        if attempt == max_retries - 1:
                            return None, "CUDA Out of Memory on RTX 3050."
                        continue
                    except Exception as e:
                        if self.cuda_available:
                            torch.cuda.empty_cache()
                        logger.error("SD 1.5 generation error attempt %d: %s", attempt + 1, e, exc_info=True)
                        if attempt == max_retries - 1:
                            return None, f"Local SD 1.5 generation error: {e}"
                        continue

                elapsed = time.time() - t0
                peak_vram = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if self.cuda_available
                    else 0.0
                )
                logger.info(
                    "Local SD 1.5 Inpainting finished in %.2fs (%d attempts, Peak VRAM: %.2f GB, best_masked_diff=%.2f)",
                    elapsed, min(attempt + 1, max_retries), peak_vram, best_diff,
                )

                if best_result is None:
                    return None, "All SD 1.5 attempts failed to produce output."

                # Upscale back to source photograph dimensions with high fidelity
                final_img = best_result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                return final_img, None

            return await loop.run_in_executor(None, _sync_generate)


# Backwards compatibility alias for SDXL named references
LocalSDXLInpaintingProvider = LocalSD15InpaintingProvider
