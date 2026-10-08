"""Local SDXL Inpainting Provider with retry logic for autonomous urban redesign."""

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

logger = logging.getLogger("resilicity.sdxl")

DEFAULT_MODEL_PATH = r"D:\huggingface_cache\sdxl-inpainting"
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, 3d render, fantasy city, "
    "floating tree, deformed tree, duplicated objects, "
    "tree in road, tree inside building, unrealistic jungle, "
    "segmentation mask, colored overlay, text, watermark, "
    "blurry, low resolution, distorted architecture, "
    "unchanged image, same as original, no modification"
)

# Minimum mean pixel difference to consider the generation as visibly changed
MIN_MASKED_DIFF = 12.0
MIN_OVERALL_DIFF = 6.0


def _compute_working_dimensions(width: int, height: int, max_dim: int = 512) -> Tuple[int, int]:
    """Compute aspect-preserving dimensions rounded to multiples of 8 for SDXL."""
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


class LocalSDXLInpaintingProvider(ImageEditingProvider):
    """Local SDXL Inpainting provider running on NVIDIA RTX GPU with retry logic."""

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

            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

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

            if hasattr(pipe, "vae") and pipe.vae is not None:
                try:
                    pipe.vae.enable_slicing()
                    logger.info("VAE slicing enabled on pipe.vae")
                except Exception as e:
                    logger.warning("Could not enable VAE slicing: %s", e)

                try:
                    pipe.vae.enable_tiling()
                    logger.info("VAE tiling enabled on pipe.vae")
                except Exception as e:
                    logger.warning("Could not enable VAE tiling: %s", e)

            try:
                pipe.enable_attention_slicing("max")
                logger.info("Attention slicing (max) enabled on SDXL pipe")
            except Exception as e:
                logger.warning("Could not enable attention slicing: %s", e)

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
        """Run a single SDXL inference pass. Returns the generated image at working resolution."""
        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=input_img,
                mask_image=work_mask,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]

        if result.size != (work_w, work_h):
            result = result.resize((work_w, work_h), Image.Resampling.LANCZOS)

        # Composite newly synthesized content strictly inside the inpainting mask.
        # Unmasked areas (existing architecture, vehicles, perspective) are preserved bit-perfect from the source image.
        if work_mask is not None:
            # Subtle boundary blur for seamless organic edge blending
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
        """Run local SDXL inpainting with spatial mask, with optional dynamic params and retry."""
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

                # Working resolution: 512 for fast, 768 for final
                max_dim = self.max_dim if quality_tier == "fast" else min(768, self.max_dim + 256)
                work_w, work_h = _compute_working_dimensions(orig_w, orig_h, max_dim=max_dim)

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
                base_steps = steps or (self.steps if quality_tier == "fast" else min(28, self.steps + 6))
                base_strength = strength or self.strength
                base_guidance = guidance_scale or self.guidance_scale

                logger.info(
                    "Local SDXL Inpainting start: res=%dx%d steps=%d strength=%.2f scale=%.1f device=%s",
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
                        "SDXL attempt %d/%d: seed=%d guidance=%.1f strength=%.3f steps=%d",
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
                            "SDXL attempt %d result: overall_diff=%.2f, masked_diff=%.2f",
                            attempt + 1, overall_diff, masked_diff,
                        )

                        # Keep the best result
                        if masked_diff > best_diff:
                            best_diff = masked_diff
                            best_result = result

                        # Accept if the masked region changed enough
                        if masked_diff >= MIN_MASKED_DIFF:
                            logger.info(
                                "SDXL accepted on attempt %d (masked_diff=%.2f, overall=%.2f)",
                                attempt + 1, masked_diff, overall_diff,
                            )
                            break

                        logger.warning(
                            "SDXL attempt %d rejected: masked_diff=%.2f < %.2f (will retry)",
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
                        logger.error("SDXL generation error attempt %d: %s", attempt + 1, e, exc_info=True)
                        if attempt == max_retries - 1:
                            return None, f"Local SDXL generation error: {e}"
                        continue

                elapsed = time.time() - t0
                peak_vram = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if self.cuda_available
                    else 0.0
                )
                logger.info(
                    "Local SDXL Inpainting finished in %.2fs (%d attempts, Peak VRAM: %.2f GB, best_masked_diff=%.2f)",
                    elapsed, min(attempt + 1, max_retries), peak_vram, best_diff,
                )

                if best_result is None:
                    return None, "All SDXL attempts failed to produce output."

                # Upscale back to source photograph dimensions with high fidelity
                final_img = best_result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                return final_img, None

            return await loop.run_in_executor(None, _sync_generate)
