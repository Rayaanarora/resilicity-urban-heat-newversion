"""Local Stable Diffusion 1.5 + ControlNet Depth Inpainting Provider.

Fulfills Requirements 3, 4, 5, 6, 17, 22, 24:
- SD 1.5 inpainting model: stable-diffusion-v1-5/stable-diffusion-inpainting (local cache).
- Single ControlNet: lllyasviel/control_v11f1p_sd15_depth (local cache).
- Depth conditioning directly controls 3D perspective and structural placement.
- Hardware-optimized for NVIDIA RTX 3050 Laptop GPU (6 GB VRAM):
  - FP16 precision
  - Attention slicing
  - VAE slicing and VAE tiling
  - Local contextual crop working resolution (512x512)
  - Batch size 1, inference_mode, zero VRAM leaks
- Zero cloud APIs, zero external network requests, zero Gemini fallback.
"""

import asyncio
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter
import torch

from .base import ImageEditingProvider
from .depth_util import preprocess_depth_for_controlnet

logger = logging.getLogger("resilicity.controlnet")

DEFAULT_SD15_MODEL_PATH = r"D:\huggingface_cache\sd15-inpaint"
DEFAULT_CONTROLNET_PATH = r"D:\huggingface_cache\controlnet-depth-sd15"

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, illustration, painting, 3d render, CGI, artificial tree, cutout, sticker, "
    "floating object, distorted trunk, deformed foliage, plastic leaves, blurry, low resolution, "
    "malformed branches, unrealistic shadow, giant tree, miniature tree, fantasy, sci-fi"
)

# Minimum mean pixel difference in masked area to confirm generative change
MIN_MASKED_DIFF = 10.0


def _compute_masked_diff(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Tuple[float, float]:
    """Compute mean absolute pixel difference overall and strictly within masked region."""
    w, h = generated.size
    orig_resized = original.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    mask_resized = mask.convert("L").resize((w, h), Image.Resampling.NEAREST)

    orig_arr = np.array(orig_resized).astype(np.float32)
    gen_arr = np.array(generated.convert("RGB")).astype(np.float32)
    mask_arr = (np.array(mask_resized) > 30).astype(np.float32)

    abs_diff = np.abs(gen_arr - orig_arr)
    overall_diff = float(np.mean(abs_diff))

    mask_3d = np.stack([mask_arr] * 3, axis=2)
    masked_pixels = mask_3d.sum()
    if masked_pixels > 0:
        masked_diff = float((abs_diff * mask_3d).sum() / masked_pixels)
    else:
        masked_diff = overall_diff

    return overall_diff, masked_diff


class LocalSD15ControlNetInpaintingProvider(ImageEditingProvider):
    """Local SD 1.5 inpainting pipeline conditioned on Depth ControlNet."""

    _instance: Optional["LocalSD15ControlNetInpaintingProvider"] = None

    def __init__(self):
        self.model_path = os.environ.get("SD15_MODEL_PATH", DEFAULT_SD15_MODEL_PATH).strip()
        self.controlnet_path = os.environ.get("CONTROLNET_DEPTH_PATH", DEFAULT_CONTROLNET_PATH).strip()
        self.steps = int(os.environ.get("SD15_STEPS", "24"))
        self.guidance_scale = float(os.environ.get("SD15_GUIDANCE_SCALE", "7.5"))
        self.strength = float(os.environ.get("SD15_STRENGTH", "1.0"))
        self.controlnet_conditioning_scale = float(os.environ.get("CONTROLNET_SCALE", "0.80"))

        self.cuda_available = torch.cuda.is_available()
        self.gpu_name = torch.cuda.get_device_name(0) if self.cuda_available else "CPU"
        self.vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if self.cuda_available
            else 0.0
        )

        self.pipe = None
        self.controlnet = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "LocalSD15ControlNetInpaintingProvider":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_pipeline(self) -> None:
        """Load SD 1.5 inpainting and Depth ControlNet models on CUDA with memory optimizations."""
        if self.is_loaded and self.pipe is not None:
            return

        if not self.cuda_available:
            self.load_error = "CUDA GPU is required for local ControlNet inpainting."
            logger.error(self.load_error)
            return

        try:
            logger.info("Initializing Local SD 1.5 + Depth ControlNet pipeline...")
            logger.info("SD 1.5 Base Model: %s", self.model_path)
            logger.info("Depth ControlNet: %s", self.controlnet_path)
            t0 = time.time()

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

            from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

            # 1. Load Depth ControlNet
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_path,
                torch_dtype=torch.float16,
                local_files_only=True,
            )

            # 2. Load SD 1.5 Inpainting Pipeline with ControlNet
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                self.model_path,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                local_files_only=True,
                safety_checker=None,
            ).to("cuda")

            # 3. Memory optimizations for 6 GB RTX 3050 Laptop GPU
            pipe.enable_attention_slicing()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

            self.controlnet = controlnet
            self.pipe = pipe
            self.is_loaded = True
            self.load_error = None

            allocated_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
            logger.info(
                "Local SD 1.5 + Depth ControlNet loaded in %.2fs on %s (VRAM allocated: %.2f GB / %.2f GB)",
                time.time() - t0, self.gpu_name, allocated_gb, self.vram_gb,
            )
        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load ControlNet inpainting pipeline: {e}"
            self.pipe = None
            self.controlnet = None
            logger.error(self.load_error, exc_info=True)

    def get_status(self) -> Dict[str, Any]:
        """Expose detailed provider status for runtime tracing and health checks."""
        return {
            "available": self.is_loaded or (Path(self.model_path).exists() and Path(self.controlnet_path).exists()),
            "loaded": self.is_loaded,
            "provider": "LocalSD15ControlNetInpaintingProvider",
            "model": "stable-diffusion-v1-5/stable-diffusion-inpainting",
            "controlnet_model": "lllyasviel/control_v11f1p_sd15_depth",
            "model_path": self.model_path,
            "controlnet_path": self.controlnet_path,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "error": self.load_error,
        }

    def _run_inference_sync(
        self,
        crop_image: Image.Image,
        crop_mask: Image.Image,
        depth_condition: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int,
        strength: float,
        controlnet_conditioning_scale: float,
        steps: int,
        guidance_scale: float,
        work_size: Tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """Run single synchronous diffusion pass on CUDA."""
        w, h = work_size
        img_512 = crop_image.convert("RGB").resize(work_size, Image.Resampling.LANCZOS)
        mask_512 = crop_mask.convert("L").resize(work_size, Image.Resampling.NEAREST)
        # Ensure mask is strictly binary
        mask_arr = np.where(np.array(mask_512) > 30, 255, 0).astype(np.uint8)
        mask_binary = Image.fromarray(mask_arr, mode="L")

        depth_512 = depth_condition.convert("RGB").resize(work_size, Image.Resampling.LANCZOS)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        with torch.inference_mode():
            raw_output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_512,
                mask_image=mask_binary,
                control_image=depth_512,
                height=h,
                width=w,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
            ).images[0]

        # Soft feather composite generated content strictly within intervention mask
        soft_mask = mask_binary.filter(ImageFilter.GaussianBlur(radius=1.8))
        composited = Image.composite(raw_output, img_512, soft_mask)
        return composited

    async def generate_intervention(
        self,
        image: Image.Image,
        mask: Image.Image,
        depth_condition: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
        strength: Optional[float] = None,
        controlnet_conditioning_scale: Optional[float] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        max_retries: int = 2,
    ) -> Tuple[Optional[Image.Image], Optional[str], Dict[str, Any]]:
        """Primary generative inpainting execution with ControlNet depth guidance and quality checks.

        Returns:
            (generated_image, error_message, trace_dict)
        """
        if not self.cuda_available:
            return None, "CUDA GPU is not available for local generation.", {}

        neg_prompt = (negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT)
        base_steps = steps or self.steps
        base_strength = strength if strength is not None else self.strength
        base_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        base_cnet_scale = (
            controlnet_conditioning_scale
            if controlnet_conditioning_scale is not None
            else self.controlnet_conditioning_scale
        )

        trace: Dict[str, Any] = {
            "provider": "LocalSD15ControlNetInpaintingProvider",
            "model": "stable-diffusion-v1-5/stable-diffusion-inpainting",
            "controlnet_model": "lllyasviel/control_v11f1p_sd15_depth",
            "steps": base_steps,
            "guidance_scale": base_guidance,
            "strength": base_strength,
            "controlnet_conditioning_scale": base_cnet_scale,
            "device": self.gpu_name,
            "attempts": 0,
            "best_masked_diff": 0.0,
            "peak_vram_gb": 0.0,
            "execution_time_s": 0.0,
        }

        async with self._lock:
            loop = asyncio.get_running_loop()

            def _sync_worker() -> Tuple[Optional[Image.Image], Optional[str], Dict[str, Any]]:
                if not self.is_loaded:
                    self.load_pipeline()
                    if not self.is_loaded or self.pipe is None:
                        return None, self.load_error or "ControlNet pipeline failed to load.", trace

                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                best_result = None
                best_diff = 0.0

                for attempt in range(max_retries):
                    trace["attempts"] = attempt + 1
                    cur_seed = (
                        seed if (seed is not None and attempt == 0)
                        else (42 if attempt == 0 else random.randint(1, 999999))
                    )
                    cur_guidance = base_guidance + (attempt * 1.0)
                    cur_steps = base_steps + (attempt * 2)

                    logger.info(
                        "SD 1.5 + ControlNet attempt %d/%d: seed=%d steps=%d scale=%.1f cnet_scale=%.2f",
                        attempt + 1, max_retries, cur_seed, cur_steps, cur_guidance, base_cnet_scale,
                    )

                    try:
                        result = self._run_inference_sync(
                            crop_image=image,
                            crop_mask=mask,
                            depth_condition=depth_condition,
                            prompt=prompt,
                            negative_prompt=neg_prompt,
                            seed=cur_seed,
                            strength=base_strength,
                            controlnet_conditioning_scale=base_cnet_scale,
                            steps=cur_steps,
                            guidance_scale=cur_guidance,
                            work_size=(512, 512),
                        )

                        overall_diff, masked_diff = _compute_masked_diff(image, result, mask)
                        logger.info(
                            "Attempt %d diff: masked=%.2f, overall=%.2f (threshold=%.1f)",
                            attempt + 1, masked_diff, overall_diff, MIN_MASKED_DIFF,
                        )

                        if masked_diff > best_diff:
                            best_diff = masked_diff
                            best_result = result
                            trace["seed"] = cur_seed

                        if masked_diff >= MIN_MASKED_DIFF:
                            logger.info("Attempt %d accepted (masked_diff=%.2f >= %.1f)", attempt + 1, masked_diff, MIN_MASKED_DIFF)
                            break

                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        logger.error("CUDA OOM on attempt %d", attempt + 1)
                        if attempt == max_retries - 1:
                            return None, "CUDA Out of Memory on RTX 3050 (6 GB).", trace
                    except Exception as e:
                        torch.cuda.empty_cache()
                        logger.error("ControlNet generation error on attempt %d: %s", attempt + 1, e, exc_info=True)
                        if attempt == max_retries - 1:
                            return None, f"Local ControlNet generation error: {e}", trace

                trace["best_masked_diff"] = round(best_diff, 2)
                trace["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
                trace["execution_time_s"] = round(time.time() - t0, 2)

                if best_result is None:
                    return None, "All ControlNet generation attempts failed.", trace

                # Resize result back to match input image dimensions
                final_output = best_result.resize((image.width, image.height), Image.Resampling.LANCZOS)
                return final_output, None, trace

            return await loop.run_in_executor(None, _sync_worker)

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
        """Fallback ImageEditingProvider interface implementation."""
        w, h = image.size
        if mask_image is None:
            mask_image = Image.new("L", (w, h), 255)

        depth_cond = preprocess_depth_for_controlnet(
            np.tile(np.linspace(1.0, 0.0, h)[:, None], (1, w)),
            target_size=(w, h),
        )

        result, err, _ = await self.generate_intervention(
            image=image,
            mask=mask_image,
            depth_condition=depth_cond,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            strength=strength,
            steps=steps,
            guidance_scale=guidance_scale,
            max_retries=max_retries,
        )
        return result, err


# Standard aliases for clean provider discovery
LocalSD15InpaintingProvider = LocalSD15ControlNetInpaintingProvider
LocalSDXLInpaintingProvider = LocalSD15ControlNetInpaintingProvider
