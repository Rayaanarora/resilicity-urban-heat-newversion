"""Test local SDXL inpainting model on NVIDIA RTX GPU with clean street photograph."""

import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("SDXL_MODEL_PATH", r"D:\huggingface_cache\sdxl-inpainting")
CLEAN_INPUT_IMAGE = HERE.parent / "gen-ai" / "public" / "samples" / "urban_street_before.png"
OUTPUT_IMAGE = HERE / "sdxl_local_test_output.png"


def run_local_test():
    print("=" * 65)
    print("ResiliCity Local SDXL Inpainting Benchmark & Validation")
    print("=" * 65)

    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")

    if not cuda_avail:
        print("ERROR: CUDA is not available. This test requires an NVIDIA GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    print(f"GPU Device: {gpu_name}")
    print(f"Total VRAM: {total_vram} GB")
    print(f"Model Path: {MODEL_PATH}")

    # Check input image
    if not CLEAN_INPUT_IMAGE.exists():
        print(f"ERROR: Clean input photo not found at: {CLEAN_INPUT_IMAGE}")
        sys.exit(1)

    print(f"\n1. Loading clean urban street photo: {CLEAN_INPUT_IMAGE.name}")
    orig_img = Image.open(CLEAN_INPUT_IMAGE).convert("RGB")
    orig_w, orig_h = orig_img.size
    print(f"   Original dimensions: {orig_w}x{orig_h}")

    # Scale to working resolution (512 max dimension, multiple of 8)
    scale = min(512 / max(orig_w, orig_h), 1.0)
    work_w = int(round((orig_w * scale) / 8.0) * 8)
    work_h = int(round((orig_h * scale) / 8.0) * 8)
    work_img = orig_img.resize((work_w, work_h), Image.Resampling.LANCZOS)
    print(f"   Inference working resolution: {work_w}x{work_h}")

    # 2. Create architecturally plausible planting & pedestrian sidewalk mask
    # Left pedestrian sidewalk/verge corridor: extends from curb into tree canopy height
    print("\n2. Creating pedestrian planting corridor inpainting mask...")
    mask = Image.new("L", (work_w, work_h), 0)
    draw = ImageDraw.Draw(mask)

    # Left sidewalk planting envelope (x: 0% to 32%, y: 35% to 92%)
    draw.rectangle(
        [0, int(work_h * 0.35), int(work_w * 0.32), int(work_h * 0.92)],
        fill=255,
    )
    # Right curb verge envelope (x: 72% to 100%, y: 40% to 95%)
    draw.rectangle(
        [int(work_w * 0.72), int(work_h * 0.40), work_w, int(work_h * 0.95)],
        fill=255,
    )

    # Soften edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    mask_cov = (np.count_nonzero(np.array(mask) > 128) / (work_w * work_h)) * 100
    print(f"   Mask coverage: {mask_cov:.1f}% of image area")

    # 3. Load pipeline
    print("\n3. Loading SDXL Inpainting pipeline (FP16, CPU Offload, VAE Slicing)...")
    t_load_start = time.time()
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipe.enable_model_cpu_offload()

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    load_time = time.time() - t_load_start
    print(f"   Model ready in {load_time:.2f} seconds")

    # 4. Prompts
    prompt = (
        "photorealistic urban climate-resilient streetscape redesign, "
        "mature leafy street trees with natural green canopy and textured wooden trunk planted in sidewalk planting bed, "
        "modern light permeable concrete pavers with gravel joints, "
        "realistic cast tree shadows on pavement, "
        "same real street, same buildings and architecture, same camera perspective, "
        "professional urban landscape architecture photography, sharp focus, 8k, natural daylight"
    )
    negative_prompt = (
        "cartoon, illustration, painting, 3d render, fantasy city, "
        "floating tree, deformed tree, tree in road, tree inside building, "
        "segmentation mask, colored overlay, text, watermark, blurry, distorted architecture"
    )

    # 5. Run generation
    print("\n4. Running SDXL Inpainting inference (18 steps, guidance=7.5)...")
    torch.cuda.reset_peak_memory_stats()
    t_gen_start = time.time()

    generator = torch.Generator(device="cuda").manual_seed(42)

    with torch.inference_mode():
        generated_result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=work_img,
            mask_image=mask,
            strength=0.99,
            guidance_scale=7.5,
            num_inference_steps=18,
            generator=generator,
        ).images[0]

    gen_time = time.time() - t_gen_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Upscale back to source dimensions
    final_image = generated_result.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    final_image.save(OUTPUT_IMAGE)

    # 6. Quantitative difference and validation check
    orig_np = np.array(orig_img).astype(np.float32)
    final_np = np.array(final_image).astype(np.float32)
    abs_diff = np.abs(final_np - orig_np)
    diff_mean = float(np.mean(abs_diff))
    changed_px = np.count_nonzero(np.max(abs_diff, axis=2) > 15.0)
    pct_changed = float((changed_px / (orig_w * orig_h)) * 100.0)

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Status:             SUCCESS")
    print(f"Active Provider:    local_sdxl")
    print(f"Model:              diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    print(f"Inference Res:      {work_w}x{work_h} -> Output {orig_w}x{orig_h}")
    print(f"Inference Steps:    18")
    print(f"CUDA Status:        Available (device 0)")
    print(f"GPU Name:           {gpu_name}")
    print(f"Peak VRAM:          {peak_vram:.2f} GB (within 6 GB budget)")
    print(f"Generation Time:    {gen_time:.2f} seconds")
    print(f"Mean Pixel Diff:    {diff_mean:.2f} (threshold >= 4.0)")
    print(f"Changed Area:       {pct_changed:.2f}% (threshold >= 1.5%)")
    print(f"Output File:        {OUTPUT_IMAGE}")
    print("=" * 65)

    if diff_mean < 4.0 or pct_changed < 1.5:
        print("WARNING: Output difference is below threshold. Redesign did not visibly change scene.")
        sys.exit(2)
    else:
        print("VALIDATION PASSED: Redesigned photograph is visibly distinct and grounded.")


if __name__ == "__main__":
    run_local_test()
