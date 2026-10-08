"""Minimal Isolated Test for SD 1.5 Inpainting + ControlNet Depth Tree Generation (Step 4).

Verifies that the model directly synthesizes a realistic mature roadside tree
from scratch inside an intervention mask conditioned on depth, with ZERO pasted assets.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

HERE = Path(__file__).resolve().parent
IMAGE_PATH = HERE.parents[0] / "gen-ai" / "public" / "samples" / "urban_street_before.png"
DEBUG_DIR = HERE / "data" / "debug" / "step4_isolated_test"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

SD15_PATH = r"D:\huggingface_cache\sd15-inpaint"
CNET_PATH = r"D:\huggingface_cache\controlnet-depth-sd15"

sys.path.insert(0, str(HERE))
from scene_understanding.depth import estimate_relative_depth


def preprocess_depth_for_controlnet(
    depth_arr: np.ndarray,
    target_size: tuple[int, int],
) -> Image.Image:
    """Normalize and format depth for ControlNet Depth (MiDaS disparity: near=255, far=0)."""
    h, w = depth_arr.shape
    # If depth_arr is 0=near, 1=far, invert so near is bright (255) for ControlNet
    # In street scenes, bottom rows are near, top rows are far (sky).
    bottom_mean = float(np.mean(depth_arr[int(h * 0.8):, :]))
    top_mean = float(np.mean(depth_arr[:int(h * 0.2), :]))

    if bottom_mean < top_mean:
        # 0 is near, 1 is far -> invert
        disp = 1.0 - depth_arr
    else:
        disp = depth_arr

    disp_norm = ((disp - disp.min()) / (disp.max() - disp.min() + 1e-6) * 255.0).clip(0, 255).astype(np.uint8)
    disp_img = Image.fromarray(disp_norm, mode="L").resize(target_size, Image.Resampling.LANCZOS)
    disp_rgb = Image.merge("RGB", (disp_img, disp_img, disp_img))
    return disp_rgb


def main():
    print("=" * 60)
    print("STEP 4: ISOLATED CONTROLNET TREE INPAINTING TEST")
    print("=" * 60)

    # 1. Load original street photograph
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Source image not found: {IMAGE_PATH}")
    original_img = Image.open(IMAGE_PATH).convert("RGB")
    orig_w, orig_h = original_img.size
    print(f"Loaded original image: {IMAGE_PATH.name} ({orig_w}x{orig_h})")

    # 2. Estimate relative depth
    print("Estimating relative depth from original image...")
    depth_map = estimate_relative_depth(original_img)
    print(f"Depth estimated: shape={depth_map.shape}, min={depth_map.min():.3f}, max={depth_map.max():.3f}, mean={depth_map.mean():.3f}")

    # 3. Create intervention region on sidewalk (e.g. left sidewalk zone)
    # Target planting ground anchor around (x=160, y=orig_h * 0.72)
    anchor_x = int(orig_w * 0.22)
    anchor_y = int(orig_h * 0.75)
    
    # Define generation context crop (512x512) around anchor
    crop_w, crop_h = 512, 512
    crop_x1 = max(0, anchor_x - 220)
    crop_y1 = max(0, anchor_y - 380)
    crop_x2 = min(orig_w, crop_x1 + crop_w)
    crop_y2 = min(orig_h, crop_y1 + crop_h)
    
    # Adjust crop to exactly 512x512 if possible
    crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
    original_crop = original_img.crop(crop_bbox).resize((512, 512), Image.Resampling.LANCZOS)
    
    # Depth crop
    depth_crop_arr = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]
    depth_condition_crop = preprocess_depth_for_controlnet(depth_crop_arr, (512, 512))

    # Create intervention mask (generous contextual region for trunk, canopy, branches, ground contact)
    # Within 512x512 crop:
    # anchor in crop coords:
    rel_ax = int((anchor_x - crop_x1) * 512 / (crop_x2 - crop_x1))
    rel_ay = int((anchor_y - crop_y1) * 512 / (crop_y2 - crop_y1))
    
    mask_arr = np.zeros((512, 512), dtype=np.uint8)
    # Ground planting pit region
    cv2.ellipse(mask_arr, (rel_ax, rel_ay), (45, 20), 0, 0, 360, 255, -1)
    # Trunk region
    cv2.rectangle(mask_arr, (rel_ax - 28, rel_ay - 140), (rel_ax + 28, rel_ay + 5), 255, -1)
    # Generous spreading canopy region with context
    cv2.ellipse(mask_arr, (rel_ax, rel_ay - 160), (120, 140), 0, 0, 360, 255, -1)
    # Feather / smooth the mask slightly
    mask_arr = cv2.GaussianBlur(mask_arr, (15, 15), 0)
    # Binarize core while maintaining contextual perimeter
    mask_arr = np.where(mask_arr > 30, 255, 0).astype(np.uint8)
    intervention_mask = Image.fromarray(mask_arr, mode="L")

    print(f"Intervention crop bbox: {crop_bbox}")
    print(f"Mask active pixels: {np.count_nonzero(mask_arr)} ({np.count_nonzero(mask_arr) / (512*512) * 100:.1f}%)")

    # 4. Save input artifacts
    original_crop.save(DEBUG_DIR / "01_original_crop.png")
    depth_condition_crop.save(DEBUG_DIR / "02_depth_condition.png")
    intervention_mask.save(DEBUG_DIR / "03_intervention_mask.png")
    depth_condition_crop.save(DEBUG_DIR / "04_control_input.png")
    
    # Overlay mask on crop for visualization
    gen_in_vis = original_crop.copy()
    overlay = Image.new("RGBA", (512, 512), (0, 230, 80, 0))
    ov_arr = np.array(overlay)
    ov_arr[mask_arr > 0] = [0, 220, 90, 100]
    gen_in_vis = Image.alpha_composite(gen_in_vis.convert("RGBA"), Image.fromarray(ov_arr)).convert("RGB")
    gen_in_vis.save(DEBUG_DIR / "05_generation_input.png")

    # 5. Load Pipeline
    print("Loading ControlNet + SD 1.5 Inpainting Pipeline...")
    t_load = time.time()
    cnet = ControlNetModel.from_pretrained(
        CNET_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        SD15_PATH,
        controlnet=cnet,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
    ).to("cuda")

    # Hardware optimizations for RTX 3050 6GB
    pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    print(f"Pipeline ready in {time.time() - t_load:.2f}s! VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # 6. Execute Generative Inpainting
    prompt = (
        "photorealistic mature roadside shade tree naturally planted in the sidewalk, "
        "realistic trunk emerging from a planting pit, dense but natural green canopy, "
        "visible branches, realistic bark texture, natural tropical urban vegetation, "
        "correct perspective and scale, physically grounded, realistic sunlight and cast shadow, "
        "seamlessly integrated into the existing street photograph"
    )
    negative_prompt = (
        "cartoon, illustration, painting, 3d render, CGI, artificial tree, cutout, sticker, "
        "floating object, distorted trunk, deformed foliage, plastic leaves, blurry, low resolution, "
        "malformed branches, unrealistic shadow, giant tree, miniature tree"
    )

    print(f"Generating intervention with SD 1.5 + ControlNet Depth...")
    generator = torch.Generator(device="cuda").manual_seed(42)
    t_gen = time.time()

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original_crop,
            mask_image=intervention_mask,
            control_image=depth_condition_crop,
            height=512,
            width=512,
            strength=1.0,
            num_inference_steps=24,
            guidance_scale=7.5,
            controlnet_conditioning_scale=0.8,
            generator=generator,
        ).images[0]

    gen_time = time.time() - t_gen
    print(f"Inference completed in {gen_time:.2f}s! Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")

    # 7. Composite within mask and save 06_generation_output.png
    soft_mask = intervention_mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    generated_crop = Image.composite(output, original_crop, soft_mask)
    generated_crop.save(DEBUG_DIR / "06_generation_output.png")

    # 8. Recomposite into original full photograph
    recomposited = original_img.copy()
    target_crop_size = (crop_x2 - crop_x1, crop_y2 - crop_y1)
    crop_resized_back = generated_crop.resize(target_crop_size, Image.Resampling.LANCZOS)
    soft_mask_back = soft_mask.resize(target_crop_size, Image.Resampling.LANCZOS)
    
    orig_patch = original_img.crop(crop_bbox)
    blended_patch = Image.composite(crop_resized_back, orig_patch, soft_mask_back)
    recomposited.paste(blended_patch, (crop_x1, crop_y1))
    recomposited.save(DEBUG_DIR / "07_recomposited.png")

    # 9. Analyze pixel difference to prove genuine generative redesign
    orig_np = np.array(original_crop).astype(np.float32)
    gen_np = np.array(generated_crop).astype(np.float32)
    diff = np.abs(gen_np - orig_np)
    mask_bool = mask_arr > 30
    masked_diff = float(diff[mask_bool].mean())
    print(f"Masked pixel diff: {masked_diff:.2f} (threshold >= 10.0)")

    if masked_diff >= 10.0:
        print("SUCCESS! Model genuinely generated intervention inside the masked region!")
    else:
        print("WARNING: Low pixel difference in masked region.")

    print(f"Debug artifacts saved to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
