import os
import time
import torch
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForInpainting

MODEL_PATH = r"D:\huggingface_cache\sdxl-inpainting"

INPUT_IMAGE = r"C:\Users\Rayaan\Desktop\Resillicity\gen-ai\public\samples\sample_1_dense_urban.jpg"
OUTPUT_IMAGE = r"C:\Users\Rayaan\Desktop\Resillicity\backend\sdxl_test_output.png"

print("=" * 60)
print("ResiliCity Local SDXL Test")
print("=" * 60)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2), "GB")

print("\nLoading SDXL Inpainting model...")
start_load = time.time()

pipe = AutoPipelineForInpainting.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=True,
)

# Important for 6 GB VRAM.
pipe.enable_model_cpu_offload()

# Additional memory optimizations.
try:
    pipe.enable_vae_slicing()
except Exception:
    pass

try:
    pipe.enable_vae_tiling()
except Exception:
    pass

print("Model loaded in", round(time.time() - start_load, 2), "seconds")

print("\nLoading test image...")
image = Image.open(INPUT_IMAGE).convert("RGB")

# Small test resolution for the RTX 3050 6GB.
image = image.resize((512, 512), Image.Resampling.LANCZOS)

# Create a test mask.
# White = area to regenerate.
# Black = preserve original.
mask = Image.new("L", (512, 512), 0)
draw = ImageDraw.Draw(mask)

# Lower-left area: pretend this is an available planting zone.
draw.rectangle(
    [50, 300, 260, 510],
    fill=255,
)

prompt = """
photorealistic urban climate-resilient streetscape redesign,
large mature leafy shade tree planted naturally in the sidewalk planting zone,
realistic textured tree trunk,
dense natural green canopy,
small landscaped planting bed around the tree,
realistic cast shadows on the pavement,
professional urban landscape architecture,
same real street,
same buildings,
same road,
same camera perspective,
same lighting,
physically plausible scale,
highly realistic architectural photography
"""

negative_prompt = """
cartoon, illustration, painting, 3d render, fantasy city,
floating tree, deformed tree, duplicated objects,
tree in road, tree inside building, unrealistic jungle,
segmentation mask, colored overlay, text, watermark,
blurry, low resolution, distorted architecture
"""

print("\nGenerating image...")
start_gen = time.time()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

generator = torch.Generator(device="cuda").manual_seed(42)

result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=image,
    mask_image=mask,
    strength=0.99,
    guidance_scale=8.0,
    num_inference_steps=15,
    generator=generator,
).images[0]

generation_time = time.time() - start_gen

result.save(OUTPUT_IMAGE)

print("\n" + "=" * 60)
print("GENERATION SUCCESSFUL")
print("=" * 60)
print("Output:", OUTPUT_IMAGE)
print("Generation time:", round(generation_time, 2), "seconds")

if torch.cuda.is_available():
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print("Peak CUDA memory:", round(peak_vram, 2), "GB")

print("=" * 60)
print("Open the output image and check whether the tree was actually generated.")
print("=" * 60)