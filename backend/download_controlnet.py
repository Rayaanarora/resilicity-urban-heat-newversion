import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

TARGET_DIR = Path(r"D:\huggingface_cache\controlnet-depth-sd15")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

repo_id = "lllyasviel/control_v11f1p_sd15_depth"

print(f"Downloading ControlNet Depth config to {TARGET_DIR}...")
config_path = hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=str(TARGET_DIR))
print(f"Config downloaded: {config_path}")

print(f"Downloading ControlNet Depth FP16 weights (~1.45 GB)...")
weights_path = hf_hub_download(
    repo_id=repo_id,
    filename="diffusion_pytorch_model.fp16.safetensors",
    local_dir=str(TARGET_DIR)
)
print(f"Weights downloaded: {weights_path}")

# Also copy or link as diffusion_pytorch_model.safetensors for universal compatibility
universal_path = TARGET_DIR / "diffusion_pytorch_model.safetensors"
if not universal_path.exists():
    shutil.copyfile(weights_path, universal_path)
    print(f"Created universal alias: {universal_path}")

print("ControlNet Depth download complete!")
