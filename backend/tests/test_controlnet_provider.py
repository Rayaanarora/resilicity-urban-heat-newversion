"""Unit and regression tests for LocalSD15ControlNetInpaintingProvider and depth conditioning."""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_generation import (
    LocalSD15ControlNetInpaintingProvider,
    preprocess_depth_for_controlnet,
    build_mask_from_intervention_geometry,
    build_intervention_prompt,
    SpatialInterventionSpec,
)


def test_depth_preprocessing_normalization():
    """Verify depth array is properly normalized, inverted for ControlNet, and formatted as RGB."""
    # Create gradient where top is far (1.0) and bottom is near (0.0)
    raw_depth = np.linspace(1.0, 0.0, 100)[:, None]
    raw_depth = np.repeat(raw_depth, 100, axis=1)

    disp_img = preprocess_depth_for_controlnet(raw_depth, target_size=(512, 512))
    assert isinstance(disp_img, Image.Image)
    assert disp_img.size == (512, 512)
    assert disp_img.mode == "RGB"

    disp_arr = np.array(disp_img)
    # Channels must be identical for grayscale depth in RGB
    assert np.array_equal(disp_arr[:, :, 0], disp_arr[:, :, 1])
    assert np.array_equal(disp_arr[:, :, 1], disp_arr[:, :, 2])

    # Bottom (near foreground) must be brighter than top (far sky) for ControlNet
    bottom_mean = np.mean(disp_arr[75:, :, 0])
    top_mean = np.mean(disp_arr[:25, :, 0])
    assert bottom_mean > top_mean


def test_intervention_prompt_builder():
    """Verify design-specific prompts for trees, shade, and pavement."""
    tree_p, tree_neg = build_intervention_prompt("tree_canopy")
    assert "roadside shade tree" in tree_p
    assert "canopy" in tree_p
    assert "cartoon" in tree_neg

    shade_p, shade_neg = build_intervention_prompt("shade_structure")
    assert "shade canopy" in shade_p
    assert "supports" in shade_p
    assert "flat colored polygon" in shade_neg

    pave_p, pave_neg = build_intervention_prompt("cool_pavement")
    assert "solar-reflective" in pave_p
    assert "lane markings" in pave_p


def test_mask_from_tree_geometry():
    """Verify mask builder creates contextual regions from tree anchors."""
    iv = SpatialInterventionSpec(
        type="tree_canopy",
        target_region="sidewalk",
        target_zone="left_sidewalk",
        placement="sidewalk curb margin",
        visual_design="mature shade trees",
        reason="high heat priority",
        anchors=[
            {"x": 150, "y": 400, "canopy_radius": 50, "canopy_center_y": 320, "scale": 0.8}
        ],
    )

    mask = build_mask_from_intervention_geometry(iv, width=600, height=600)
    assert isinstance(mask, Image.Image)
    assert mask.mode == "L"

    mask_arr = np.array(mask)
    # Active pixels should exist around anchor (150, 400) and canopy (150, 320)
    assert mask_arr[400, 150] == 255  # Ground pit
    assert mask_arr[320, 150] == 255  # Canopy
    assert mask_arr[50, 50] == 0      # Unrelated corner


def test_controlnet_provider_singleton_and_status():
    """Verify provider singleton pattern and initial status reporting."""
    provider1 = LocalSD15ControlNetInpaintingProvider.get_instance()
    provider2 = LocalSD15ControlNetInpaintingProvider.get_instance()
    assert provider1 is provider2

    status = provider1.get_status()
    assert status["provider"] == "LocalSD15ControlNetInpaintingProvider"
    assert status["model"] == "stable-diffusion-v1-5/stable-diffusion-inpainting"
    assert status["controlnet_model"] == "lllyasviel/control_v11f1p_sd15_depth"
    assert "cuda_available" in status
