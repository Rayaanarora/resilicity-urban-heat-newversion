"""Unit tests for Stage 4 procedural inpainting engine and endpoint."""

import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inpainting import ResilientInpainter, encode_image_to_base64


def test_inpainter_generates_image():
    inpainter = ResilientInpainter()
    img = Image.new("RGB", (200, 200), (100, 100, 100))
    polygons = {
        "roof": [[[10, 10], [190, 10], [100, 50]]],
        "road": [[[0, 150], [200, 150], [200, 200], [0, 200]]],
        "pavement": [[[0, 130], [50, 130], [50, 150], [0, 150]]],
    }
    interventions = [
        {"type": "cool_roof", "target_region": "roof", "coverage": 0.8},
        {"type": "cool_pavement", "target_region": "road", "coverage": 0.5},
        {"type": "tree_canopy", "target_region": "pavement", "coverage": 0.6},
    ]

    out = inpainter.inpaint(img, polygons, interventions)
    assert isinstance(out, Image.Image)
    assert out.size == (200, 200)

    # Base64 encoding check
    data_url = encode_image_to_base64(out)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_inpainter_graceful_with_empty_inputs():
    inpainter = ResilientInpainter()
    img = Image.new("RGB", (100, 100), (50, 50, 50))
    out = inpainter.inpaint(img, {}, [])
    assert out.size == (100, 100)
