"""ADE20K to ResiliCity taxonomy mapping and surface metadata."""

from typing import Dict, Tuple

# ResiliCity target classes
RESILICITY_CLASSES = [
    "road",
    "pavement",
    "vegetation",
    "wall",
    "roof",
    "water",
    "sky",
    "other",
]

# (label, default_albedo, default_emissivity, color)
CLASS_METADATA: Dict[str, Tuple[str, float, float, str]] = {
    "road": ("Asphalt / road", 0.08, 0.94, "#e11d48"),
    "wall": ("Concrete / walls", 0.22, 0.88, "#d97706"),
    "roof": ("Roof surfaces", 0.14, 0.90, "#eab308"),
    "vegetation": ("Vegetation", 0.25, 0.97, "#059669"),
    "pavement": ("Pavement", 0.25, 0.95, "#0d9488"),
    "water": ("Water", 0.10, 0.96, "#0284c7"),
    "sky": ("Sky", 0.50, 0.90, "#38bdf8"),
    "other": ("Other / unclassified", 0.20, 0.90, "#94a3b8"),
}

# Explicit mapping from ADE20K index to ResiliCity class.
# Inspected against nvidia/segformer-b0-finetuned-ade-512-512 config.id2label.
#
# NOTE ON ROOFS: In ADE20K street photos, rooftops are viewed from ground level and typically
# absorbed into building/house/wall. Only awnings (86) and canopies (106) map to roof.
# True distinct rooftop segmentation requires the future locally fine-tuned SegFormer model (Part B).
ADE20K_TO_RESILICITY: Dict[int, str] = {
    # Road / traffic surfaces
    6: "road",       # road
    54: "road",      # runway
    91: "road",      # dirt track
    20: "road",      # car (on road surface)
    80: "road",      # bus (on road surface)
    83: "road",      # truck (on road surface)
    102: "road",     # van (on road surface)
    
    # Sidewalk / ground / walking surfaces
    3: "pavement",   # floor / paved ground
    11: "pavement",  # sidewalk
    13: "pavement",  # earth / bare soil
    46: "pavement",  # sand
    52: "pavement",  # path
    53: "pavement",  # stairs
    59: "pavement",  # stairway
    94: "pavement",  # land / ground
    121: "pavement", # step
    140: "pavement", # pier
    
    # Vegetation
    4: "vegetation",   # tree
    9: "vegetation",   # grass
    17: "vegetation",  # plant
    29: "vegetation",  # field
    66: "vegetation",  # flower
    72: "vegetation",  # palm
    
    # Walls / Built structures
    0: "wall",       # wall
    1: "wall",       # building
    8: "wall",       # windowpane
    14: "wall",      # door
    25: "wall",      # house
    32: "wall",      # fence
    38: "wall",      # railing
    42: "wall",      # column
    48: "wall",      # skyscraper
    51: "wall",      # grandstand
    79: "wall",      # hovel
    84: "wall",      # tower
    88: "wall",      # booth
    95: "wall",      # bannister
    
    # Roofs (documented approximation for ADE20K)
    86: "roof",      # awning
    106: "roof",     # canopy
    
    # Water
    21: "water",     # water
    26: "water",     # sea
    60: "water",     # river
    109: "water",    # swimming pool
    113: "water",    # waterfall
    128: "water",    # lake
    
    # Sky
    2: "sky",        # sky
}


def build_label_map(id2label: Dict[int, str]) -> Dict[int, str]:
    """Build a mapping from model prediction IDs to ResiliCity target classes.
    
    If the model already uses ResiliCity classes (e.g., future Part B fine-tuned model),
    it matches them directly. If the model uses ADE20K classes, it applies ADE20K_TO_RESILICITY.
    """
    label_map = {}
    is_resilicity_native = any(
        label.lower() in RESILICITY_CLASSES for label in id2label.values()
    )
    
    for idx, label_name in id2label.items():
        clean_name = label_name.strip().lower()
        if is_resilicity_native and clean_name in RESILICITY_CLASSES:
            label_map[idx] = clean_name
        elif idx in ADE20K_TO_RESILICITY:
            label_map[idx] = ADE20K_TO_RESILICITY[idx]
        else:
            label_map[idx] = "other"
            
    return label_map
