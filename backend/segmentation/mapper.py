"""ADE20K to ResiliCity taxonomy mapping, surface metadata, and protected object separation.

Maintains two distinct conceptual layers:
1. SURFACE CLASSES (road, pavement, vegetation, wall, roof, water, sky, other)
2. PROTECTED OBJECT CLASSES (vehicles, pedestrians, street infrastructure, architectural details)

Protected object masks are never mapped into editable road or pavement regions,
preventing the inpainting generator from erasing or mutating cars, pedestrians, signs, or facades.
"""

from typing import Dict, List, Set, Tuple

# Layer 1: ResiliCity primary surface classes
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

# Layer 2: Protected object classes (must be excluded from editable masks)
PROTECTED_OBJECT_CLASSES = [
    "vehicle",          # car, bus, truck, van, motorcycle, bicycle
    "person",           # pedestrians, cyclists
    "street_furniture", # bench, trash can, fountain
    "infrastructure",   # traffic light, streetlight/lamp, pole, sign
    "facade_detail",    # doors, windowpanes, columns, railings, fences, banisters
    "awning_canopy",    # storefront awnings, canopies (not true rooftops)
]

# (label, default_albedo, default_emissivity, color)
CLASS_METADATA: Dict[str, Tuple[str, float, float, str]] = {
    "road": ("Asphalt / roadway", 0.08, 0.94, "#e11d48"),
    "pavement": ("Pedestrian pavement / sidewalk", 0.25, 0.95, "#0d9488"),
    "vegetation": ("Vegetation / canopy", 0.25, 0.97, "#059669"),
    "wall": ("Building facade / walls", 0.22, 0.88, "#d97706"),
    "roof": ("Rooftop surfaces", 0.14, 0.90, "#eab308"),
    "water": ("Water", 0.10, 0.96, "#0284c7"),
    "sky": ("Sky", 0.50, 0.90, "#38bdf8"),
    "other": ("Other / unclassified", 0.20, 0.90, "#94a3b8"),
    # Protected object metadata for inspection/overlay
    "vehicle": ("Vehicles (protected)", 0.15, 0.92, "#6366f1"),
    "person": ("Pedestrians (protected)", 0.20, 0.95, "#ec4899"),
    "infrastructure": ("Traffic & lighting (protected)", 0.20, 0.90, "#8b5cf6"),
    "facade_detail": ("Facade details (protected)", 0.25, 0.88, "#f59e0b"),
}

# Fine-grained ADE20K semantic categorization
# Explicitly inspected against nvidia/segformer-b0-finetuned-ade-512-512 config.id2label

# 1. Clean Surface Mapping:
ADE20K_SURFACES: Dict[int, str] = {
    # Roadway only (VEHICLES EXCLUDED)
    6: "road",       # road, route
    54: "road",      # runway
    91: "road",      # dirt track

    # Pedestrian walking surfaces (earth, sand, stairs EXCLUDED from editable sidewalk paver mask)
    3: "pavement",   # floor / paved ground
    11: "pavement",  # sidewalk
    52: "pavement",  # path

    # Vegetation
    4: "vegetation",   # tree
    9: "vegetation",   # grass
    17: "vegetation",  # plant
    29: "vegetation",  # field
    66: "vegetation",  # flower
    72: "vegetation",  # palm

    # Building facades & structural walls
    0: "wall",       # wall
    1: "wall",       # building
    25: "wall",      # house
    48: "wall",      # skyscraper
    79: "wall",      # hovel
    84: "wall",      # tower
    88: "wall",      # booth

    # True Rooftops (Note: In ADE20K street photos, rooftops are rarely visible from ground level.
    # Awnings (86) and canopies (106) are storefront structures and NOT actual building roofs.)
    # True rooftop segmentation is preserved when native roof class exists or verified flat top.

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

# 2. Protected Objects (MUST NOT be edited or overwritten by diffusion inpainting)
ADE20K_PROTECTED_OBJECTS: Dict[int, str] = {
    # Vehicles (Do NOT map to road!)
    20: "vehicle",       # car
    80: "vehicle",       # bus
    83: "vehicle",       # truck
    102: "vehicle",      # van
    116: "vehicle",      # motorcycle
    127: "vehicle",      # bicycle
    76: "vehicle",       # boat

    # People / Pedestrians
    12: "person",        # person / pedestrian

    # Traffic & Municipal Infrastructure
    43: "infrastructure",  # signboard / sign
    87: "infrastructure",  # streetlight / street lamp
    93: "infrastructure",  # utility pole
    136: "infrastructure", # traffic light / signal
    36: "infrastructure",  # lamp
    82: "infrastructure",  # light source

    # Street furniture & amenities
    69: "street_furniture",  # bench
    138: "street_furniture", # trash can / wastebin
    104: "street_furniture", # fountain

    # Architectural facade details & barriers (Do not paint over windows/doors/railings)
    8: "facade_detail",      # windowpane
    14: "facade_detail",     # door
    32: "facade_detail",     # fence
    38: "facade_detail",     # railing
    42: "facade_detail",     # column / pillar
    95: "facade_detail",     # bannister / balustrade
    51: "facade_detail",     # grandstand

    # Storefront awnings & canopies (not true rooftops)
    86: "awning_canopy",     # awning
    106: "awning_canopy",    # canopy
}

# Unpaved/other ground surfaces that should NOT be edited as paved sidewalk
ADE20K_UNPAVED_GROUND: Set[int] = {
    13,  # earth / bare soil
    46,  # sand
    53,  # stairs
    59,  # stairway
    94,  # land / ground
    121, # step
    140, # pier
}


def build_label_map(id2label: Dict[int, str]) -> Dict[int, str]:
    """Build a mapping from model prediction IDs to ResiliCity target classes.
    
    Preserves clean separation between surface classes and protected objects.
    Vehicles, pedestrians, and infrastructure are mapped to 'other' in the surface taxonomy,
    preventing contamination of road and pavement surface masks.
    """
    label_map = {}
    is_resilicity_native = any(
        label.lower() in RESILICITY_CLASSES for label in id2label.values()
    )

    for idx, label_name in id2label.items():
        clean_name = label_name.strip().lower()
        if is_resilicity_native and clean_name in RESILICITY_CLASSES:
            label_map[idx] = clean_name
        elif idx in ADE20K_SURFACES:
            label_map[idx] = ADE20K_SURFACES[idx]
        elif idx in ADE20K_PROTECTED_OBJECTS:
            # Categorized as other in surface mix to avoid contaminating road/pavement
            label_map[idx] = "other"
        else:
            label_map[idx] = "other"

    return label_map


def build_protected_object_map(id2label: Dict[int, str]) -> Dict[int, str]:
    """Build a mapping from model prediction IDs to protected object classes."""
    prot_map = {}
    for idx in id2label.keys():
        if idx in ADE20K_PROTECTED_OBJECTS:
            prot_map[idx] = ADE20K_PROTECTED_OBJECTS[idx]
    return prot_map
