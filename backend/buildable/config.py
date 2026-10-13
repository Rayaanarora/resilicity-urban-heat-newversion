"""Buildable-design configuration: every threshold, dimension, cost and coefficient lives here.

INDICATIVE VALUES ONLY: replace with the local Schedule of Rates (e.g. CPWD DSR / state SoR)
and the applicable street design guidelines (e.g. IRC:103 pedestrian facilities, IRC SP:117,
the city's street design manual) before using any output for estimates, tenders or approvals.

Nothing outside this file may hardcode a cost, clearance, spacing or cooling number.
"""

CONFIG_LABEL = "indicative, replace with local Schedule of Rates / street design guidelines"

# ---------------------------------------------------------------------------
# 1. Metric estimation (image -> metres)
# ---------------------------------------------------------------------------
METRICS = {
    "default_road_width_m": 7.0,        # carriageway width used as the scale reference when none is given
    "uncertainty_pct": 30,              # +/- band reported on every metric estimate
    "truncated_uncertainty_pct": 50,    # band when the road runs off the frame (scale is biased)
    "row_band": (0.50, 0.97),           # fraction of image height sampled (near-field ground rows)
    "min_road_span_frac": 0.05,         # ignore rows where the road spans < 5% of image width
    "min_side_px": 2,                   # minimum sidewalk pixels in a row to count it
    "min_valid_rows": 5,                # fewer valid rows -> estimate unavailable
    "min_plausible_sidewalk_m": 0.5,    # narrower medians are segmentation specks, reported as not measurable
    "conflict_strip_frac": 0.02,        # kerb-strip dilation around each sidewalk (fraction of width)
    "junction_band": (0.45, 0.80),      # rows inspected for side-street openings
    "junction_min_row_frac": 0.25,      # share of those rows that must show an opening
}

# ---------------------------------------------------------------------------
# 2. Clearances and geometry rules used by gating
# ---------------------------------------------------------------------------
CLEARANCES = {
    "min_clear_walkway_m": 1.8,            # clear pedestrian path that must remain after any fixture
    "kerb_buffer_m": 0.3,                  # buffer between kerb face and pit / post edge
    "min_vertical_clearance_m": 2.4,       # headroom under awnings and canopies
    "junction_sight_clearance_m": 15.0,    # no trees within this distance of a junction (sight triangle)
    "max_pole_strip_frac": 0.04,           # pole/lamp/signal pixels in the kerb strip above this -> no pits
    "max_vehicle_strip_frac": 0.15,        # vehicles parked on/along the sidewalk strip above this -> no pits
    "max_person_strip_frac": 0.10,         # dense pedestrian/vendor use above this -> no pits
    "parking_lane_width_m": 2.0,           # width of a kerbside parking lane / build-out
    "min_carriageway_after_buildout_m": 6.0,  # carriageway that must remain after a kerb build-out
    "deep_canyon_hw_proxy_min": 0.7,       # street-canyon H/W proxy above which roadway cool pavement is allowed
    "min_wall_density_for_cool_wall": 0.15,
    "min_wall_density_for_awning": 0.10,
    "min_solar_exposure_for_pv": 0.45,
    "max_canyon_strength_for_pv": 0.60,
    "min_road_width_for_bus_shelter_m": 7.0,
    "humid_climates": ["tropical", "tropical_humid", "tropical_monsoon", "humid_subtropical"],
    "default_climate_zone": "tropical_humid",
    "street_length_m": 100.0,              # implementation plan quantities are per this length
    "default_sidewalk_m_if_unmeasured": 2.0,  # only used for quantities, always flagged
}

# ---------------------------------------------------------------------------
# 3. Buildable catalog
#    cost_inr: (low, high) per unit. cooling_c: (low, central, high) indicative localized
#    cooling at full coverage; NOT air temperature for the whole street.
#    phase: "quick_win" (0-3 months), "medium" (3-12 months), "long_term" (12+ months).
# ---------------------------------------------------------------------------
CATALOG = {
    "tree_canopy": {
        "title": "Young Street Trees in Tree Pits",
        "zone_kind": "sidewalk",
        "dimensions": {"pit_m": 1.2, "planting_height_m": 4.0, "girth_mm": 60},
        "spacing_m": 8.0,
        "min_clear_width_m": 1.8,
        "unit": "tree",
        "cost_inr": (25_000, 60_000),
        "lifespan_years": 40,
        "maintenance": "Watering and staking for 3 years, formative pruning every 2 years, grate cleaning.",
        "permits": ["Horticulture / parks department approval", "Underground utility clearance (trial pits)",
                    "Road-cutting permission from the road-owning agency"],
        "phase": "medium",
        "cooling_c": (0.5, 1.2, 2.0),
        "kontext": ("plant young street trees about 4 m tall with slim trunks and modest canopies in 1.2 m "
                    "square tree pits with metal grates, spaced about {spacing_m:.0f} m apart at the kerb of {zones}, "
                    "leaving the walkway clear"),
    },
    "parking_lane_tree": {
        "title": "Parking-Lane Trees in Kerb Build-Outs",
        "zone_kind": "kerb",
        "dimensions": {"buildout_m": (2.0, 3.0), "planting_height_m": 4.0},
        "spacing_m": 10.0,
        "min_clear_width_m": 0.0,
        "unit": "tree + build-out",
        "cost_inr": (80_000, 180_000),
        "lifespan_years": 40,
        "maintenance": "Watering for 3 years, kerb inlet and gully cleaning before monsoon.",
        "permits": ["Traffic police NOC for parking removal", "Road-owning agency approval for kerb realignment",
                    "Stormwater drainage check at each build-out"],
        "phase": "medium",
        "cooling_c": (0.4, 1.0, 1.7),
        "kontext": ("add young street trees in small raised kerb build-outs within the parking lane along {zones}, "
                    "spaced about {spacing_m:.0f} m apart, with painted kerbs"),
    },
    "cool_pavement": {
        "title": "Reflective Road Coating",
        "zone_kind": "roadway",
        "dimensions": {"albedo_target": 0.35},
        "spacing_m": None,
        "min_clear_width_m": 0.0,
        "unit": "m2",
        "cost_inr": (600, 1_200),
        "lifespan_years": 4,
        "maintenance": "Recoat every 3-5 years; skid resistance check after application.",
        "permits": ["Road-owning agency approval", "Skid resistance / friction test", "Traffic diversion plan"],
        "phase": "quick_win",
        "cooling_c": (0.3, 0.8, 1.2),
        "kontext": "resurface {zones} with a light-gray reflective coating while keeping lane markings and kerbs",
    },
    "permeable_pave": {
        "title": "Permeable Footpath Pavers",
        "zone_kind": "sidewalk",
        "dimensions": {"paver_mm": 80},
        "spacing_m": None,
        "min_clear_width_m": 0.0,
        "unit": "m2",
        "cost_inr": (1_800, 3_000),
        "lifespan_years": 20,
        "maintenance": "Vacuum sweeping of joints twice a year.",
        "permits": ["Road-owning agency approval", "Soil infiltration test", "Utility mapping"],
        "phase": "medium",
        "cooling_c": (0.2, 0.6, 0.9),
        "kontext": "repave {zones} with light-colored permeable concrete pavers",
    },
    "shade_structure": {
        "title": "Tensile Shade Canopy",
        "zone_kind": "sidewalk",
        "dimensions": {"module_m": 6.0, "depth_m": 2.5},
        "spacing_m": None,
        "min_clear_width_m": 1.8,
        "unit": "m2",
        "cost_inr": (4_000, 8_000),
        "lifespan_years": 12,
        "maintenance": "Fabric cleaning yearly, tension and anchor inspection every 6 months.",
        "permits": ["Structural design approval", "ULB permission for structures in the right of way"],
        "phase": "medium",
        "cooling_c": (0.3, 0.7, 1.1),
        "kontext": "add light fabric shade sails on slender steel posts over {zones}",
    },
    "cool_roof": {
        "title": "Reflective Cool Roof",
        "zone_kind": "roof",
        "dimensions": {"assumed_roof_depth_m": 10.0},
        "spacing_m": None,
        "min_clear_width_m": 0.0,
        "unit": "m2",
        "cost_inr": (200, 450),
        "lifespan_years": 8,
        "maintenance": "Wash yearly, recoat every 5-10 years.",
        "permits": ["Building owner consent"],
        "phase": "quick_win",
        "cooling_c": (0.4, 0.9, 1.3),
        "kontext": "coat {zones} with bright white reflective roofing",
    },
    "green_roof": {
        "title": "Extensive Green Roof",
        "zone_kind": "roof",
        "dimensions": {"assumed_roof_depth_m": 10.0, "substrate_mm": 100},
        "spacing_m": None,
        "min_clear_width_m": 0.0,
        "unit": "m2",
        "cost_inr": (2_500, 5_000),
        "lifespan_years": 30,
        "maintenance": "Weeding and irrigation checks quarterly.",
        "permits": ["Structural load assessment", "Building owner consent", "Waterproofing warranty"],
        "phase": "long_term",
        "cooling_c": (0.6, 1.4, 2.0),
        "kontext": "cover {zones} with low green sedum roofs",
    },
    "facade_awning": {
        "title": "Cantilevered Shopfront Awnings",
        "zone_kind": "facade",
        "dimensions": {"depth_m": 1.2, "clearance_m": 2.4},
        "spacing_m": None,
        "min_clear_width_m": 1.8,
        "unit": "linear m",
        "cost_inr": (6_000, 12_000),
        "lifespan_years": 10,
        "maintenance": "Fabric or sheet cleaning yearly, bracket inspection.",
        "permits": ["Building owner consent", "ULB permission for projections over the footpath"],
        "phase": "quick_win",
        "cooling_c": (0.2, 0.5, 0.8),
        "kontext": ("add simple cantilevered metal shopfront awnings about 1.2 m deep, mounted above head height "
                    "over the ground-floor shops along {zones}"),
    },
    "bus_shelter": {
        "title": "Cool-Roof Bus Shelter",
        "zone_kind": "sidewalk",
        "dimensions": {"length_m": 6.0, "depth_m": 1.5},
        "spacing_m": None,
        "min_clear_width_m": 1.8,
        "unit": "shelter",
        "cost_inr": (600_000, 1_200_000),
        "lifespan_years": 15,
        "maintenance": "Roof coating renewal every 5 years, seating and lighting upkeep.",
        "permits": ["Transit agency approval (bus stop must exist)", "ULB permission"],
        "phase": "medium",
        "cooling_c": (0.1, 0.3, 0.5),
        "kontext": "add one modern bus shelter with a white reflective roof on {zones}",
    },
    "solar_canopy": {
        "title": "Solar PV Pergola over Walkway",
        "zone_kind": "sidewalk",
        "dimensions": {"depth_m": 2.5, "clearance_m": 2.7, "kwp_per_m2": 0.18},
        "spacing_m": None,
        "min_clear_width_m": 1.8,
        "unit": "m2",
        "cost_inr": (14_000, 22_000),
        "lifespan_years": 25,
        "maintenance": "Panel cleaning monthly, inverter service yearly.",
        "permits": ["Structural design approval", "DISCOM net-metering approval", "Electrical inspectorate clearance"],
        "phase": "long_term",
        "cooling_c": (0.3, 0.7, 1.0),
        "kontext": "add a slim solar panel pergola on steel columns over {zones}",
    },
    "kerb_bioswale": {
        "title": "Kerb Extension Rain Garden",
        "zone_kind": "kerb",
        "dimensions": {"length_m": 6.0, "width_m": 1.8},
        "spacing_m": 25.0,
        "min_clear_width_m": 0.0,
        "unit": "rain garden",
        "cost_inr": (65_000, 130_000),
        "lifespan_years": 20,
        "maintenance": "Desilting before monsoon, replanting as needed.",
        "permits": ["Stormwater drain department approval", "Soil infiltration test", "Utility mapping"],
        "phase": "medium",
        "cooling_c": (0.1, 0.3, 0.5),
        "kontext": "add small planted rain gardens in kerb extensions with low grasses along {zones}",
    },
    "cool_wall": {
        "title": "Reflective Wall Coating",
        "zone_kind": "facade",
        "dimensions": {"coated_height_m": 7.0, "solar_reflectance": 0.6},
        "spacing_m": None,
        "min_clear_width_m": 0.0,
        "unit": "m2",
        "cost_inr": (150, 350),
        "lifespan_years": 6,
        "maintenance": "Repaint every 5-8 years.",
        "permits": ["Building owner consent", "Heritage committee approval where applicable", "Glare check"],
        "phase": "quick_win",
        "cooling_c": (0.1, 0.3, 0.5),
        "kontext": "repaint the sun-facing walls along {zones} in a light, matte reflective tone",
    },
    "misting_system": {
        "title": "Pedestrian Misting Line",
        "zone_kind": "sidewalk",
        "dimensions": {"nozzle_spacing_m": 1.0},
        "spacing_m": None,
        "min_clear_width_m": 1.8,
        "unit": "linear m",
        "cost_inr": (3_000, 6_000),
        "lifespan_years": 8,
        "maintenance": "Nozzle descaling monthly, water treatment.",
        "permits": ["Water supply connection", "ULB permission"],
        "phase": "quick_win",
        "cooling_c": (0.5, 1.0, 2.0),
        "kontext": "add a slim misting line along {zones}",
    },
    # Never generated by the planner today; the prompt only greens facades when a plan includes it.
    "green_facade": {
        "title": "Trellis Green Facade",
        "zone_kind": "facade",
        "dimensions": {"trellis_height_m": 4.0},
        "spacing_m": None,
        "min_clear_width_m": 0.0,
        "unit": "m2",
        "cost_inr": (3_000, 7_000),
        "lifespan_years": 15,
        "maintenance": "Irrigation and pruning quarterly.",
        "permits": ["Building owner consent"],
        "phase": "medium",
        "cooling_c": (0.1, 0.3, 0.6),
        "kontext": "add climbing plants on steel trellises on parts of the walls along {zones}",
    },
}

# Quantity rules for the implementation plan (per CLEARANCES["street_length_m"] of street, per zone).
QUANTITY = {
    "awning_frontage_share": 0.6,     # share of frontage with shopfronts that receive awnings
    "cool_wall_frontage_share": 0.7,  # share of frontage with sun-facing walls
    "canopy_length_share": 0.5,       # share of street length covered by tensile / PV canopies
    "misting_length_share": 0.3,
    "green_facade_frontage_share": 0.3,
    "bus_shelters_per_length": 1,     # only where a bus stop exists
}

# ---------------------------------------------------------------------------
# 4. Thermal model coefficients
# ---------------------------------------------------------------------------
THERMAL = {
    # Land-cover shift seen by the Landsat regression per unit coverage, and its cap.
    "land_cover_shift": {
        "tree_canopy": {"f_tree": (0.10, 0.06)},
        "parking_lane_tree": {"f_tree": (0.08, 0.05)},
        "green_roof": {"f_grass": (0.05, 0.03)},
        "kerb_bioswale": {"f_grass": (0.03, 0.02)},
    },
    # Tree cooling ramps with coverage: central * (base + slope * coverage).
    "tree_coverage_ramp": (0.6, 0.4),
    "fallback_baseline_c": 38.5,
}

# Expected-impact deltas per unit coverage used by the planner's before/after summary.
IMPACT = {
    "tree_canopy":       {"heat": -0.15, "shade": 45.0, "veg": 18.0, "imp": -8.0, "ramp": True},
    "parking_lane_tree": {"heat": -0.12, "shade": 35.0, "veg": 12.0, "imp": -6.0, "ramp": True},
    "shade_structure":   {"heat": -0.10, "shade": 30.0, "veg": 0.0, "imp": 0.0},
    "facade_awning":     {"heat": -0.06, "shade": 20.0, "veg": 0.0, "imp": 0.0},
    "solar_canopy":      {"heat": -0.10, "shade": 30.0, "veg": 0.0, "imp": 0.0},
    "bus_shelter":       {"heat": -0.03, "shade": 5.0, "veg": 0.0, "imp": 0.0},
    "cool_pavement":     {"heat": -0.12, "shade": 0.0, "veg": 0.0, "imp": 0.0},
    "permeable_pave":    {"heat": -0.08, "shade": 0.0, "veg": 0.0, "imp": -16.0},
    "kerb_bioswale":     {"heat": -0.04, "shade": 0.0, "veg": 4.0, "imp": -4.0},
    "green_roof":        {"heat": -0.10, "shade": 0.0, "veg": 15.0, "imp": 0.0},
    "cool_roof":         {"heat": -0.08, "shade": 0.0, "veg": 0.0, "imp": 0.0},
    "cool_wall":         {"heat": -0.04, "shade": 0.0, "veg": 0.0, "imp": 0.0},
    "misting_system":    {"heat": -0.05, "shade": 0.0, "veg": 0.0, "imp": 0.0},
    "green_facade":      {"heat": -0.04, "shade": 0.0, "veg": 3.0, "imp": 0.0},
}

# ---------------------------------------------------------------------------
# 5. Budget selection
# ---------------------------------------------------------------------------
BUDGET = {
    "inr_per_lakh": 100_000,
    "max_interventions_with_budget": 8,
}
