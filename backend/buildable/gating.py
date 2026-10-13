"""Rule-based gating: can this intervention be built in this zone, and if not, why not.

Every decision carries a rule id, a plain-language reason and the measured values behind it, so
rejected candidates can be reviewed by an engineer. All thresholds come from config.CLEARANCES
and the catalog.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .catalog import get_spec
from .config import CLEARANCES
from .models import StreetMetrics
from .observations import SiteObservations


@dataclass
class SiteContext:
    metrics: StreetMetrics
    observations: SiteObservations
    climate_zone: str
    canyon_hw_proxy: float
    canyon_strength: float
    wall_density: float
    solar_exposure: Dict[str, float]  # zone -> exposure proxy
    heritage_roadway: bool = False
    heritage_material: str = ""


@dataclass
class GateResult:
    allowed: bool
    rule: str = ""
    reason: str = ""
    measured: Dict[str, Any] = field(default_factory=dict)
    # Conditions attached to an allowed item (e.g. junction setback, verify width on site).
    conditions: List[str] = field(default_factory=list)
    # Sub-items excluded even though the intervention is allowed (e.g. pits inside a sight triangle).
    partial_rejections: List[Dict[str, Any]] = field(default_factory=list)
    # Street length (m) excluded from planting, e.g. a junction setback.
    excluded_length_m: float = 0.0

    def rejection(self, itype: str, zone: str) -> Dict[str, Any]:
        return {"type": itype, "target_zone": zone, "reason": self.reason, "conflict": self.rule,
                "rule": self.rule, "measured": self.measured}


def _zone_label(zone: str) -> str:
    return zone.replace("_", " ")


def _reject(rule: str, reason: str, **measured: Any) -> GateResult:
    return GateResult(False, rule, reason, measured)


def _width_check(itype: str, zone: str, ctx: SiteContext, required_m: float, what: str) -> Optional[GateResult]:
    """Reject when the measured sidewalk cannot fit `required_m`; None when it fits.

    Without a carriageway scale the photo says nothing about widths, so the item is not rejected;
    _verify_condition attaches a survey requirement instead. With a scale but no sidewalk found,
    the absence is evidence and the item is rejected.
    """
    if not ctx.metrics.scale_available:
        return None
    width = ctx.metrics.sidewalk_m(zone)
    rng = ctx.metrics.left_range_m if zone == "left_sidewalk" else ctx.metrics.right_range_m
    if width is None:
        return _reject("no_measurable_sidewalk",
                       f"No measurable sidewalk on the {_zone_label(zone)}; {what} needs {required_m:.1f} m.",
                       required_m=required_m)
    if width < required_m:
        return _reject(f"{itype}_clear_width",
                       f"{_zone_label(zone).capitalize()} is about {width:.1f} m (range {rng[0]:.1f}-{rng[1]:.1f} m); "
                       f"{what} needs {required_m:.1f} m.",
                       sidewalk_m=width, sidewalk_range_m=rng, required_m=required_m)
    return None


def _verify_condition(zone: str, ctx: SiteContext, required_m: float) -> List[str]:
    if not ctx.metrics.scale_available:
        return [f"Width not measurable from the photo; survey that the {_zone_label(zone)} has at least "
                f"{required_m:.1f} m before construction."]
    rng = ctx.metrics.left_range_m if zone == "left_sidewalk" else ctx.metrics.right_range_m
    if rng and rng[0] < required_m:
        return [f"Width fits only at the central estimate; verify on site that the {_zone_label(zone)} "
                f"is at least {required_m:.1f} m."]
    return []


def _obstacle_check(itype: str, zone: str, ctx: SiteContext, groups: List[str]) -> Optional[GateResult]:
    fr = ctx.observations.obstacle_fractions.get(zone, {})
    limits = {"pole": CLEARANCES["max_pole_strip_frac"], "vehicle": CLEARANCES["max_vehicle_strip_frac"],
              "person": CLEARANCES["max_person_strip_frac"]}
    words = {"pole": "utility poles, lamps or signals", "vehicle": "vehicles parked along the kerb",
             "person": "dense pedestrian or vendor activity"}
    for g in groups:
        if fr.get(g, 0.0) > limits[g]:
            return _reject(f"tree_near_{g}",
                           f"{words[g].capitalize()} occupy {fr[g]*100:.0f}% of the {_zone_label(zone)} kerb strip "
                           f"(limit {limits[g]*100:.0f}%); no tree pits there.",
                           strip_fraction=fr[g], limit=limits[g])
    return None


def _junction_setback(itype: str, zone: str, ctx: SiteContext, result: GateResult) -> GateResult:
    if ctx.observations.junction_suspected.get(zone):
        d = CLEARANCES["junction_sight_clearance_m"]
        result.excluded_length_m = d
        result.conditions.append(f"Keep the first {d:.0f} m from the junction on the {_zone_label(zone)} free of trees (sight triangle).")
        result.partial_rejections.append({
            "type": itype, "target_zone": zone, "rule": "junction_sight_line", "conflict": "junction_sight_line",
            "reason": f"Possible junction on the {_zone_label(zone)}: no trees within {d:.0f} m of it to keep sight lines clear.",
            "measured": {"setback_m": d},
        })
    return result


def gate(itype: str, zone: str, ctx: SiteContext) -> GateResult:
    """Decide whether `itype` can be built in `zone`. Unknown types are allowed without conditions."""
    spec = get_spec(itype)
    c = CLEARANCES

    if itype == "misting_system" and ctx.climate_zone in c["humid_climates"]:
        return _reject("misting_humid_climate",
                       f"Misting and evaporative cooling are ineffective and raise discomfort in humid climates "
                       f"({ctx.climate_zone}).", climate_zone=ctx.climate_zone)

    if itype == "cool_pavement":
        if ctx.heritage_roadway:
            return _reject("heritage_preservation",
                           f"Preserving the {ctx.heritage_material.replace('_', ' ') or 'heritage'} roadway; "
                           "synthetic coating rejected.", material=ctx.heritage_material)
        if ctx.canyon_hw_proxy < c["deep_canyon_hw_proxy_min"]:
            return _reject("cool_pavement_open_street",
                           f"Street is not a deep canyon (H/W proxy {ctx.canyon_hw_proxy:.2f} < "
                           f"{c['deep_canyon_hw_proxy_min']:.2f}); a reflective road would bounce sun onto pedestrians.",
                           hw_proxy=ctx.canyon_hw_proxy, required=c["deep_canyon_hw_proxy_min"])
        return GateResult(True)

    if itype == "tree_canopy":
        required = spec.dimensions["pit_m"] + c["kerb_buffer_m"] + c["min_clear_walkway_m"]
        fail = _width_check(itype, zone, ctx, required, f"a {spec.dimensions['pit_m']:.1f} m tree pit plus a "
                            f"{c['min_clear_walkway_m']:.1f} m clear walkway")
        if fail:
            return fail
        fail = _obstacle_check(itype, zone, ctx, ["pole", "vehicle", "person"])
        if fail:
            return fail
        return _junction_setback(itype, zone, ctx, GateResult(True, conditions=_verify_condition(zone, ctx, required)))

    if itype in ("parking_lane_tree", "kerb_bioswale"):
        remaining = ctx.metrics.road_width_m - c["parking_lane_width_m"]
        if remaining < c["min_carriageway_after_buildout_m"]:
            return _reject("buildout_carriageway_width",
                           f"Carriageway of {ctx.metrics.road_width_m:.1f} m would drop to {remaining:.1f} m with a "
                           f"kerb build-out; at least {c['min_carriageway_after_buildout_m']:.1f} m must remain.",
                           road_width_m=ctx.metrics.road_width_m, remaining_m=remaining,
                           required_m=c["min_carriageway_after_buildout_m"])
        result = GateResult(True, conditions=["Assumes a kerbside parking lane exists; confirm with the traffic police."])
        if itype == "parking_lane_tree":
            fail = _obstacle_check(itype, zone, ctx, ["pole"])
            if fail:
                return fail
            result = _junction_setback(itype, zone, ctx, result)
        return result

    if itype == "facade_awning":
        if ctx.wall_density < c["min_wall_density_for_awning"]:
            return _reject("no_frontage", "Too little building frontage visible to mount awnings.",
                           wall_density=ctx.wall_density, required=c["min_wall_density_for_awning"])
        depth = spec.dimensions["depth_m"]
        width = ctx.metrics.sidewalk_m(zone)
        if width is None:
            return GateResult(True, conditions=[f"No footpath measured; awnings must stay within the property line "
                                                f"and keep {c['min_vertical_clearance_m']:.1f} m headroom."])
        fail = _width_check(itype, zone, ctx, depth + c["kerb_buffer_m"], f"a {depth:.1f} m awning set back from the kerb")
        return fail or GateResult(True, conditions=[f"Mount above {c['min_vertical_clearance_m']:.1f} m headroom."])

    if itype in ("shade_structure", "solar_canopy", "bus_shelter", "misting_system"):
        extra = spec.dimensions.get("depth_m", 0.0) if itype == "bus_shelter" else c["kerb_buffer_m"]
        required = c["min_clear_walkway_m"] + extra
        fail = _width_check(itype, zone, ctx, required, f"posts/structure plus a {c['min_clear_walkway_m']:.1f} m clear walkway")
        if fail:
            return fail
        if itype == "solar_canopy":
            sol = ctx.solar_exposure.get(zone, 0.0)
            if sol < c["min_solar_exposure_for_pv"] or ctx.canyon_strength > c["max_canyon_strength_for_pv"]:
                return _reject("pv_insufficient_sun",
                               f"Too little sun for PV on the {_zone_label(zone)} (exposure {sol:.2f}, canyon "
                               f"{ctx.canyon_strength:.2f}).", solar_exposure=sol, canyon_strength=ctx.canyon_strength)
        if itype == "bus_shelter" and ctx.metrics.road_width_m < c["min_road_width_for_bus_shelter_m"]:
            return _reject("bus_shelter_minor_street",
                           f"Carriageway {ctx.metrics.road_width_m:.1f} m is below a typical bus route "
                           f"({c['min_road_width_for_bus_shelter_m']:.1f} m).", road_width_m=ctx.metrics.road_width_m)
        conditions = _verify_condition(zone, ctx, required)
        if itype == "bus_shelter":
            conditions.append("Only where a bus stop exists on this street.")
        return GateResult(True, conditions=conditions)

    if itype == "cool_wall" and ctx.wall_density < c["min_wall_density_for_cool_wall"]:
        return _reject("no_frontage", "Too little sun-facing wall visible to justify a wall coating.",
                       wall_density=ctx.wall_density, required=c["min_wall_density_for_cool_wall"])

    return GateResult(True)
