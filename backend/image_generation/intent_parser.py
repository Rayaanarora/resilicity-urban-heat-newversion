"""Natural-language refinement intent parser for Generative AI urban redesign.

Interprets arbitrary natural-language instructions from the user, decomposes compound requests,
extracts additions, removals, modifications, and preservation constraints, and produces a
structured design intent specification for Gemini image editing.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

from .schemas import RefinementIntent

logger = logging.getLogger("resilicity.intent_parser")

DEFAULT_PRESERVATION_RULES = [
    "existing buildings, facades, storefronts, and architectural window patterns",
    "camera viewpoint, horizon line, and 3D street perspective",
    "road alignment, curb lines, and vehicular traffic corridors",
    "vehicles, utility poles, streetlights, and existing structural elements",
    "consistent ambient sunlight angle, natural shadows, and sky lighting",
]

PRESERVATION_TRIGGER_PATTERNS = [
    (r"keep (?:all |the )?(?:existing )?buildings?(?: and storefronts?)?(?: unchanged| intact| as is| where they are)?", "buildings and storefronts"),
    (r"keep (?:all |the )?(?:existing )?storefronts?", "storefronts and commercial signage"),
    (r"keep (?:the )?(?:road|street)(?: geometry)?(?: unchanged)?", "road geometry and lanes"),
    (r"keep (?:the )?(?:vehicles|cars)", "vehicles and traffic"),
    (r"do not change (?:the )?buildings?", "buildings and architectural facades"),
    (r"do not change (?:the )?road", "road layout and geometry"),
    (r"preserve (?:the )?architecture", "original architectural identity"),
]

REMOVAL_TRIGGER_PATTERNS = [
    (r"(?:remove|delete|take away|eliminate) (?:the |all )?pergola", "pergola shade structure"),
    (r"(?:remove|delete|take away|eliminate) (?:the |all )?shade (?:structures?|canop(?:y|ies))", "shade structures and canopies"),
    (r"(?:remove|delete|take away|eliminate) (?:the |all )?benches?(?: and seating)?", "benches and seating"),
    (r"(?:remove|delete|take away|eliminate) (?:the |all )?trees?", "selected trees"),
    (r"(?:reduce|decrease|lower) (?:the )?(?:number of |density of )?trees?", "tree density and quantity"),
    (r"(?:reduce|make less dense) (?:the )?greenery", "excessive dense vegetation"),
    (r"(?:make|render) (?:the )?trees? (?:smaller|shorter|younger|more widely spaced)", "tree canopy scale (smaller, widely spaced)"),
    (r"keep (?:the )?original pavement", "revert to original pavement surface"),
]

ADDITION_TRIGGER_PATTERNS = [
    (r"(?:add|plant|place|grow) (?:(\w+) )?(?:large |native |mature )?trees?", "native shade trees"),
    (r"(?:add|install|create) (?:a )?(?:modern |timber |architectural )?pergola", "timber pergola shade structure"),
    (r"(?:add|install|provide) (?:shaded )?seating|benches", "shaded pedestrian seating and benches"),
    (r"(?:add|install) (?:a |lightweight )?shade (?:canopy|sail|structure)", "architectural tensile fabric shade canopy"),
    (r"(?:add|create|plant) (?:a )?(?:planted )?bioswale|rain garden", "vegetated bioswale / rain garden for stormwater"),
    (r"(?:turn|convert) (?:this |the )?(?:empty )?(?:corner|space|roadside) into (?:a )?(?:shaded )?pocket park", "pocket park with shade trees and permeable seating plaza"),
    (r"(?:add|install) planter boxes|green buffer", "linear planter boxes and green buffer strip"),
    (r"(?:add|plant) climbing plants|vertical greenery|green wall", "vertical climbing vegetation on bare vertical walls"),
    (r"(?:add|create) (?:a )?shaded bus stop", "shaded transit bus stop shelter"),
    (r"(?:replace|change) (?:the )?pavement with (?:light |reflective )?permeable pavers", "light-colored interlocking permeable stone pavers"),
    (r"(?:make|apply) (?:the )?pavement (?:lighter|more reflective|cool)", "solar-reflective high-albedo light-gray pavement coating"),
    (r"(?:increase|more) (?:pedestrian )?greenery", "expanded pedestrian greenery and shrubs"),
    (r"(?:add|create) (?:a )?continuous shaded pedestrian (?:path|corridor|walkway)", "continuous shaded pedestrian corridor with aligned canopy trees"),
]


def parse_refinement_intent_rules(instruction: str, context: Optional[str] = None) -> RefinementIntent:
    """Deterministic, robust natural-language intent parser.
    
    Extracts goals, additions, removals, modifications, and preservation constraints
    from compound or colloquial user instructions without requiring fixed button categories.
    """
    clean_instr = instruction.strip()
    lower_instr = clean_instr.lower()

    add_items: List[str] = []
    remove_items: List[str] = []
    modify_items: List[str] = []
    preserve_items: List[str] = list(DEFAULT_PRESERVATION_RULES)
    spatial_constraints: List[str] = []

    # 1. Detect Preservations
    for pattern, label in PRESERVATION_TRIGGER_PATTERNS:
        if re.search(pattern, lower_instr):
            if label not in preserve_items:
                preserve_items.insert(0, f"STRICT USER REQUIREMENT: {label}")

    # 2. Detect Removals / Reductions
    for pattern, label in REMOVAL_TRIGGER_PATTERNS:
        if re.search(pattern, lower_instr):
            remove_items.append(label)

    # 3. Detect Additions
    for pattern, label in ADDITION_TRIGGER_PATTERNS:
        match = re.search(pattern, lower_instr)
        if match:
            # Check if count or specific qualifier exists
            matched_text = match.group(0)
            add_items.append(matched_text)

    # 4. Handle Modifications
    if "lighter" in lower_instr or "reflective" in lower_instr:
        modify_items.append("increase pavement solar reflectance (albedo) to a clean light-stone tone")
    if "smaller" in lower_instr or "less dense" in lower_instr or "widely spaced" in lower_instr:
        modify_items.append("scale down canopy volume and increase tree spacing")
    if "wider" in lower_instr:
        modify_items.append("adjust pedestrian and vehicular proportions plausibly")

    # 5. Fallback for open-ended or un-patterned instructions
    if not add_items and not remove_items and not modify_items:
        # User provided an open-ended request like "Make this street look like a Chennai resilience project"
        # or "Make the space more wheelchair accessible"
        modify_items.append(clean_instr)

    # 6. Physical Plausibility and Safety Heuristics
    # Prevent hallucinating trees in the middle of active road traffic
    if re.search(r"(?:in|on) (?:the )?(?:middle of the |center of the )?road", lower_instr):
        spatial_constraints.append(
            "Plausibility Adaptation: Position trees and vegetation along sidewalk curbs, verges, or dedicated median islands. Do NOT obstruct active traffic travel lanes."
        )
    else:
        spatial_constraints.append(
            "Respect functional street geometry: keep vehicle travel lanes clear, anchor pedestrian shade to walkways, and ensure curb visibility."
        )

    # Derive high-level goal
    goal = clean_instr
    if len(goal) > 120:
        goal = goal[:117] + "..."

    return RefinementIntent(
        goal=goal,
        add=add_items,
        remove=remove_items,
        modify=modify_items,
        preserve=preserve_items,
        spatial_constraints=spatial_constraints,
    )


def parse_refinement_intent(instruction: str, context: Optional[str] = None) -> RefinementIntent:
    """Parse user instruction into structured RefinementIntent using LLM if available, else rules."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return parse_refinement_intent_rules(instruction, context)

    # Use lightweight Gemini call to structure complex/ambiguous open-ended requests
    prompt = f"""You are an urban design planning assistant. Parse this user design edit request into structured JSON for an architectural image-editing model.

User Request: "{instruction}"

Extract the following JSON schema:
{{
  "goal": "Brief summary of user design objective",
  "add": ["specific architectural or vegetative elements to add"],
  "remove": ["specific elements to remove, delete, or reduce"],
  "modify": ["specific elements to alter in scale, material, or color"],
  "preserve": ["specific scene features to preserve, especially if user explicitly requested keeping buildings, storefronts, or road intact"],
  "spatial_constraints": ["physical placement rules so the edit is realistic and does not block traffic or float"]
}}

Return ONLY valid raw JSON without markdown code fences."""

    endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 600,
        }
    }

    try:
        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(raw_text)
                
                # Combine default preservation with extracted preservations
                preserves = list(DEFAULT_PRESERVATION_RULES)
                for p in parsed.get("preserve", []):
                    if p not in preserves:
                        preserves.insert(0, p)

                return RefinementIntent(
                    goal=parsed.get("goal", instruction),
                    add=parsed.get("add", []),
                    remove=parsed.get("remove", []),
                    modify=parsed.get("modify", []),
                    preserve=preserves,
                    spatial_constraints=parsed.get("spatial_constraints", [
                        "Respect functional street geometry: do not block vehicular lanes, keep pedestrian paths clear."
                    ]),
                )
    except Exception as e:
        logger.warning(f"LLM intent parser fallback to rule parser: {e}")

    return parse_refinement_intent_rules(instruction, context)
