import sys, time, json
from pathlib import Path
from PIL import Image
from segmentation import SegFormerEngine
from scene_understanding import analyze_scene
from planner import generate_spatial_plan

img_path = Path('../gen-ai/public/samples/sample_1_dense_urban.jpg')
img = Image.open(img_path).convert('RGB')
print(f"Image: {img.size}")

t0 = time.time()
engine = SegFormerEngine.get_instance()
seg = engine.segment_image(img)
print(f"SegFormer time: {time.time()-t0:.2f}s")
surfaces = {m['className']: m['areaPercentage'] for m in seg['masks']}
print("Surfaces:", surfaces)
print("Protected objects:", [f"{p['id']} ({p['percentage']} pct)" for p in seg.get('protected_objects', [])])

t1 = time.time()
scene = analyze_scene(img, seg)
print(f"Scene understanding time: {time.time()-t1:.2f}s")
print("Geometry:", json.dumps(scene.street_geometry.__dict__, indent=2))
print("Left Sidewalk:", json.dumps(scene.left_sidewalk.__dict__, indent=2))
print("Right Sidewalk:", json.dumps(scene.right_sidewalk.__dict__, indent=2))
print("Roadway:", json.dumps(scene.roadway.__dict__, indent=2))
print("Heat priority:", scene.heat_priority_summary)

t2 = time.time()
plan = generate_spatial_plan(img, surfaces, seg_result=seg)
print(f"Planner time: {time.time()-t2:.2f}s")
print("Planner source:", plan.planner_source)
print("Site summary:", plan.site_summary)
for iv in plan.interventions:
    print(f"  - [{iv.priority}] {iv.title} on {iv.target_zone or iv.target_region} (Utility: {iv.utility_score})")
    for ev in iv.evidence:
        print(f"      * {ev}")
