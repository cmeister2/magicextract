#!/usr/bin/env python3
"""Explicitly author coherent pinhole-camera views of a complete planar scene.

Authoring uses Pillow==10.2.0, cached pinned source PNGs, and reviewed local
backgrounds. Tests load the saved JPEGs; they never invoke this tool.

Reusable API: render_scene(recipe, sources, seed=...) returns (RGB, cards);
write_scene(directory, recipe, sources, seed=..., overwrite=False) returns
(manifest_entry, image_hashes). Recipes use the base scene author's world-plane
coordinates and add camera:{pitch_degrees,azimuth_degrees,roll_degrees,
focal_length,distance}. Pitch is the angle from the plane normal. The defaults
are 0,0,0,900,1050. Camera effects apply after projecting the entire plane.
"""

import argparse
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

import PIL
from PIL import Image, ImageChops, features


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures/camera"
DEFAULT_SEED = 20261019
SIZE = (640, 520)
PLANE_SIZE = (640, 520)
PRINCIPAL_POINT = (319.5, 259.5)


def load_base_renderer():
    """Import the reviewed plane renderer without invoking its authoring CLI."""
    specification = importlib.util.spec_from_file_location(
        "camera_world_scene_author", ROOT / "scripts/build-scene-fixtures.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BASE = load_base_renderer()


def dot(first, second):
    """Compute a three-dimensional scalar product."""
    return sum(a * b for a, b in zip(first, second))


def cross(first, second):
    """Compute a three-dimensional vector product."""
    return (first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0])


def normalize(vector):
    """Return a unit vector, rejecting degenerate camera axes."""
    length = math.sqrt(dot(vector, vector))
    if length < 1e-12:
        raise ValueError("Camera axes must define a nondegenerate view")
    return tuple(value / length for value in vector)


def project_point(matrix, point):
    """Apply the shared plane-to-image homography to a world-plane point."""
    homogeneous = (point[0], point[1], 1.0)
    denominator = dot(matrix[2], homogeneous)
    if denominator <= 0:
        raise ValueError("The entire annotated plane must remain in front of the camera")
    return [dot(matrix[0], homogeneous) / denominator,
            dot(matrix[1], homogeneous) / denominator]


def camera_projection(parameters):
    """Construct proper world-to-camera extrinsics and K[R|t] for one plane.

    The plane origin is its center. World +x is right, +y is down, and +z points
    away from an overhead camera. World-plane pixel (u,v) is (u-319.5,v-259.5,0).
    The camera orbits the plane center and looks at it, with a declared roll.
    """
    camera = {
        "pitch_degrees": 0.0, "azimuth_degrees": 0.0, "roll_degrees": 0.0,
        "focal_length": 900.0, "distance": 1050.0,
    }
    camera.update(parameters or {})
    for name, value in camera.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Camera {name} must be a finite number")
    if not 0 <= camera["pitch_degrees"] < 85 or camera["focal_length"] <= 0 or camera["distance"] <= 0:
        raise ValueError("Use pitch in [0,85), a positive focal length, and positive distance")
    pitch, azimuth, roll = (math.radians(camera[key]) for key in
                            ("pitch_degrees", "azimuth_degrees", "roll_degrees"))
    distance = camera["distance"]
    center = (distance * math.sin(pitch) * math.cos(azimuth),
              distance * math.sin(pitch) * math.sin(azimuth),
              -distance * math.cos(pitch))
    forward = normalize(tuple(-value for value in center))
    right = normalize(cross(forward, (0.0, -1.0, 0.0)))
    down = normalize(cross(forward, right))
    rolled_right = tuple(math.cos(roll) * a + math.sin(roll) * b for a, b in zip(right, down))
    rolled_down = tuple(-math.sin(roll) * a + math.cos(roll) * b for a, b in zip(right, down))
    rotation = [list(rolled_right), list(rolled_down), list(forward)]
    translation = [-dot(axis, center) for axis in rotation]
    half_width, half_height = ((value - 1) / 2 for value in PLANE_SIZE)
    plane_rows = [[axis[0], axis[1], offset - axis[0] * half_width - axis[1] * half_height]
                  for axis, offset in zip(rotation, translation)]
    focal = camera["focal_length"]
    cx, cy = PRINCIPAL_POINT
    matrix = [[focal * plane_rows[0][i] + cx * plane_rows[2][i] for i in range(3)],
              [focal * plane_rows[1][i] + cy * plane_rows[2][i] for i in range(3)],
              list(plane_rows[2])]
    world_plane = [[0.0, 0.0], [PLANE_SIZE[0] - 1.0, 0.0],
                   [PLANE_SIZE[0] - 1.0, PLANE_SIZE[1] - 1.0], [0.0, PLANE_SIZE[1] - 1.0]]
    projected_plane = [project_point(matrix, point) for point in world_plane]
    metadata = {
        "model": "pinhole camera; entire mat, cards, print and occluders share one plane",
        "world_axes": "+x right, +y down, +z away from the overhead camera",
        "world_plane_size": list(PLANE_SIZE), "world_plane_origin": [half_width, half_height],
        "image_size": list(SIZE), "principal_point": list(PRINCIPAL_POINT),
        "intrinsic_matrix": [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        "camera_center_world": list(center), "world_to_camera_rotation": rotation,
        "world_to_camera_translation": translation, "plane_to_image_homography": matrix,
        "projected_plane_corners": projected_plane,
        "horizontal_fov_degrees": math.degrees(2 * math.atan(SIZE[0] / (2 * focal))),
        "vertical_fov_degrees": math.degrees(2 * math.atan(SIZE[1] / (2 * focal))),
        "pose": camera,
    }
    return matrix, projected_plane, metadata


def project_plane(source, plane_corners, size=SIZE):
    """Project a complete RGBA plane using the same map as every annotation."""
    return BASE.project(source, plane_corners, size=size)


def world_visibility_masks(recipe, sources):
    """Recover source and visible alpha masks before the shared camera projection."""
    originals, visible = [], []
    for card in recipe.get("cards", []):
        alpha = BASE.project(sources[card["source"]], card["corners"]).getchannel("A")
        inverse = ImageChops.invert(alpha)
        visible = [ImageChops.multiply(mask, inverse) for mask in visible]
        originals.append(alpha)
        visible.append(alpha.copy())
    for occluder in recipe.get("occluders", []):
        inverse = ImageChops.invert(BASE.occluder_layer(occluder).getchannel("A"))
        visible = [ImageChops.multiply(mask, inverse) for mask in visible]
    return originals, visible


def alpha_sum(image):
    """Sum fractional visible coverage in an antialiased alpha mask."""
    return sum(value * count for value, count in enumerate(image.histogram()))


def project_mask(mask, plane_corners, size=SIZE):
    """Transform an occlusion mask without introducing a new geometric model."""
    layer = Image.new("RGBA", PLANE_SIZE, (255, 255, 255, 0))
    layer.putalpha(mask)
    return project_plane(layer, plane_corners, size=size).getchannel("A")


def render_scene(recipe, sources, seed=DEFAULT_SEED):
    """Render the complete world plane, then photograph it with one pinhole camera."""
    matrix, plane_corners, projection = camera_projection(recipe.get("camera", {}))
    world_recipe = deepcopy(recipe)
    world_recipe["effects"] = []
    world, cards = BASE.render_scene(world_recipe, sources, seed=seed)
    projected = project_plane(world.convert("RGBA"), plane_corners)
    # The neutral tabletop around a finite mat lies on the same plane. Its
    # uniform color is invariant to the projective mapping.
    outside = tuple(recipe.get("table_color", (63, 58, 51)))
    image = Image.new("RGBA", SIZE, outside + (255,))
    image = Image.alpha_composite(image, projected).convert("RGB")
    originals, visible = world_visibility_masks(world_recipe, sources)
    for card, original, observed in zip(cards, originals, visible):
        world_corners = card["corners"]
        card["world_corners"] = deepcopy(world_corners)
        card["corners"] = [project_point(matrix, point) for point in world_corners]
        card["projected_edge_angles_degrees"] = [
            math.degrees(math.atan2(card["corners"][(i + 1) % 4][1] - card["corners"][i][1],
                                   card["corners"][(i + 1) % 4][0] - card["corners"][i][0]))
            for i in range(4)
        ]
        denominator = alpha_sum(project_mask(original, plane_corners))
        numerator = alpha_sum(project_mask(observed, plane_corners))
        if any(not (0 <= x < SIZE[0] and 0 <= y < SIZE[1]) for x, y in card["corners"]):
            left = math.floor(min(x for x, _ in card["corners"])) - 1
            top = math.floor(min(y for _, y in card["corners"])) - 1
            right = math.ceil(max(x for x, _ in card["corners"])) + 2
            bottom = math.ceil(max(y for _, y in card["corners"])) + 2
            shifted_plane = [[x - left, y - top] for x, y in plane_corners]
            denominator = alpha_sum(project_mask(original, shifted_plane, (right - left, bottom - top)))
        card["visible_fraction"] = round(min(1.0, numerator / denominator), 6) if denominator else 0.0
    for effect in recipe.get("effects", []):
        image = BASE.apply_effect(image, effect, seed)
    return image, cards


def write_scene(directory, recipe, sources, seed=DEFAULT_SEED, overwrite=False):
    """Save one authoritative JPEG and independent final-JPEG appearance references."""
    directory = Path(directory)
    identifier = recipe["id"]
    if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in identifier):
        raise ValueError("Scene IDs must be lowercase filename-safe identifiers")
    relative = f"images/{identifier}.jpg"
    destination = directory / relative
    reference_paths = [f"references/{identifier}-{index}-{card['source']}.png"
                       for index, card in enumerate(recipe.get("cards", []))]
    if not overwrite and any(path.exists() for path in [destination] + [directory / p for p in reference_paths]):
        raise FileExistsError(f"Refusing to replace saved camera scene {identifier}")
    image, cards = render_scene(recipe, sources, seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="JPEG", quality=recipe.get("jpeg_quality", 94),
               subsampling=0, optimize=False, progressive=False)
    hashes = {relative: BASE.digest(destination)}
    with Image.open(destination) as decoded:
        decoded = decoded.convert("RGB")
        for card, reference in zip(cards, reference_paths):
            path = directory / reference
            path.parent.mkdir(parents=True, exist_ok=True)
            BASE.rectify_reference(decoded, card["corners"]).save(path, format="PNG")
            card["reference"] = reference
            hashes[reference] = BASE.digest(path)
    params = deepcopy(recipe)
    _, _, params["projection"] = camera_projection(recipe.get("camera", {}))
    entry = {
        "id": identifier, "file": relative, "tier": recipe.get("tier", "challenge"),
        "description": recipe["description"], "tags": recipe.get("tags", []), "cards": cards,
        "corner_tolerance": 8.0, "pixel_mae_limit": 22.0 if cards else None, "params": params,
    }
    return entry, hashes


def background(kind):
    """Reuse only hash-pinned reviewed playmat bitmaps and declared printed zones."""
    if kind == "solid":
        return {"kind": "solid", "color": [75, 115, 90]}
    filename = "fantasy-forest.png" if kind == "forest" else "arcane-zones.png"
    relative = f"tests/fixtures/playmats/backgrounds/{filename}"
    result = {"kind": "image", "file": relative, "sha256": BASE.digest(ROOT / relative)}
    if kind == "printed":
        result["overlays"] = [
            {"kind": "rounded_rectangle", "bounds": [65, 90, 185, 258], "radius": 10,
             "color": [225, 204, 125, 190], "width": 3},
            {"kind": "rounded_rectangle", "bounds": [455, 260, 575, 428], "radius": 10,
             "color": [225, 204, 125, 190], "width": 3},
            {"kind": "line", "points": [[35, 345], [605, 345]], "color": [200, 184, 125, 160], "width": 2},
        ]
    return result


def scene_recipe(identifier, pitch, azimuth, kind="solid", roll=0, source="lightning-bolt", **extra):
    """Declare world geometry independently of the eventual camera view."""
    result = {
        "id": identifier, "tier": "challenge", "description": identifier.replace("-", " "),
        "tags": ["camera", "coherent-plane", kind], "background": background(kind),
        "cards": [{"source": source, "corners": BASE.rotated_corners((319.5, 259.5), 180, 0)}],
        "camera": {"pitch_degrees": pitch, "azimuth_degrees": azimuth, "roll_degrees": roll,
                   "focal_length": 900.0, "distance": 1050.0},
        "effects": [], "occluders": [], "jpeg_quality": 94,
    }
    result.update(extra)
    return result


def recipes():
    """Return the 32 mat views followed by 17 continuous-table pose controls."""
    result = [scene_recipe(f"camera-{kind}-normal-control", 0, 0, kind,
                           tags=["camera", "normal-control", kind])
              for kind in ["solid", "forest", "arcane", "printed"]]
    for pitch in [20, 35, 50, 65]:
        for azimuth in [0, 90, 180, 270]:
            result.append(scene_recipe(f"camera-solid-p{pitch}-a{azimuth:03d}", pitch, azimuth))
    for pitch, azimuth, kind, roll, source in [
        (20, 45, "forest", 12, "counterspell"), (35, 135, "arcane", -12, "black-lotus"),
        (50, 225, "forest", 8, "sol-ring"), (65, 315, "printed", -8, "lightning-bolt"),
    ]:
        result.append(scene_recipe(f"camera-{kind}-p{pitch}-a{azimuth:03d}-roll", pitch, azimuth,
                                   kind, roll, source, tags=["camera", "coherent-plane", "roll", kind]))
    for pitch, azimuth, kind in [(20, 30, "solid"), (35, 120, "forest"),
                                  (50, 210, "arcane"), (65, 300, "printed")]:
        result.append(scene_recipe(f"camera-{kind}-p{pitch}-two-cards", pitch, azimuth, kind,
                                   cards=[{"source": "counterspell", "corners": BASE.rotated_corners((195, 259.5), 140, -7)},
                                          {"source": "sol-ring", "corners": BASE.rotated_corners((444, 259.5), 140, 9)}],
                                   tags=["camera", "coherent-plane", "multiple-cards", kind]))
    for pitch, azimuth, kind, effects, occluders in [
        (20, 60, "solid", [{"kind": "gaussian_blur", "radius": 0.65}], []),
        (35, 150, "forest", [{"kind": "lighting", "gradient": True, "minimum": 0.50, "maximum": 1.05}], []),
        (50, 240, "arcane", [{"kind": "gaussian_blur", "radius": 0.55}],
         [{"kind": "ellipse", "bounds": [208, 110, 252, 156], "color": [185, 137, 106]}]),
        (65, 330, "printed", [{"kind": "lighting", "factor": 0.85, "tint": [1.10, 0.98, 0.88]},
                               {"kind": "gaussian_blur", "radius": 0.45}],
         [{"kind": "rectangle", "bounds": [390, 335, 423, 400], "color": [154, 105, 74]}]),
    ]:
        result.append(scene_recipe(f"camera-{kind}-p{pitch}-combined", pitch, azimuth, kind,
                                   effects=effects, occluders=occluders,
                                   tags=["camera", "coherent-plane", "combined-effects", kind]))
    assert len(result) == 32
    result.extend(continuous_surface_recipes())
    assert len(result) == 49
    return result


def continuous_surface_recipes():
    """Isolate pose on an unbounded solid surface without a visible mat border."""
    poses = [(0, 0)] + [(pitch, azimuth) for pitch in [20, 35, 50, 65]
                       for azimuth in [0, 90, 180, 270]]
    return [scene_recipe(
        f"camera-table-p{pitch}-a{azimuth:03d}", pitch, azimuth,
        table_color=[75, 115, 90],
        tags=["camera", "coherent-plane", "solid", "continuous-surface", "pose-control"],
        description=f"A continuous solid tabletop isolates a {pitch}-degree camera pitch from approach azimuth {azimuth} without a finite mat border.",
    ) for pitch, azimuth in poses]


def append_continuous_surface(output):
    """Append new pose controls while retaining every earlier JPEG and reference."""
    if PIL.__version__ != BASE.AUTHORING_PILLOW_VERSION:
        raise ValueError(f"Authoring requires Pillow=={BASE.AUTHORING_PILLOW_VERSION}")
    output = Path(output)
    manifest_path, recipes_path = output / "manifest.json", output / "recipes.json"
    manifest = json.loads(manifest_path.read_text())
    collection = json.loads(recipes_path.read_text())
    additions = continuous_surface_recipes()
    existing_ids = {entry["id"] for entry in manifest["scenes"]}
    if existing_ids.intersection(recipe["id"] for recipe in additions):
        raise FileExistsError("Continuous-surface camera fixtures already exist; refusing replacement")
    if existing_ids != {recipe["id"] for recipe in collection}:
        raise ValueError("Existing camera recipes and manifest IDs must agree before appending")
    for relative, expected in manifest["sha256"].items():
        if BASE.digest(output / relative) != expected:
            raise ValueError(f"Existing camera asset changed: {relative}")
    source_manifest = json.loads(BASE.SOURCE_MANIFEST.read_text())
    for source in source_manifest["cards"]:
        if BASE.digest(BASE.SOURCE_DIRECTORY / source["file"]) != source["sha256"]:
            raise ValueError(f"Changed or missing pinned PNG: {source['file']}")
    sources = BASE.load_sources()
    for recipe in additions:
        entry, pins = write_scene(output, recipe, sources)
        manifest["scenes"].append(entry)
        manifest["sha256"].update(pins)
        collection.append(recipe)
        print(f"saved {entry['id']}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    recipes_path.write_text(json.dumps(collection, indent=2) + "\n")
    BASE.contact_sheet(output, manifest["scenes"], output / "preview.jpg")
    return manifest["scenes"]


def build(output):
    """Author a new collection once from pinned local inputs, without downloads."""
    if PIL.__version__ != BASE.AUTHORING_PILLOW_VERSION:
        raise ValueError(f"Authoring requires Pillow=={BASE.AUTHORING_PILLOW_VERSION}")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace existing camera fixtures in {output}")
    source_manifest = json.loads(BASE.SOURCE_MANIFEST.read_text())
    for source in source_manifest["cards"]:
        if BASE.digest(BASE.SOURCE_DIRECTORY / source["file"]) != source["sha256"]:
            raise ValueError(f"Changed or missing pinned PNG: {source['file']}")
    sources = BASE.load_sources()
    collection = recipes()
    output.mkdir(parents=True, exist_ok=True)
    (output / "recipes.json").write_text(json.dumps(collection, indent=2) + "\n")
    entries, hashes = [], {}
    for recipe in collection:
        entry, pins = write_scene(output, recipe, sources)
        entries.append(entry)
        hashes.update(pins)
        print(f"saved {entry['id']}", flush=True)
    manifest = {
        "format_version": 1, "references": {}, "sha256": hashes, "scenes": entries,
        "generator": {"script": "scripts/build-camera-fixtures.py", "seed": DEFAULT_SEED,
                      "pillow": PIL.__version__, "libjpeg": features.version("jpg"),
                      "projection": "shared physical pinhole camera applied to the entire coplanar scene",
                      "authoritative_inputs": "saved JPEG and final-JPEG appearance PNG bytes"},
        "sources": {Path(source["file"]).stem: source for source in source_manifest["cards"]},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    BASE.contact_sheet(output, entries, output / "preview.jpg")
    return entries


def main():
    """Require explicit authoring and refuse replacement of an existing corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path, default=FIXTURES)
    parser.add_argument("--append-continuous-surface", action="store_true",
                        help="append the 17 new solid-table controls without replacing existing scenes")
    options = parser.parse_args()
    if not options.write:
        parser.error("pass --write to author saved camera fixtures; tests never invoke this tool")
    entries = (append_continuous_surface(options.output) if options.append_continuous_surface
               else build(options.output))
    print(f"Saved {len(entries)} camera scenes in {options.output}")


if __name__ == "__main__":
    try:
        main()
    except (ImportError, OSError, ValueError, KeyError) as error:
        print(f"Camera authoring failed: {error}", file=sys.stderr)
        sys.exit(1)
