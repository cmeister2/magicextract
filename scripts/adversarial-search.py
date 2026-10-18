#!/usr/bin/env python3
"""Search for extraction failures and preserve counterexamples as saved fixtures.

This is an offline authoring/search tool, never a prerequisite for cargo test.
Candidates use pinned PNG sources, seeded geometry, and a named stressor.
Camera tilt projects the whole scene; --camera-angles combines oblique views
with the other stressors. Search reports stay in the scratch output directory.
The Rust evaluator checks full outlines and crop fidelity against saved JPEGs.
"""

import argparse
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
MUTATIONS = ("dim", "blur", "motion", "edge-occlusion", "corner-occlusion", "glare", "shadow", "low-contrast", "camera-tilt")


def authoring_module():
    """Reuse the explicit scene author without downloading or altering fixtures."""
    specification = importlib.util.spec_from_file_location(
        "magicextract_scene_author", Path(__file__).with_name("build-scene-fixtures.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if module.PIL.__version__ != module.AUTHORING_PILLOW_VERSION:
        raise ValueError(f"Use Pillow=={module.AUTHORING_PILLOW_VERSION} for fixture authoring")
    for source in json.loads(module.SOURCE_MANIFEST.read_text())["cards"]:
        path = module.SOURCE_DIRECTORY / source["file"]
        if not path.exists() or module.digest(path) != source["sha256"]:
            raise ValueError(f"Missing or changed pinned PNG: {path}; run fetch-scryfall-fixtures.py")
    return module


@lru_cache(maxsize=1)
def camera_module():
    """Load the coherent whole-plane camera renderer only when a pose is needed."""
    specification = importlib.util.spec_from_file_location(
        "magicextract_camera_author", Path(__file__).with_name("build-camera-fixtures.py")
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_candidate(author, directory, recipe, sources, seed):
    """Persist camera views and ordinary views through their appropriate renderer."""
    renderer = camera_module() if "camera" in recipe else author
    return renderer.write_scene(directory, recipe, sources, seed=seed)


def candidate_recipe(author, seed, index, mutation, severity, camera_angles=False):
    """Vary capture conditions while preserving a known complete card outline."""
    rng = random.Random(seed * 1_000_003 + index)
    source = rng.choice(author.CARD_NAMES)
    width = rng.uniform(150.0, 215.0)
    degrees = rng.uniform(-65.0, 65.0)
    center = (rng.uniform(285.0, 355.0), rng.uniform(235.0, 285.0))
    corners = author.rotated_corners(center, width, degrees)
    recipe = {
        "id": f"seed-{seed}-case-{index:04d}-{mutation}",
        "tier": "challenge",
        "description": f"Seeded {mutation} counterexample search, severity {severity:.6f}.",
        "tags": ["adversarial", mutation],
        "background": {"kind": "solid", "color": list(rng.choice(
            [(75, 115, 90), (24, 60, 39), (210, 210, 205), (100, 65, 45)]
        ))},
        "cards": [{"source": source, "corners": corners}],
        "appearance_reference": True,
        "effects": [],
        "occluders": [],
        "jpeg_quality": 90,
        "search": {"seed": seed, "index": index, "mutation": mutation, "severity": severity,
                   "camera_angles": camera_angles},
    }
    if mutation == "camera-tilt" or camera_angles:
        focal_length = rng.uniform(600.0, 1200.0)
        recipe["camera"] = {
            "pitch_degrees": 65.0 * severity if mutation == "camera-tilt" else rng.uniform(15.0, 65.0),
            "azimuth_degrees": rng.uniform(0.0, 360.0),
            "roll_degrees": rng.uniform(-15.0, 15.0),
            "focal_length": focal_length, "distance": focal_length * 7.0 / 6.0,
        }
        recipe["table_color"] = list(recipe["background"]["color"])
        recipe["tags"] += ["camera", "coherent-plane", "continuous-surface"]
    if mutation == "dim":
        recipe["effects"] = [{"kind": "brightness", "factor": 1.0 - 0.80 * severity}]
    elif mutation == "blur":
        recipe["effects"] = [{"kind": "gaussian_blur", "radius": 2.6 * severity}]
    elif mutation == "motion":
        recipe["effects"] = [{"kind": "motion_blur", "length": 1 + 2 * round(7 * severity)}]
    elif mutation == "edge-occlusion":
        cx = (corners[1][0] + corners[2][0]) / 2
        cy = (corners[1][1] + corners[2][1]) / 2
        radius = width * 0.32 * severity
        if radius > 0:
            recipe["occluders"] = [{"kind": "ellipse", "bounds": [cx-radius, cy-radius, cx+radius, cy+radius],
                                     "color": [185, 137, 106]}]
    elif mutation == "corner-occlusion":
        cx, cy = corners[0]
        radius = width * 0.24 * severity
        if radius > 0:
            recipe["occluders"] = [{"kind": "ellipse", "bounds": [cx-radius, cy-radius, cx+radius, cy+radius],
                                     "color": [185, 137, 106]}]
    elif mutation == "glare":
        if severity > 0:
            cx, cy = corners[1]
            recipe["effects"] = [{"kind": "lighting", "glare": [cx, cy, width * 1.15 * severity]}]
    elif mutation == "shadow":
        recipe["effects"] = [{"kind": "lighting", "gradient": True,
                               "minimum": 1.0 - 0.88 * severity, "maximum": 1.0}]
    elif mutation == "low-contrast":
        recipe["effects"] = [{"kind": "contrast", "factor": 1.0 - 0.90 * severity}]
    elif mutation == "camera-tilt":
        pass
    else:
        raise ValueError(f"Unknown mutation: {mutation}")
    return recipe


def write_manifest(directory, entries, hashes, provenance):
    """Write annotations and byte hashes for actual, already-saved image files."""
    directory = Path(directory)
    path = directory / "manifest.json"
    manifest = {
        "format_version": 1, "references": {}, "sha256": hashes,
        "sources": provenance, "scenes": entries,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def evaluate(binary, manifest, output):
    """Accept scored failures as findings; distinguish infrastructure failures."""
    process = subprocess.run(
        [str(binary), "--manifest", str(manifest), "--output", str(output), "--include-challenges"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if process.returncode not in (0, 1) or not output.exists():
        raise RuntimeError(f"Evaluator could not score fixtures: {process.stderr.strip() or process.stdout.strip()}")
    result = json.loads(output.read_text())
    try:
        expected_ids = {scene["id"] for scene in json.loads(manifest.read_text())["scenes"]}
        scenes = result["scenes"]
        reported_ids = [scene["id"] for scene in scenes]
        valid = (
            len(reported_ids) == len(expected_ids)
            and set(reported_ids) == expected_ids
            and all(type(scene["passed"]) is bool
                    and isinstance(scene["failures"], list)
                    and all(isinstance(message, str) for message in scene["failures"])
                    and scene["passed"] == (not scene["failures"]) for scene in scenes)
            and result["summary"]["scenes"] == len(scenes)
            and result["summary"]["failed"] == sum(not scene["passed"] for scene in scenes)
            and (process.returncode == 1) == (result["summary"]["failed"] > 0)
        )
    except (KeyError, TypeError) as error:
        raise RuntimeError("Evaluator returned an incomplete or invalid report") from error
    if not scenes or not valid:
        raise RuntimeError("Evaluator report does not match the requested cases and exit status")
    return result


def single_probe(author, sources, provenance, binary, directory, recipe, seed):
    """Save and score one reduction attempt without changing any earlier JPEG."""
    entry, hashes = write_candidate(author, directory, recipe, sources, seed)
    manifest = write_manifest(directory, [entry], hashes, provenance)
    result = evaluate(binary, manifest, directory / "evaluation.json")
    return result["scenes"][0]


def reduce_failure(author, sources, provenance, binary, output, recipe, steps, seed):
    """Find a lower-severity failure; detection need not be globally monotonic."""
    search = recipe["search"]
    index, mutation = search["index"], search["mutation"]
    high, low = search["severity"], 0.0
    best = recipe
    probes = []
    for step in range(steps + 1):
        severity = 0.0 if step == 0 else (high + low) / 2
        candidate = candidate_recipe(author, seed, index, mutation, severity,
                                     camera_angles=search.get("camera_angles", False))
        candidate["id"] += f"-reduction-{step}"
        directory = output / "reductions" / candidate["id"]
        result = single_probe(author, sources, provenance, binary, directory, candidate, seed)
        probes.append({"severity": severity, "passed": result["passed"], "failures": result["failures"]})
        if result["passed"]:
            low = severity
        else:
            best, high = candidate, severity
            if severity == 0:
                break
    best = deepcopy(best)
    best["search"]["reduction_probes"] = probes
    best["search"]["initial_severity"] = search["severity"]
    best["search"]["reduction_method"] = "bounded severity bisection; not a proof of global minimality"
    return best


def select_failures(recipes, evaluations, limit):
    """Keep low-severity failures from different stressors before taking repeats."""
    by_id = {scene["id"]: scene for scene in evaluations}
    failing = [recipe for recipe in recipes if not by_id[recipe["id"]]["passed"]]
    failing.sort(key=lambda recipe: (recipe["search"]["severity"], recipe["id"]))
    selected = []
    covered = set()
    for recipe in failing:
        mutation = recipe["search"]["mutation"]
        if mutation not in covered:
            selected.append(recipe)
            covered.add(mutation)
            if len(selected) == limit:
                return selected
    for recipe in failing:
        if recipe not in selected:
            selected.append(recipe)
            if len(selected) == limit:
                break
    return selected


def run(options):
    """Execute a bounded local search and optionally archive reduced failures."""
    if options.cases < 1 or options.max_save < 1 or not 0 <= options.reduce_steps <= 8:
        raise ValueError("Use positive case/save counts and 0..8 reduction steps")
    options.output = options.output.resolve()
    if options.save_failures:
        options.save_failures = options.save_failures.resolve()
    if options.output.exists() and any(options.output.iterdir()):
        raise ValueError("Search output already exists; choose a new output directory")
    if options.save_failures and options.save_failures.exists() and any(options.save_failures.iterdir()):
        raise ValueError("Failure archive already exists; choose a new archive to preserve existing fixtures")
    author = authoring_module()
    sources = author.load_sources()
    provenance = {Path(card["file"]).stem: card for card in json.loads(author.SOURCE_MANIFEST.read_text())["cards"]}
    if options.evaluator:
        binary = options.evaluator.resolve()
    else:
        subprocess.run(["cargo", "build", "--locked", "--offline", "--release", "--example", "evaluate_scenes"],
                       cwd=ROOT, check=True)
        binary = ROOT / "target/release/examples/evaluate_scenes"
    if not binary.is_file():
        raise ValueError(f"Evaluator executable does not exist: {binary}")
    options.output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(options.seed)
    recipes, entries, hashes = [], [], {}
    for index in range(options.cases):
        mutation = MUTATIONS[index % len(MUTATIONS)]
        severity = rng.uniform(0.15, 1.0)
        recipe = candidate_recipe(author, options.seed, index, mutation, severity,
                                  camera_angles=options.camera_angles)
        entry, saved_hashes = write_candidate(author, options.output, recipe, sources, options.seed)
        recipes.append(recipe)
        entries.append(entry)
        hashes.update(saved_hashes)
        if index % 12 == 0 or index + 1 == options.cases:
            print(f"Saved candidate {index + 1}/{options.cases}", flush=True)
    manifest = write_manifest(options.output, entries, hashes, provenance)
    evaluation_path = options.output / "evaluation.json"
    evaluation = evaluate(binary, manifest, evaluation_path)
    if len(evaluation["scenes"]) != options.cases:
        raise RuntimeError("Evaluator did not score every candidate")
    selected = select_failures(recipes, evaluation["scenes"], options.max_save)
    reductions = []
    for recipe in selected:
        print(f"Reducing {recipe['id']}", flush=True)
        reductions.append(reduce_failure(author, sources, provenance, binary, options.output,
                                         recipe, options.reduce_steps, options.seed))
    archive_result = None
    if options.save_failures and reductions:
        options.save_failures.mkdir(parents=True, exist_ok=True)
        archive_entries, archive_hashes = [], {}
        for recipe in reductions:
            entry, saved_hashes = write_candidate(author, options.save_failures, recipe, sources, options.seed)
            entry["finding"] = recipe["search"]
            archive_entries.append(entry)
            archive_hashes.update(saved_hashes)
        archive_manifest = write_manifest(options.save_failures, archive_entries, archive_hashes, provenance)
        archive_result = evaluate(binary, archive_manifest, options.output / "archived-evaluation.json")
        if archive_result["summary"]["failed"] != len(archive_entries):
            raise RuntimeError("A saved counterexample no longer reproduces its failure; inspect the archived evaluation")
    report = {
        "seed": options.seed, "candidates": options.cases,
        "camera_angles": options.camera_angles,
        "mutations": list(MUTATIONS), "summary": evaluation["summary"],
        "evaluation": str(evaluation_path), "saved_counterexamples": len(reductions) if options.save_failures else 0,
        "archive": str(options.save_failures) if options.save_failures and reductions else None,
        "archive_summary": archive_result["summary"] if archive_result else None,
        "evaluator_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "pillow": author.PIL.__version__,
        "authoring_script_sha256": hashlib.sha256(Path(author.__file__).read_bytes()).hexdigest(),
        "camera_authoring_script_sha256": hashlib.sha256(
            Path(__file__).with_name("build-camera-fixtures.py").read_bytes()).hexdigest(),
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((ROOT / "src").glob("*.rs"))},
    }
    (options.output / "search-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if options.save_failures and reductions:
        (options.output / "discovery.json").write_text(json.dumps({
            "search": report, "observed_failures": archive_result["scenes"],
        }, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Search report: {options.output / 'search-report.json'}", flush=True)
    return 1 if report["summary"]["failed"] else 0


def main():
    """Keep search cost explicit and exit nonzero when counterexamples are found."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--cases", type=int, default=96)
    parser.add_argument("--output", type=Path, default=ROOT / "target/adversarial/seed-20261018")
    parser.add_argument("--save-failures", type=Path)
    parser.add_argument("--max-save", type=int, default=8)
    parser.add_argument("--reduce-steps", type=int, default=3)
    parser.add_argument("--evaluator", type=Path, help="use a prebuilt evaluator instead of cargo build")
    parser.add_argument("--camera-angles", action="store_true",
                        help="combine every stressor with a seeded 15–65 degree oblique camera view")
    options = parser.parse_args()
    try:
        return run(options)
    except (ImportError, KeyError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Adversarial search failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
