#!/usr/bin/env python3
"""Explicitly author saved playmat JPEGs from the reviewed recipe collection.

Tests never run this script. Source cards must already be cached as pinned PNGs.
Background artwork, recipes, annotations, references and final JPEGs live under
tests/fixtures/playmats. Refuse to overwrite an existing scene collection.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import PIL
from PIL import features


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures/playmats"
RENDERER = ROOT / "scripts/build-scene-fixtures.py"
SEED = 20261019


def load_renderer():
    """Reuse the explicit authoring renderer without running its CLI."""
    spec = importlib.util.spec_from_file_location("scene_authoring", RENDERER)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    return renderer


def build(output):
    """Write a new corpus from pinned inputs; never replace saved scene bytes."""
    renderer = load_renderer()
    if PIL.__version__ != renderer.AUTHORING_PILLOW_VERSION:
        raise ValueError(f"Authoring requires Pillow=={renderer.AUTHORING_PILLOW_VERSION}")
    if (output / "manifest.json").exists() or (output / "images").exists():
        raise FileExistsError(f"Refusing to replace saved scenes in {output}")
    recipes_path = FIXTURES / "recipes.json"
    recipes = json.loads(recipes_path.read_text())
    sources_path = ROOT / "tests/fixtures/scryfall/manifest.json"
    source_manifest = json.loads(sources_path.read_text())
    for source in source_manifest["cards"]:
        path = renderer.SOURCE_DIRECTORY / source["file"]
        if renderer.digest(path) != source["sha256"]:
            raise ValueError(f"Source PNG checksum mismatch: {path}")
    sources = renderer.load_sources()
    scenes, hashes = [], {}
    for recipe in recipes:
        entry, pins = renderer.write_scene(output, recipe, sources, seed=SEED)
        scenes.append(entry)
        hashes.update(pins)
        print(f"saved {entry['id']}", flush=True)
    background_paths = sorted({recipe["background"]["file"] for recipe in recipes})
    manifest = {
        "format_version": 1,
        "generator": {
            "script": "scripts/build-playmat-fixtures.py",
            "script_sha256": renderer.digest(__file__),
            "renderer_sha256": renderer.digest(RENDERER),
            "recipes": "tests/fixtures/playmats/recipes.json",
            "recipes_sha256": renderer.digest(recipes_path),
            "pillow": PIL.__version__, "libjpeg": features.version("jpg"),
            "seed": SEED, "size": list(renderer.SIZE),
            "background_fit": "centered LANCZOS fit; background print precedes cards",
            "authoritative_inputs": "saved JPEG and reference PNG bytes",
        },
        "sources": {Path(source["file"]).stem: source for source in source_manifest["cards"]},
        "backgrounds": {path: renderer.digest(ROOT / path) for path in background_paths},
        "references": {}, "sha256": hashes, "scenes": scenes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    renderer.contact_sheet(output, scenes, ROOT / "target/playmat-preview/contact-sheet.jpg")
    return scenes


def main():
    """Require an explicit write action and report the saved fixture location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path, default=FIXTURES)
    options = parser.parse_args()
    if not options.write:
        parser.error("pass --write to author saved fixtures; tests never invoke this tool")
    scenes = build(options.output)
    print(f"Saved {len(scenes)} scenes in {options.output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Playmat authoring failed: {error}", file=sys.stderr)
        sys.exit(1)
