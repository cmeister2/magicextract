#!/usr/bin/env python3
"""Explicitly author saved border and art-treatment scenes from pinned PNGs.

Tests read saved JPEGs and references; they never call this tool or use its cache.
Run with --write and --sources pointing to the two original cached Scryfall PNGs.
Replacing an existing collection additionally requires --force.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import PIL
from PIL import Image, features


ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests/fixtures"
CORPORA = ("borders", "art-treatments")
RENDERER = ROOT / "scripts/build-scene-fixtures.py"
SEED = 20261019


def load_renderer():
    """Load the shared explicit renderer without running its command line."""
    spec = importlib.util.spec_from_file_location("border_scene_authoring", RENDERER)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    return renderer


def load_sources(directory, records, renderer):
    """Verify each pinned original PNG before decoding its preserved alpha."""
    sources = {}
    for record in records:
        filename = record["file"]
        if Path(filename).name != filename or Path(filename).suffix != ".png":
            raise ValueError(f"Expected a single cached PNG filename: {filename}")
        path = directory / filename
        if renderer.digest(path) != record["sha256"]:
            raise ValueError(f"Cached source PNG checksum mismatch: {path}")
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected an original PNG source: {path}")
            sources[record["key"]] = image.convert("RGBA")
    return sources


def build(output, source_directory, force=False, corpus="borders"):
    """Author a complete saved corpus after validating all pinned local inputs."""
    if corpus not in CORPORA:
        raise ValueError(f"Unknown saved corpus: {corpus}")
    renderer = load_renderer()
    if PIL.__version__ != renderer.AUTHORING_PILLOW_VERSION:
        raise ValueError(f"Authoring requires Pillow=={renderer.AUTHORING_PILLOW_VERSION}")
    existing = {path.name for path in output.iterdir()} if output.exists() else set()
    if existing - {"recipes.json", "sources.json"} and not force:
        raise FileExistsError(f"Refusing to replace {output}; pass --force intentionally")
    recipes_path = FIXTURES / corpus / "recipes.json"
    source_manifest_path = FIXTURES / corpus / "sources.json"
    recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
    records = json.loads(source_manifest_path.read_text(encoding="utf-8"))["cards"]
    sources = load_sources(source_directory, records, renderer)
    backgrounds = {}
    for recipe in recipes:
        background = recipe["background"]
        if background["kind"] == "image":
            filename = background["file"]
            actual = renderer.digest(ROOT / filename)
            if actual != background["sha256"]:
                raise ValueError(f"Background checksum mismatch: {filename}")
            backgrounds[filename] = actual
    scenes, hashes = [], {}
    for recipe in recipes:
        entry, pins = renderer.write_scene(output, recipe, sources, seed=SEED, overwrite=force)
        scenes.append(entry)
        hashes.update(pins)
        print(f"saved {entry['id']}", flush=True)
    manifest = {
        "format_version": 1,
        "generator": {
            "script": "scripts/build-border-fixtures.py",
            "script_sha256": renderer.digest(__file__),
            "renderer": "scripts/build-scene-fixtures.py",
            "renderer_sha256": renderer.digest(RENDERER),
            "recipes": str(recipes_path.relative_to(ROOT)),
            "recipes_sha256": renderer.digest(recipes_path),
            "source_manifest": str(source_manifest_path.relative_to(ROOT)),
            "source_manifest_sha256": renderer.digest(source_manifest_path),
            "pillow": PIL.__version__,
            "libjpeg": features.version("jpg"),
            "seed": SEED,
            "size": list(renderer.SIZE),
            "jpeg_quality": 94,
            "jpeg_subsampling": 0,
            "source_alpha": "preserved",
            "appearance_references": "independent rectifications of decoded saved JPEGs using known physical card corners",
            "authoritative_inputs": "saved JPEG and reference PNG bytes",
            "limits": "Generated planar scenes and simple bright sleeve outlines; not independent camera capture, plastic optics, or metallic reflections."
            + (" Silver identifies the printed card border." if corpus == "borders" else ""),
        },
        "sources": {record["key"]: record for record in records},
        "backgrounds": backgrounds,
        "references": {},
        "sha256": hashes,
        "scenes": scenes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return scenes


def main():
    """Require explicit writing and keep source caches outside saved test inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="explicitly authorize fixture creation")
    parser.add_argument("--force", action="store_true", help="intentionally replace an existing corpus")
    parser.add_argument("--corpus", choices=CORPORA, default="borders")
    parser.add_argument("--output", type=Path, help="destination; defaults to the selected saved corpus")
    parser.add_argument("--sources", type=Path, help="directory containing the two pinned original PNGs")
    options = parser.parse_args()
    if not options.write:
        parser.error("pass --write to author saved fixtures; tests never invoke this tool")
    if options.sources is None:
        parser.error("pass --sources with the existing offline PNG cache directory")
    output = options.output if options.output is not None else FIXTURES / options.corpus
    scenes = build(output, options.sources, force=options.force, corpus=options.corpus)
    print(f"Saved {len(scenes)} scenes; manifest: {output / 'manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Border fixture authoring failed: {error}", file=sys.stderr)
        sys.exit(1)
