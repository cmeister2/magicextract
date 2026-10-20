#!/usr/bin/env python3
"""Build and run the known-corner crop benchmark using locked existing dependencies.

Use --build-only to prepare without timing; --run-only refuses changed sources or
binary bytes. No Cargo manifest, production file, or fixture is modified.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WORK = ROOT / "target" / "rectification-benchmark"
BINARY = WORK / "crop_benchmark"
BUILD = WORK / "build.json"
SCENES = [
    ("scenes", "baseline-lightning-bolt-background-green"),
    ("playmats", "playmat-fantasy-forest-lightning-bolt"),
    ("camera", "camera-forest-normal-control"),
]
SOURCES = [
    *sorted(ROOT.glob("src/*.rs")),
    ROOT / "Cargo.toml",
    ROOT / "Cargo.lock",
    HERE / "crop_benchmark.rs",
    HERE / "run.py",
]


def digest(path):
    """Hash exact bytes, including sources and compiled artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    """Identify the complete library and benchmark source state."""
    return {str(path.relative_to(ROOT)): digest(path) for path in SOURCES}


def command(arguments):
    """Run a command from the checkout without shell interpretation."""
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()


def build():
    """Resolve locked Cargo artifacts and build the standalone optimized helper."""
    WORK.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    cargo_command = [
        "cargo", "build", "--release", "--locked", "--example", "benchmark",
        "--message-format=json",
    ]
    artifacts = [
        json.loads(line)
        for line in command(cargo_command).splitlines()
        if line.startswith("{")
    ]
    library = next(
        Path(filename)
        for artifact in artifacts
        if artifact.get("reason") == "compiler-artifact"
        and artifact["target"]["name"] == "magicextract"
        for filename in artifact["filenames"]
        if filename.endswith(".rlib")
    )
    rustc_command = [
        "rustc", "--edition=2024", "-C", "opt-level=3",
        str(HERE / "crop_benchmark.rs"),
        "-L", f"dependency={library.parent}",
        "--extern", f"magicextract={library}",
        "-o", str(BINARY),
    ]
    subprocess.run(rustc_command, cwd=ROOT, check=True)
    if before != source_hashes():
        raise RuntimeError("source files changed during compilation; rebuild")
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "compiled_source_sha256": before,
        "executable_sha256": digest(BINARY),
        "library_sha256": digest(library),
        "cargo_command": cargo_command,
        "rustc_command": rustc_command,
        "rustc": command(["rustc", "-Vv"]),
        "cargo": command(["cargo", "-V"]),
        "build_environment": {
            key: value for key, value in os.environ.items()
            if key in {"RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTUP_TOOLCHAIN"}
            or key.startswith("CARGO_PROFILE_RELEASE_")
        },
    }
    BUILD.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Built {BINARY}; no timing has run.")


def summarize(samples):
    """Use the median and nearest-rank p95 for the retained per-call samples."""
    ordered = sorted(samples)
    return {
        "calls": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[math.ceil(len(samples) * 0.95) - 1],
        "max_ms": ordered[-1],
    }


def measure(output):
    """Verify fixtures, execute the prepared helper, and save all raw evidence."""
    metadata = json.loads(BUILD.read_text())
    if metadata["compiled_source_sha256"] != source_hashes():
        raise RuntimeError("sources changed after compilation; rebuild before timing")
    if metadata["executable_sha256"] != digest(BINARY):
        raise RuntimeError("benchmark executable changed after compilation")
    fixtures = []
    paths = []
    for collection, identifier in SCENES:
        manifest_path = ROOT / "tests" / "fixtures" / collection / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        scene = next(scene for scene in manifest["scenes"] if scene["id"] == identifier)
        image_path = manifest_path.parent / scene["file"]
        actual_hash = digest(image_path)
        if actual_hash != manifest["sha256"][scene["file"]]:
            raise RuntimeError(f"fixture integrity check failed: {image_path}")
        fixtures.append({
            "id": identifier,
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": digest(manifest_path),
            "image": str(image_path.relative_to(ROOT)),
            "image_sha256": actual_hash,
        })
        paths.append(str(image_path))
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next(
        (line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
         if line.startswith("model name")),
        platform.processor(),
    ) if cpuinfo.exists() else platform.processor()
    metadata.update({
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "command": [str(BINARY), *paths],
    })
    samples = json.loads(command([str(BINARY), *paths]))
    if len(samples) != len(fixtures):
        raise RuntimeError("helper returned an unexpected number of scene results")
    for fixture, sample in zip(fixtures, samples):
        if len(sample["extract_ms"]) != 31 or len(sample["rectify_ms"]) != 31:
            raise RuntimeError("helper did not return 31 calls per operation")
        fixture.update(sample)
        fixture["extract"] = summarize(sample["extract_ms"])
        fixture["rectify"] = summarize(sample["rectify_ms"])
        fixture["median_ratio_extract_to_rectify"] = (
            fixture["extract"]["median_ms"] / fixture["rectify"]["median_ms"]
        )
    report = {
        "format_version": 1,
        "metadata": metadata,
        "methodology": {
            "platform_scope": "Desktop host only; no Android device measurement",
            "extract": "Full mode, maximum detection dimension 640, RGB8 input, output 250x350",
            "rectify": "Known source-pixel corners; validation, allocation, and RGB warp included",
            "calls_per_operation_per_scene": 31,
            "warmups_per_operation_per_scene": 5,
            "order": "Alternate extract/rectify first each call; invert first operation for odd scene index",
            "pixels_verified": "Detect once before timing; rectify output exactly equals extract output",
            "excluded": [
                "fixture I/O and integrity checks", "image decoding", "input RGB8 conversion",
                "crop verification", "output serialization and destruction",
                "camera capture and display", "OCR", "tracking",
                "detection needed to obtain corners for rectify",
            ],
            "p95": "Nearest rank: sorted sample at ceil(0.95 * count)-1",
        },
        "scenes": fixtures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    for scene in fixtures:
        print(scene["id"])
        for operation in ("extract", "rectify"):
            timing = scene[operation]
            print(
                f"  {operation}: median {timing['median_ms']:.3f} ms, "
                f"p95 {timing['p95_ms']:.3f} ms, max {timing['max_ms']:.3f} ms"
            )
    print(f"Saved {output}")


def main():
    """Separate compilation from timing so callers can reserve a quiet CPU window."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    parser.add_argument("--output", type=Path, default=WORK / "results.json")
    options = parser.parse_args()
    if not options.run_only:
        build()
    if not options.build_only:
        measure(options.output.resolve())


if __name__ == "__main__":
    main()
