#!/usr/bin/env python3
"""Draw actual production detections/crops alongside archived benchmark evidence.

Run from any directory: python3 benchmarks/2026-10-20/render-detection-examples.py
Requires the locked Rust toolchain/dependencies, Pillow, and DejaVu Sans fonts.
No production source or measured benchmark source is modified. Source, fixture,
and regenerated detection checks refuse stale or mismatched evidence.
"""

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTPUT = HERE / "images"
WORK = ROOT / "target/performance-pr"
REPORT = HERE / "domain-final.json.gz"
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
BG = "#f3f5f8"
INK = "#142234"
MUTED = "#536478"
TRUTH = "#39efad"
ACTUAL = "#ff4fba"
ROI = "#ffd35c"
GREEN = "#087b55"
RED = "#b43b46"


def digest(path):
    """Hash exact source, input, or output bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments):
    """Run a checked build/helper command from the checkout root."""
    return subprocess.check_output(arguments, cwd=ROOT, text=True)


def font(size, bold=False, mono=False):
    """Load the explicit, widely available font used for reproducible labels."""
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    return ImageFont.truetype(str(FONT_ROOT / f"{name}{'-Bold' if bold else ''}.ttf"), size)


def label(draw, position, text, size=20, fill=INK, bold=False, mono=False):
    """Draw one line of human-readable evidence."""
    draw.text(position, text, font=font(size, bold, mono), fill=fill)


def wrapped(draw, position, text, width, size=20, fill=MUTED, spacing=8):
    """Wrap prose by measured glyph width."""
    x, y = position
    lines = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(candidate, font=font(size)) > width:
            lines.append(word)
        elif lines:
            lines[-1] = candidate
        else:
            lines.append(word)
    for line in lines:
        label(draw, (x, y), line, size, fill)
        y += size + spacing
    return y


def build_helper():
    """Resolve the exact Cargo library artifact and compile the small public-API caller."""
    output = run([
        "cargo", "build", "--release", "--locked", "--example", "benchmark",
        "--message-format=json",
    ])
    artifacts = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    library = next(
        Path(filename)
        for artifact in artifacts
        if artifact.get("reason") == "compiler-artifact"
        and artifact["target"]["name"] == "magicextract"
        for filename in artifact["filenames"]
        if filename.endswith(".rlib")
    )
    helper = WORK / "extract-for-visuals"
    run([
        "rustc", "--edition=2024", "-O", str(HERE / "extract-for-visuals.rs"),
        "-L", f"dependency={library.parent}", "--extern", f"magicextract={library}",
        "-o", str(helper),
    ])
    return helper


def prepare_case(report, helper, scene_id, variant_name):
    """Extract a case and prove it matches the exact archived benchmark geometry."""
    variant = next(item for item in report["variants"] if item["name"] == variant_name)
    result = next(item for item in variant["scenes"] if item["id"] == scene_id)
    manifest_path = ROOT / result["manifest"]
    manifest = json.loads(manifest_path.read_text())
    scene = next(item for item in manifest["scenes"] if item["id"] == scene_id)
    source = manifest_path.parent / scene["file"]
    if digest(source) != manifest["sha256"][scene["file"]]:
        raise ValueError(f"Fixture checksum mismatch: {source}")
    settings = variant["settings"]
    crop_directory = WORK / scene_id / variant_name
    detections = json.loads(run([
        str(helper), str(source), str(crop_directory), settings["detection_mode"],
        str(settings["max_detection_dimension"]),
        "roi" if result["detection_region"] else "frame",
    ]))
    archived = result["accuracy"]["detections"]
    if len(detections) != len(archived):
        raise ValueError(f"Detection count differs from archive: {scene_id}/{variant_name}")
    for actual, expected in zip(detections, archived):
        for point, reference in zip(actual["corners"], expected["corners"]):
            if any(abs(a - b) > 0.0001 for a, b in zip(point, reference)):
                raise ValueError(f"Corner differs from archive: {scene_id}/{variant_name}")
    if not result["domain_eligible"]:
        raise ValueError(f"Selected case is outside the domain: {scene_id}/{variant_name}")
    crops = [crop_directory / f"card-{index:03}.png" for index in range(len(detections))]
    return {
        "scene_id": scene_id,
        "variant": variant_name,
        "source": source,
        "scene": scene,
        "result": result,
        "detections": detections,
        "crops": crops,
    }


def dashed(draw, points, fill, width=3, dash=12):
    """Draw a dashed annotation without obscuring a nearby detected outline."""
    for start, end in zip(points, points[1:] + points[:1]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = (dx * dx + dy * dy) ** 0.5
        for offset in range(0, max(1, int(length)), dash * 2):
            a, b = offset / length, min(offset + dash, length) / length
            draw.line(
                [(start[0] + dx * a, start[1] + dy * a),
                 (start[0] + dx * b, start[1] + dy * b)],
                fill=fill, width=width,
            )


def annotated_frame(case, width):
    """Resize only for display, then overlay original-pixel annotations."""
    frame = Image.open(case["source"]).convert("RGB")
    scale = width / frame.width
    frame = frame.resize((width, round(frame.height * scale)), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(frame)
    region = case["result"]["detection_region"]
    if region:
        x, y = region["x"] * scale, region["y"] * scale
        right, bottom = x + region["width"] * scale, y + region["height"] * scale
        dashed(draw, [(x, y), (right, y), (right, bottom), (x, bottom)], ROI, 3, 9)
    for detection in case["detections"]:
        points = [(x * scale, y * scale) for x, y in detection["corners"]]
        draw.line(points + points[:1], fill="#172133", width=7)
        draw.line(points + points[:1], fill=ACTUAL, width=3)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=ACTUAL, outline="#172133")
    for card in case["scene"]["cards"]:
        points = [(x * scale, y * scale) for x, y in card["corners"]]
        dashed(draw, points, TRUTH, 2, 6)
    return frame


def header(image, title, subtitle):
    """Add the title and consistent annotation key to a figure."""
    draw = ImageDraw.Draw(image)
    label(draw, (38, 28), title, 34, bold=True)
    label(draw, (40, 78), subtitle, 18, MUTED)
    draw.rounded_rectangle((38, 116, 1242, 164), radius=10, fill="#233147")
    draw.line([(58, 140), (93, 140)], fill=ACTUAL, width=3)
    label(draw, (105, 127), "Returned outline", 18, "#ffffff")
    dashed(draw, [(340, 140), (375, 140)], TRUTH, 3, 7)
    label(draw, (387, 127), "Annotated physical card", 18, "#ffffff")
    dashed(draw, [(708, 140), (743, 140)], ROI, 3, 7)
    label(draw, (755, 127), "Fixed ROI: central 80% of each axis", 18, "#ffffff")


def status(case):
    """Return evidence labels from the archived scorer, not visual judgment."""
    accuracy = case["result"]["accuracy"]
    if accuracy["passed"]:
        return "PASS", GREEN
    return ("NO CROP" if accuracy["detected_count"] == 0 else "WRONG CROP"), RED


def metrics(draw, position, case, width):
    """Show the stored timing and scoring results without rerunning timings."""
    x, y = position
    result = case["result"]
    text, color = status(case)
    label(draw, (x, y), text, 22, color, bold=True)
    label(draw, (x, y + 34), f"{result['extract_ms']['median']:.2f} ms extraction", 22, bold=True)
    matches = result["accuracy"]["matches"]
    if matches:
        error = max(match["max_corner_error"] for match in matches)
        wrapped(draw, (x, y + 70), f"Max corner error: {error:.2f} px (limit: 8 px)", width, 17)
    else:
        label(draw, (x, y + 70), "0 of 1 annotated cards returned", 17, MUTED)


def draw_success(cases):
    """Compare two successful production configurations on one saved input."""
    image = Image.new("RGB", (1280, 892), BG)
    header(
        image, "Perspective extraction with a fixed camera window",
        "Saved synthetic scene: camera-table-p35-a000.jpg  ·  640 × 520 input",
    )
    draw = ImageDraw.Draw(image)
    for index, (case, title) in enumerate(zip(cases, ["Full 640", "Fast 480 + ROI"])):
        x = 38 + index * 620
        draw.rounded_rectangle((x, 188, x + 584, 736), radius=16, fill="white", outline="#dbe2eb")
        label(draw, (x + 20, 209), title, 28, bold=True)
        label(draw, (x + 20, 251), "Detected corners in the source frame", 16, MUTED)
        label(draw, (x + 394, 251), "Actual crop", 16, MUTED)
        image.paste(annotated_frame(case, 354), (x + 20, 288))
        crop = Image.open(case["crops"][0]).convert("RGB").resize((170, 238), Image.Resampling.LANCZOS)
        image.paste(crop, (x + 394, 311))
        metrics(draw, (x + 20, 600), case, 540)
    wrapped(
        draw, (40, 758),
        "Both modes return original-image corners and rectify from the original pixels. "
        "ROI = (64, 52, 512, 416); no angle restriction. Crops shown are the library's "
        "250 × 350 RGB outputs, resized only for display.",
        1200, 19,
    )
    label(
        draw, (40, 856),
        "Per-scene medians of 3 release calls on Ryzen 7 PRO 7840U; not Android phone timings.",
        17, MUTED,
    )
    return image


def draw_tradeoffs(cases):
    """Show actual failure output instead of implying optional modes are equivalent."""
    image = Image.new("RGB", (1280, 1110), BG)
    header(
        image, "Faster modes can lose the physical card boundary",
        "Saved synthetic challenge: extended-art-playmat.jpg  ·  identical input in every column",
    )
    draw = ImageDraw.Draw(image)
    labels = ["Full 640", "Full 640 + ROI", "Fast 480"]
    for index, (case, title) in enumerate(zip(cases, labels)):
        x = 38 + index * 410
        draw.rounded_rectangle((x, 188, x + 384, 974), radius=16, fill="white", outline="#dbe2eb")
        label(draw, (x + 18, 211), title, 26, bold=True)
        image.paste(annotated_frame(case, 348), (x + 18, 258))
        label(draw, (x + 18, 562), "Actual returned crop", 17, MUTED)
        if case["crops"]:
            crop = Image.open(case["crops"][0]).convert("RGB").resize((164, 230), Image.Resampling.LANCZOS)
            image.paste(crop, (x + 110, 599))
        else:
            draw.rounded_rectangle((x + 110, 599, x + 274, 829), radius=8, fill="#f1f4f8", outline="#dbe2eb")
            label(draw, (x + 136, 693), "No crop", 22, MUTED)
        metrics(draw, (x + 18, 850), case, 348)
    wrapped(
        draw, (40, 997),
        "Full recovers the exterior. ROI selects a background patch; Fast misses this card. "
        "These are measured accuracy tradeoffs: Full passes 111/111 scenes, Full + ROI "
        "102/107 eligible scenes, and Fast 480 99/111.",
        1200, 19,
    )
    label(
        draw, (40, 1073),
        "Per-scene extraction medians: 3 release calls, Ryzen 7 PRO 7840U. Synthetic corpus; not handset validation.",
        16, MUTED,
    )
    return image


def main():
    """Regenerate both pictures, real crops, and a compact provenance record."""
    report = json.load(gzip.open(REPORT))
    for relative, expected in report["metadata"]["compiled_source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Measured source changed; regenerate benchmark before rendering: {relative}")
    OUTPUT.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    helper = build_helper()
    success = [
        prepare_case(report, helper, "camera-table-p35-a000", variant)
        for variant in ["full-640", "fast-480-roi"]
    ]
    tradeoffs = [
        prepare_case(report, helper, "extended-art-playmat", variant)
        for variant in ["full-640", "full-640-roi", "fast-480"]
    ]
    paths = [OUTPUT / "detection-examples.png", OUTPUT / "accuracy-tradeoffs.png"]
    draw_success(success).save(paths[0], optimize=True)
    draw_tradeoffs(tradeoffs).save(paths[1], optimize=True)
    evidence = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "generator_sha256": digest(Path(__file__)),
        "helper_sha256": digest(HERE / "extract-for-visuals.rs"),
        "benchmark": str(REPORT.relative_to(ROOT)),
        "benchmark_sha256": digest(REPORT),
        "source_sha256": report["metadata"]["compiled_source_sha256"],
        "limits": "Saved synthetic scenes. Desktop timings, no phone timing claim. "
                  "No retouched crops: production extract() output, resized only for display.",
        "figures_sha256": {path.name: digest(path) for path in paths},
        "cases": [{
            "scene_id": case["scene_id"],
            "variant": case["variant"],
            "source": str(case["source"].relative_to(ROOT)),
            "source_sha256": digest(case["source"]),
            "detections": case["detections"],
            "crops_sha256": [digest(path) for path in case["crops"]],
            "archived_result": case["result"],
        } for case in success + tradeoffs],
    }
    (OUTPUT / "detection-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print("\n".join(str(path.relative_to(ROOT)) for path in paths))


if __name__ == "__main__":
    main()
