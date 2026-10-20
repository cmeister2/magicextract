#!/usr/bin/env python3
"""Render actual known-corner crops and their source outline for the PR.

Run from any directory: python3 benchmarks/rectification/render-example.py
Requires Python/Pillow, Cargo, and Rust. The Rust helper calls only the public
rectify API, using recorded source-pixel corners from results-desktop.json.
PNG crops are pasted at native resolution; only the source outline and labels
are drawn. No detection, timing, AI generation, or crop resampling is performed.
"""

import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
WORK = ROOT / "target" / "rectification-example"
SCENE_ID = "playmat-fantasy-forest-lightning-bolt"
REGULAR_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def digest(path):
    """Hash exact file bytes for artifact provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(arguments):
    """Run an explicit argument vector without shell interpretation."""
    return subprocess.check_output(arguments, cwd=ROOT, text=True).strip()


def font(size, bold=False):
    """Use a fixed font family whose exact bytes are recorded below."""
    return ImageFont.truetype(str(BOLD_FONT if bold else REGULAR_FONT), size)


def source_hashes():
    """Identify the library and renderer that produced the visual."""
    paths = [
        *sorted(ROOT.glob("src/*.rs")),
        ROOT / "Cargo.toml",
        ROOT / "Cargo.lock",
        HERE / "render_example.rs",
        HERE / "render-example.py",
    ]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def main():
    """Verify the saved input, execute the API, and compose one evidence image."""
    WORK.mkdir(parents=True, exist_ok=True)
    report_path = HERE / "results-desktop.json"
    report = json.loads(report_path.read_text())
    scene = next(scene for scene in report["scenes"] if scene["id"] == SCENE_ID)
    fixture = ROOT / scene["image"]
    if digest(fixture) != scene["image_sha256"]:
        raise RuntimeError("fixture does not match the recorded image checksum")
    corners = scene["corners"]
    before = source_hashes()
    cargo_command = [
        "cargo", "build", "--release", "--locked", "--example", "benchmark",
        "--message-format=json",
    ]
    events = [json.loads(line) for line in command(cargo_command).splitlines()]
    library = next(
        Path(filename)
        for event in events
        if event.get("reason") == "compiler-artifact"
        and event["target"]["name"] == "magicextract"
        for filename in event["filenames"]
        if filename.endswith(".rlib")
    )
    binary = WORK / "render_example"
    rustc_command = [
        "rustc", "--edition=2024", "-C", "opt-level=3",
        str(HERE / "render_example.rs"),
        "-L", f"dependency={library.parent}",
        "--extern", f"magicextract={library}",
        "-o", str(binary),
    ]
    subprocess.run(rustc_command, cwd=ROOT, check=True)
    if before != source_hashes():
        raise RuntimeError("sources changed during compilation; render again")
    invocation = [
        str(binary), str(fixture), str(WORK),
        *(str(value) for point in corners for value in point),
    ]
    verification = json.loads(command(invocation))
    decoded = {
        name: Image.open(WORK / name).convert("RGB")
        for name in ["source.png", "crop.png", "crop-rotated.png"]
    }
    if decoded["source.png"].size != (640, 520):
        raise RuntimeError("the evidence layout expects the saved 640 x 520 fixture")
    if any(decoded[name].size != (250, 350) for name in ["crop.png", "crop-rotated.png"]):
        raise RuntimeError("unexpected public-API crop dimensions")

    background = "#f4f6f8"
    ink = "#172b40"
    muted = "#52667a"
    accent = "#00d5b3"
    board = Image.new("RGB", (1262, 772), background)
    draw = ImageDraw.Draw(board)
    draw.text((28, 24), "Rectify a known outline", font=font(32, True), fill=ink)
    draw.text(
        (28, 73),
        "Original source pixels to an RGB crop. The caller supplies corners; detection is skipped.",
        font=font(18), fill=muted,
    )
    columns = [
        (28, "Supplied outline", "640 x 520 source | original-pixel corners"),
        (700, "Preserve supplied order", "[0, 1, 2, 3] | 250 x 350"),
        (982, "Choose a 180-degree turn", "[2, 3, 0, 1] | 250 x 350"),
    ]
    for x, heading, subtitle in columns:
        draw.text((x, 120), heading, font=font(17, True), fill=ink)
        draw.text((x, 148), subtitle, font=font(14), fill=muted)
    top = 180
    board.paste(decoded["source.png"], (28, top))
    board.paste(decoded["crop.png"], (700, top))
    board.paste(decoded["crop-rotated.png"], (982, top))
    points = [(28 + x, top + y) for x, y in corners]
    draw.line(points + [points[0]], fill=ink, width=6)
    draw.line(points + [points[0]], fill=accent, width=3)
    for index, (x, y) in enumerate(points):
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=ink, outline=accent, width=2)
        draw.text((x, y - 1), str(index), font=font(15, True), fill="white", anchor="mm")
    for x, width, height in [(28, 640, 520), (700, 250, 350), (982, 250, 350)]:
        draw.rectangle((x - 1, top - 1, x + width, top + height), outline="#c5ced8")
    draw.multiline_text(
        (700, 557),
        "Samples the source image\nwith bilinear interpolation.\nNo detection pass.",
        font=font(16), fill=muted, spacing=10,
    )
    draw.multiline_text(
        (982, 557),
        "Same source coordinates.\nChanging the start corner\nchooses a 180-degree turn.",
        font=font(16), fill=muted, spacing=10,
    )
    draw.line((28, 725, 1234, 725), fill="#d4dce4", width=1)
    draw.text(
        (28, 742),
        "Saved synthetic fixture | actual public-API outputs | crop pixels shown at native resolution",
        font=font(14), fill=muted,
    )
    artifact = HERE / "example.png"
    board.save(artifact)

    provenance = {
        "format_version": 1,
        "scene": SCENE_ID,
        "fixture": scene["image"],
        "fixture_sha256": digest(fixture),
        "corner_source": str(report_path.relative_to(ROOT)),
        "corner_source_sha256": digest(report_path),
        "corners_original_image_pixels": corners,
        "first_order": [0, 1, 2, 3],
        "second_order": [2, 3, 0, 1],
        "method": "Two actual CardExtractor::rectify calls; no detect call or timing",
        "composition": "Native-size source and crops; numbered outline drawn only on the source",
        "verification": verification,
        "compiled_source_sha256": before,
        "library_sha256": digest(library),
        "executable_sha256": digest(binary),
        "rustc": command(["rustc", "-Vv"]),
        "cargo_command": cargo_command,
        "rustc_command": rustc_command,
        "invocation": invocation,
        "pillow_version": pillow_version,
        "font_sha256": {str(path): digest(path) for path in [REGULAR_FONT, BOLD_FONT]},
        "api_output_sha256": {
            name: {
                "png_file": digest(WORK / name),
                "rgb_pixels": hashlib.sha256(image.tobytes()).hexdigest(),
                "dimensions": list(image.size),
            }
            for name, image in decoded.items()
        },
        "artifact": str(artifact.relative_to(ROOT)),
        "artifact_sha256": digest(artifact),
    }
    (HERE / "example-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Rendered {artifact}")
    print(json.dumps(verification))


if __name__ == "__main__":
    main()
