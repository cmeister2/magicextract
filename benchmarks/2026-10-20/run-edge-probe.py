#!/usr/bin/env python3
"""Compile the standalone edge probe against this checkout's locked dependencies."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
artifacts = subprocess.check_output(
    ["cargo", "build", "--locked", "--release", "--example", "evaluate_scenes", "--message-format=json"],
    cwd=root,
    text=True,
)
libraries = {}
for line in artifacts.splitlines():
    event = json.loads(line)
    if event.get("reason") != "compiler-artifact":
        continue
    name = event["target"]["name"]
    if name in {"image", "imageproc", "magicextract"}:
        libraries[name] = next(Path(file) for file in event["filenames"] if file.endswith(".rlib"))
output = root / "target" / "performance" / "edge_probe"
output.parent.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "rustc", "--edition=2024", "-C", "opt-level=3",
        str(Path(__file__).with_name("edge_probe.rs")), "-o", str(output),
        "-L", f"dependency={libraries['image'].parent}",
        "--extern", f"image={libraries['image']}",
        "--extern", f"imageproc={libraries['imageproc']}",
    ],
    cwd=root,
    check=True,
)
subprocess.run([str(output), *sys.argv[1:]], cwd=root, check=True)
