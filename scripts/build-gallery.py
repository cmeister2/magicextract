#!/usr/bin/env python3
"""Build the static gallery from pinned fixtures and actual Rust extractor output."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
CASES = ROOT / "scripts/gallery-cases.json"
TEMPLATE = ROOT / "scripts/gallery-template.html"
REPOSITORY = "https://github.com/cmeister2/magicextract"


def run(arguments, allowed=(0,)):
    """Run a command from the checkout, retaining its output for useful errors."""
    result = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True)
    if result.returncode not in allowed:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(str, arguments))}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout


def digest(path):
    """Return a lowercase SHA-256 digest of exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def escaped(value):
    """Escape text used in HTML content and quoted attributes."""
    return html.escape(str(value), quote=True)


def noun(count, singular, plural=None):
    """Format a count with its singular or plural noun."""
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def validate_result(group, case, report):
    """Refuse to publish a caption whose classification contradicts the evaluator."""
    expected = report["expected_count"]
    detected = report["detected_count"]
    passed = report["passed"]
    result = case["result"]
    valid = {
        "matches": passed and expected > 0 and result == "Match",
        "misses": (
            not passed
            and expected > 0
            and (
                (detected == 0 and result == "Miss")
                or (detected > 0 and result == "Wrong crop")
            )
        ),
        "false-matches": (
            not passed and expected == 0 and detected > 0 and result == "False match"
        ),
        "non-matches": (
            passed and expected == 0 and detected == 0 and result == "Correct non-match"
        ),
    }
    if not valid.get(group, False):
        raise ValueError(f"Gallery classification no longer holds: {case['id']} ({result})")


def render_case(group, case, report, input_path, crops, source_url, first):
    """Render one input alongside every actual crop, including failed crops."""
    title = escaped(case["title"])
    description = escaped(case["description"])
    expected = report["expected_count"]
    detected = report["detected_count"]
    loading = "eager" if first else "lazy"
    lines = [
        f'<article id="{escaped(case["id"])}" class="example {group}">',
        f'<header><h3>{title}</h3><span class="result">{escaped(case["result"])}</span></header>',
        f'<p class="description">{description}</p>',
        '<div class="comparison">',
        '<figure class="input">',
        f'<figcaption>Input · {noun(expected, "card")} present</figcaption>',
        f'<a href="{escaped(input_path)}"><img src="{escaped(input_path)}" '
        f'width="640" height="520" alt="{description}" loading="{loading}"></a>',
        '</figure>',
        '<figure class="output">',
        f'<figcaption>Output · {noun(detected, "crop")} returned</figcaption>',
    ]
    if crops:
        lines.append('<div class="crops">')
        for index, path in enumerate(crops, 1):
            alt = escaped(f"Returned crop {index} of {detected}: {case['title'].lower()}.")
            lines.append(
                f'<a href="{escaped(path)}"><img src="{escaped(path)}" '
                f'width="250" height="350" alt="{alt}" loading="lazy"></a>'
            )
        lines.append('</div>')
    else:
        lines.append('<p class="empty-output">No crops returned.</p>')
    lines.extend([
        '</figure>',
        '</div>',
        f'<p class="example-links"><a href="{escaped(source_url)}">Source fixture</a></p>',
    ])
    if report["failures"]:
        summary = {
            "Miss": "Why this is a miss",
            "Wrong crop": "Why this is a wrong crop",
            "False match": "Why this is a false match",
        }[case["result"]]
        lines.append(f'<details class="checks"><summary>{summary}</summary><ul>')
        lines.extend(f'<li>{escaped(failure)}</li>' for failure in report["failures"])
        lines.append('</ul></details>')
    lines.append('</article>')
    return "\n".join(lines)


def build(skip_build):
    """Evaluate each selected corpus once and publish an internally checked gallery."""
    if not skip_build:
        print("Building Rust extraction examples…", flush=True)
        run([
            "cargo", "build", "--release", "--locked",
            "--example", "extract", "--example", "evaluate_scenes",
        ])
    extract = ROOT / "target/release/examples/extract"
    evaluate = ROOT / "target/release/examples/evaluate_scenes"
    if not extract.is_file() or not evaluate.is_file():
        raise FileNotFoundError("Build the release examples before using --skip-build")
    template = TEMPLATE.read_text(encoding="utf-8")
    for placeholder in ("<!-- GALLERY -->", "<!-- PROVENANCE -->"):
        if template.count(placeholder) != 1:
            raise ValueError(f"Template must contain exactly one {placeholder}")
    sections = json.loads(CASES.read_text(encoding="utf-8"))["sections"]
    selected = [case for section in sections for case in section["cases"]]
    identifiers = [case["id"] for case in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Gallery scene identifiers must be unique")
    revision = run(["git", "rev-parse", "HEAD"]).strip()
    report_directory = ROOT / "target/gallery-reports"
    report_directory.mkdir(parents=True, exist_ok=True)
    reports, manifests = {}, {}
    for corpus in sorted({case["corpus"] for case in selected}):
        manifest_path = ROOT / f"tests/fixtures/{corpus}/manifest.json"
        report_path = report_directory / f"{corpus}.json"
        print(f"Evaluating {corpus}…", flush=True)
        run([
            str(evaluate), "--manifest", str(manifest_path.relative_to(ROOT)),
            "--include-challenges", "--output", str(report_path),
        ], allowed=(0, 1))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports[corpus] = {scene["id"]: scene for scene in report["scenes"]}
        manifests[corpus] = (
            manifest_path,
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    source_files = sorted((ROOT / "src").rglob("*.rs")) + [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    evidence = {
        "format_version": 1,
        "source_revision": revision,
        "source_url": f"{REPOSITORY}/tree/{revision}",
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_files},
        "generator": "scripts/build-gallery.py",
        "generator_sha256": digest(Path(__file__)),
        "selection_sha256": digest(CASES),
        "extractor": "CardExtractor::default()",
        "output_dimensions": [250, 350],
        "inputs": "Saved generated test scenes; not independent camera photographs.",
        "cases": [],
    }
    rendered_sections = []
    with tempfile.TemporaryDirectory(prefix="gallery-", dir=ROOT / "target") as temporary:
        staging = Path(temporary)
        image_directory = staging / "images"
        image_directory.mkdir()
        first = True
        for section in sections:
            group = section["id"]
            lines = [
                f'<section id="{escaped(group)}" aria-labelledby="{escaped(group)}-title">',
                f'<h2 id="{escaped(group)}-title">{escaped(section["title"])}</h2>',
                f'<p>{escaped(section["intro"])}</p>',
            ]
            for case in section["cases"]:
                corpus, scene_id = case["corpus"], case["id"]
                report = reports[corpus][scene_id]
                validate_result(group, case, report)
                manifest_path, manifest = manifests[corpus]
                scene = next(scene for scene in manifest["scenes"] if scene["id"] == scene_id)
                source = manifest_path.parent / scene["file"]
                input_hash = digest(source)
                if input_hash != manifest["sha256"][scene["file"]]:
                    raise ValueError(f"Input fixture checksum mismatch: {scene_id}")
                case_directory = image_directory / scene_id
                case_directory.mkdir()
                shutil.copyfile(source, case_directory / "input.jpg")
                run([str(extract), str(source), str(case_directory)])
                actual_crops = sorted(case_directory.glob("card-*.png"))
                if len(actual_crops) != report["detected_count"]:
                    raise ValueError(f"Extractor and evaluator counts differ: {scene_id}")
                input_path = f"images/{scene_id}/input.jpg"
                crop_paths = [f"images/{scene_id}/{path.name}" for path in actual_crops]
                source_relative = str(source.relative_to(ROOT))
                source_url = f"{REPOSITORY}/blob/{revision}/{source_relative}"
                lines.append(render_case(group, case, report, input_path, crop_paths, source_url, first))
                evidence["cases"].append({
                    **case,
                    "group": group,
                    "input": input_path,
                    "input_source": source_relative,
                    "input_sha256": input_hash,
                    "source_manifest": str(manifest_path.relative_to(ROOT)),
                    "source_manifest_sha256": digest(manifest_path),
                    "crops": crop_paths,
                    "crop_sha256": {path: digest(actual) for path, actual in zip(crop_paths, actual_crops)},
                    "report": report,
                })
                first = False
            lines.append('</section>')
            rendered_sections.append("\n".join(lines))
        provenance = (
            f'<p>Generated with the default extractor at '
            f'<a href="{REPOSITORY}/tree/{revision}">{revision[:7]}</a>. '
            '<a href="evidence.json">Inputs, checksums and evaluation results</a>.</p>'
        )
        page = template.replace("<!-- GALLERY -->", "\n\n".join(rendered_sections))
        page = page.replace("<!-- PROVENANCE -->", provenance)
        SITE.mkdir(exist_ok=True)
        if (SITE / "images").exists():
            shutil.rmtree(SITE / "images")
        shutil.move(str(image_directory), str(SITE / "images"))
        (SITE / "evidence.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (SITE / "index.html").write_text(page, encoding="utf-8")
    print(f"Built {len(selected)} examples in site/index.html", flush=True)


def main():
    """Parse the single build-cache option and report actionable failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-build", action="store_true", help="Use already built release examples")
    arguments = parser.parse_args()
    try:
        build(arguments.skip_build)
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as error:
        parser.exit(1, f"Gallery build failed: {error}\n")


if __name__ == "__main__":
    main()
