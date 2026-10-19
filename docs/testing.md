# Testing

Run the scene tests and evaluator from a repository checkout; these tools and
their saved image corpus are excluded from the published library package.

Saved scene tests read the permanent JPEG scenes in
[`tests/fixtures/scenes`](../tests/fixtures/scenes/manifest.json) and
[`tests/fixtures/playmats`](../tests/fixtures/playmats/manifest.json), plus
[camera views](../tests/fixtures/camera/manifest.json) and
[white/silver borders](../tests/fixtures/borders/manifest.json). They never
render replacements or download images. No Python or source-image cache is needed.

```sh
cargo test --locked
```

## Rust checks

CI runs these on stable and the Rust 1.90.0 MSRV:

```sh
for toolchain in stable 1.90.0; do
  cargo +"$toolchain" check --locked --all-targets --all-features
  cargo +"$toolchain" test --locked --all-features
  cargo +"$toolchain" fmt --all -- --check
  cargo +"$toolchain" clippy --locked --all-targets --all-features -- -D warnings
done
```

Public and private items require documentation, including actual error and panic
conditions. CI also runs separate `cargo msrv verify` and LLVM coverage jobs.

```sh
cargo msrv verify -- cargo check --locked --all-targets --all-features
cargo llvm-cov --locked --all-features --workspace --lcov --output-path lcov.info
```

Install `cargo-msrv`, `cargo-llvm-cov`, and the Rust `llvm-tools-preview` component
before running those optional local tools.

## Saved scene contract

Each scene has a JPEG, full card outlines, expected card count, visibility
fractions, and appearance references. The manifest pins every image with SHA-256.
Missing or changed bytes fail the integrity test instead of regenerating data.

Required regressions cover the original backgrounds, rotations, perspective,
shadows, mild blur, JPEG compression, noise, and multiple separated cards, plus
negative objects and partial occlusion. Hidden corners are annotated at the full
card outline, not at the occluder's boundary.

Scoring uses the default extractor and unique one-to-one card matches:

- Exact card counts and clockwise outlines; maximum corner error is 8 pixels.
- Crops are 250 × 350 RGB pixels.
- Interior RGB mean absolute error is below 22, or 26 for the smaller four-card
  scene, on the 0–255 scale. Comparison excludes a 20-pixel rim and permits a
  180-degree rotation, never reflection.

Unaltered scenes compare against clean source-face PNGs. Lighting, blur, and
occlusion cases can use independent ground-truth rectifications of the saved
JPEG, so the expected crop retains its actual lighting and occluding pixels.
Bounds and references are explicit in the manifest; fixtures cannot override
extractor settings.

Procedural unit tests separately cover invalid input, configuration, original
pixel detail after detection resizing, deterministic ordering, and canonical
corners. Shared scorer tests verify unique assignment and reject reflection.

## Reports and challenges

```sh
cargo run --release --locked --example evaluate_scenes -- --output target/scenes.json
cargo run --release --locked --example evaluate_scenes -- --include-challenges --output target/challenges.json
```

The JSON report includes counts, detector polygons, unique matches, corner errors,
pixel errors, and every failed criterion. Exit codes are **0** for all selected
scenes passing, **1** for scoring failures, and **2** for invalid fixtures or I/O.
The challenge tier deliberately includes unresolved failures and is excluded
from default CI assertions, while its image integrity is always checked.

Controlled scenes do not establish real-camera accuracy or video throughput.

The border corpus uses original white-border Lightning Bolt and silver-border
Booster Tutor faces from Scryfall in ten generated planar scenes. Seven plain and
playmat scenes are required regressions; two bright sleeve-outline controls and
the upper-right white-border playmat scene are challenge entries. Full physical
card corners and observed-face references are pinned for every scene. Silver describes the printed border in the source image;
these scenes do not simulate metallic reflections or plastic optics.

Both border colours are tested at each of three playmat placements: the original
left-hand pose, a smaller card at the upper right, and a smaller card near the
bottom centre. Positions, angles, and sizes are fixed before evaluation. The
original scenes stay pinned; four additional saved scenes test different regions
of the same detailed background. All four new results are shown in the gallery,
including the upper-right white card, whose correct card crop is accompanied by
an extra background crop. That scene remains a challenge; no placement or scoring
threshold was changed after evaluation.

The [source credits and PNG hashes](../tests/fixtures/borders/sources.json) and
[authoring recipes](../tests/fixtures/borders/recipes.json) are saved separately
from the source cache. Tests need neither that cache nor Python. To intentionally
author the scenes, use Pillow 10.2.0 and an offline cache containing the two pinned
original PNGs named in the source credits:

```sh
python3 scripts/build-border-fixtures.py --write --sources /path/to/border-source-cache
```

Existing generated output requires `--force` for intentional replacement.
Use `--output target/border-preview` to create a separate collection. The tool
verifies cached PNG and background hashes before writing and never downloads.

The extended research corpora contain 768 source-face-split learning scenes,
192 saved sequence frames, and 40 material cases covering card backs, sleeves,
occlusion, and empty-pocket distractors. These are permanent JPEGs with pinned
crop references and full physical outlines. `tests/research_fixtures.rs` checks
their bytes, scoring limits, and source splits without running a model or using
downloaded PNGs. They remain challenge data; their presence does not imply the
default extractor passes them. See the [material preview](../tests/fixtures/materials/preview.jpg)
and [learning source credits](../tests/fixtures/learning/sources.json).

## Example gallery

The [GitHub Pages gallery](https://cmeister2.github.io/magicextract/) shows a
selection of saved inputs beside the default extractor's actual PNG output.
It separates successful matches, misses and wrong crops, false matches, and
correct non-matches. These are generated test scenes, not camera photographs.

Regenerate the static page, crops, and scored evidence from a checkout:

```sh
python3 scripts/build-gallery.py
```

The builder uses the existing Rust extractor and evaluator. It copies the saved
JPEGs without changing them and retains failed crops. `site/evidence.json`
records source hashes, the evaluated code revision, and each scene's metrics.
Edit `scripts/gallery-cases.json` to change the selection,
`scripts/gallery-template.html` for page text, and `site/styles.css` for styling.
The Pages workflow publishes `site/` after changes reach `main`.

## Source images

Scenes were authored from Scryfall PNGs, preserving transparent rounded corners
before projection. The [source manifest](../tests/fixtures/scryfall/manifest.json)
records card IDs, artists, URLs, and hashes. Authoring uses four cached files;
the downloader identifies itself, spaces requests, and stops on HTTP errors.
The separate learning corpus uses 60 additional credited source faces, split
36/12/12 between training, validation, and test; none of the original four
faces enter that training set. Its source downloads are also cached only for
authoring. Learning loaders read saved scenes and apply only recorded training
transforms; tests never synthesize replacement scenes.

Saved scene images and references carry third-party card artwork, copyright
Wizards of the Coast and/or its artists. The project's MIT license covers its
own code. Source PNG downloads are excluded from the published crate.
