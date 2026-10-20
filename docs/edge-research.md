# Edge frontend investigation

ELSED is a promising optional source of line segments, but the existing research
adapter does not justify replacing the production detector. Its C++ frontend is
fast on the representative scenes; quadrilateral grouping and selecting the
physical exterior remain separate speed and accuracy problems. The current
native detector and its restricted-domain benchmark remain the baseline.

## What the primary sources establish

[ELSED, Suárez et al.](https://arxiv.org/pdf/2108.03144), section 4.5 and table 4,
reports segment detection on 101 YUD images at 640×480: 5.38 ms on an i7-8750H,
10.20 ms on a OnePlus 7 Pro with Snapdragon 855, and 59.99 ms on a Samsung J5
2017. Its edge drawing and line fitting share one traversal, with optional jumps
over discontinuities. These are segment timings on the authors' images and
hardware; they do not include finding card quadrilaterals, selecting exteriors,
rectifying RGB crops, or a camera application's frame processing.

[PiDiNet, Su et al.](https://arxiv.org/pdf/2108.07009), figure 7 and table 5,
reports 92 FPS for PiDiNet and 215 FPS for PiDiNet-Tiny-L on an NVIDIA RTX 2080 Ti.
The [authors' repository](https://github.com/hellozhuo/pidinet) also identifies
its throughput test as GPU execution. These results support investigating a small
learned edge model, but supply no phone CPU, mobile GPU, NPU, or complete card
pipeline measurement. No PiDiNet model was downloaded or added as a dependency
for this investigation.

[TEED, Soria et al.](https://arxiv.org/pdf/2308.06468), section 5.1 and table 2,
is another small learned model, with 58,000 parameters. Its reported laptop
throughput is 2.6 FPS on an Intel i5-10210U, compared with 392.5 FPS for Canny in
that experiment. These measurements use the paper's data and implementation,
so they cannot predict this application's latency. They do show why a small
parameter count alone is insufficient evidence for a 15 FPS phone pipeline.
TEED was reviewed but not downloaded or integrated.

## Local ELSED measurements

Measurements on 2026-10-20 used the already cached author revision
`1878213b2f5f06a9261d8b1838f53d48e5fd128d`, compiled unchanged with `-O3`,
OpenCV core/imgproc, one OpenCV thread, and the existing research adapter settings
including Gaussian sigma 1 and kernel size 7. The machine was an AMD Ryzen 7 PRO
7840U running x86-64 Linux under WSL2. Other project timing jobs yielded the CPU
for these runs.

The three saved inputs match the native detector's representative timing matrix:

- Plain: `tests/fixtures/scenes/images/baseline-lightning-bolt-background-green.jpg`.
- Playmat: `tests/fixtures/playmats/images/playmat-fantasy-forest-lightning-bolt.jpg`.
- Camera: `tests/fixtures/camera/images/camera-forest-normal-control.jpg`.

All originals are 640×520. Detection grayscale uses the existing OpenCV BGR2GRAY
and INTER_AREA adapter. The frontend column is the median of 31 in-process
`detectSalient` calls after five warmups, reusing the detector. It excludes image
decoding, preprocessing, I/O, grouping, geometry, selection, and crop generation.
The grouping column is one execution of the existing Python/NumPy
`line_groups.propose` implementation. The sum combines those component
measurements; it is not a measured complete pipeline latency.

| Scene | Longest side | ELSED frontend, ms | Existing grouping, ms | Component sum, ms |
| --- | ---: | ---: | ---: | ---: |
| Plain | 640 | 0.83 | 236.90 | 237.73 |
| Playmat | 640 | 7.77 | 389.61 | 397.38 |
| Camera | 640 | 3.59 | 234.39 | 237.98 |
| Plain | 480 | 0.52 | 113.78 | 114.30 |
| Playmat | 480 | 2.76 | 187.47 | 190.23 |
| Camera | 480 | 2.05 | 115.32 | 117.37 |
| Plain | 320 | 0.25 | 46.05 | 46.30 |
| Playmat | 320 | 1.33 | 36.32 | 37.65 |
| Camera | 320 | 0.94 | 20.02 | 20.96 |

The unchanged PGM CLI was measured separately, including process startup, reading
PGM pixels, and JSON output. Its five-run frontend medians ranged from 7.1 to
15.1 ms across the same full-frame configurations. In-process calls remove this
adapter overhead; a phone application should not spawn this CLI per frame.

A fixed central 80% window in each axis was also tested, cropped before grayscale
and resizing. At maximum dimension 640 it reduces actual pixels from 640×520 to
512×416 and decreases frontend times from 0.83/7.77/3.59 ms to
0.71/3.22/3.05 ms for plain/playmat/camera. At maximum dimension 480, both full
frame and window become 480×390, so cropping instead increases the card's
apparent size. Segment count and Python grouping then increase: grouping changes
from 114/187/115 ms to 315/324/131 ms. A smaller source domain therefore does not
guarantee faster line grouping at a fixed working resolution. Angle restrictions
were not added to this isolated adapter.

Raw results are archived as
[`elsed-components.json.gz`](../benchmarks/2026-10-20/elsed-components.json.gz)
and [`elsed-inprocess.json.gz`](../benchmarks/2026-10-20/elsed-inprocess.json.gz),
with the [CLI/grouping timing script](../benchmarks/2026-10-20/elsed-benchmark.py)
and [in-process timing adapter](../benchmarks/2026-10-20/elsed-timed_cli.cpp).
The grouping script requires the existing ignored research checkout, cached
ELSED CLI, OpenCV, and NumPy; it is an experiment record rather than a portable
crate example. The local build provenance remains in
`target/research-v2/learned-lines/elsed-build.json`. No production dependencies or
algorithm were changed for this ELSED experiment.

## Accuracy and the integration decision

The existing completed 205-image exploratory campaign already tests this ELSED
frontend with unchanged native physical geometry and declared selectors. Its
`strong` selection passes 117/170 synthetic scenes and 83/98 required scenes,
returning 207 quads. It produces zero oriented-IoU≥0.90 matches for the 98 fully
visible primary binder cards while returning 154 quads. The permissive `near`
selection reaches 22/98 binder matches while returning 2,186 quads and passes
10/170 synthetic scenes. Neither selection satisfies the production accuracy
gate.

For the three representative full-resolution saved scenes, the existing `strong`
result passes plain and camera. Playmat fails the crop appearance check: interior
RGB MAE 50.90 against a limit of 22, despite a 6.87-pixel corner error. This
illustrates why plausible segments and an approximately correct quadrilateral do
not establish usable rectified card content. The new restricted-resolution and
window timings above have not been submitted as accuracy results.

The generous, uncapped proposal pool contains truth-selected oriented-IoU≥0.90
outlines for 130/169 synthetic targets and 62/98 binder targets. These are oracle
pool ceilings, not final selection accuracy, strict corner accuracy, or crop
fidelity. Many nested card panels and background rectangles remain. The 48-line
pool limit can also discard useful lines before grouping. Existing campaign
records are `target/research-v2/learned-lines/campaign-summary.json` and
`evaluation/elsed-{strong,near,native-union}.json`; method and reproduction details
are in the local `research/methods/learned-lines.md` investigation (the research
directory and cached campaign artifacts are excluded from the published crate).
All 35 binder images are exploratory data, including the previously consumed
14-image holdout; these results supply no new independent holdout.

Defer production integration. A useful next ELSED experiment would move bounded
line grouping into native code, apply orientation/region restrictions before
combinatorial pairing, and evaluate a label-independent exterior selector with
unchanged corner and crop checks. Benchmark the whole path, including
preprocessing and RGB rectification, against the optimized native detector before
choosing it. A segment frontend alone supplies no such comparison.

Android is the intended platform; no particular phone model or executable phone
benchmark target was provided. This workspace
has no `adb` executable or connected phone runtime. A Rust Android compilation
target alone does not establish a phone deployment or phone timing. The authors'
phone measurements make ELSED worth retaining as an option, while 15 FPS for
this application's camera pipeline remains to be measured on the intended
phone.
