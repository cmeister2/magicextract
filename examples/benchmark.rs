//! Benchmark decoded frames, domain restrictions, and crop accuracy in release mode.
//!
//! Run `cargo run --release --example benchmark -- --help` for selection flags.
//! Fixture verification, decoding, and accuracy scoring stay outside timed loops.

use std::{
    collections::{BTreeMap, BTreeSet},
    env, fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use image::{DynamicImage, RgbImage};
use magicextract::{
    CardDetection, CardExtractor, DetectionMode, DetectionRegion, ExtractedCard, ExtractorConfig,
    OrientationConstraint,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

/// Shared fixture integrity checks used by the permanent regression tests.
#[path = "../tests/support/mod.rs"]
pub mod scene_evaluation;

/// A fallible benchmark or fixture operation.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Command-line settings shared by all compared variants.
struct Options {
    /// Corpus manifests, each checked before timing starts.
    manifests: Vec<PathBuf>,
    /// Individual manifests whose challenge tier is included without broadening other corpora.
    challenge_manifests: BTreeSet<PathBuf>,
    /// Optional exact scene identifiers, selected across all supplied manifests.
    scenes: BTreeSet<String>,
    /// Optional variant identifiers; empty selects the complete comparison.
    variants: BTreeSet<String>,
    /// Number of measured calls to each API for each scene.
    iterations: usize,
    /// Unmeasured calls to each API before collecting samples.
    warmup: usize,
    /// Whether to include explicitly challenging saved scenes.
    challenges: bool,
    /// Optional report destination; stdout otherwise contains the JSON report.
    output: Option<PathBuf>,
    /// Fixed central rectangle as normalized x, y, width, and height.
    roi: [f32; 4],
    /// Optional fixed pixel rectangle, overriding the normalized ROI.
    pixel_region: Option<DetectionRegion>,
    /// Allowed short-edge orientation in clockwise degrees modulo 180.
    angle: f32,
    /// Allowed deviation from the requested short-edge orientation.
    angle_tolerance: f32,
}

/// One annotated physical card, with original-image coordinates.
#[derive(Clone, Deserialize)]
struct ExpectedCard {
    /// Identifier used to resolve its clean appearance reference.
    source: String,
    /// Full clockwise physical outline, including occluded corners.
    corners: [[f32; 2]; 4],
    /// Optional scene-specific observed appearance.
    reference: Option<String>,
}

/// Minimal scoring fields from a verified permanent scene annotation.
#[derive(Deserialize)]
struct Scene {
    /// Stable identifier used in reports and selection.
    id: String,
    /// Relative path of the decoded JPEG.
    file: String,
    /// Regression or challenge classification.
    tier: String,
    /// Physical cards that should be returned.
    cards: Vec<ExpectedCard>,
    /// Maximum accepted displacement of any outline corner.
    corner_tolerance: f32,
    /// Optional maximum accepted interior RGB mean absolute error.
    pixel_mae_limit: Option<f32>,
}

/// The relevant manifest fields, validated by the shared integrity checker.
#[derive(Deserialize)]
struct Manifest {
    /// Appearance references indexed by source identifier.
    references: BTreeMap<String, String>,
    /// Saved image annotations in stable order.
    scenes: Vec<Scene>,
}

/// All decoded data needed for timing and independent accuracy scoring.
struct Input {
    /// Manifest path recorded to distinguish duplicate scene identifiers.
    manifest: String,
    /// Immutable scene annotation in original-image coordinates.
    scene: Scene,
    /// Decoded camera-like input, reused without cloning in timed calls.
    image: DynamicImage,
    /// Decoded canonical references in annotation order.
    references: Vec<Option<RgbImage>>,
}

/// A named combination of resolution, algorithm, and domain restrictions.
struct Variant {
    /// Stable command-line and report identifier.
    name: &'static str,
    /// Maximum detection dimension, or the default for the baseline.
    dimension: u32,
    /// Whether to use the low-latency detector path.
    fast: bool,
    /// Whether to restrict the searched image rectangle.
    roi: bool,
    /// Whether to restrict allowed short-edge angles.
    angle: bool,
}

/// Measured API durations and independent scoring for one scene and variant.
struct Measurement {
    /// Input index, allowing exact comparisons across variants.
    input: usize,
    /// Whether every annotated card lies within the configured domain.
    eligible: bool,
    /// Detection durations in milliseconds, excluding decoding and scoring.
    detect: Vec<f64>,
    /// Complete detection plus canonical-crop durations in milliseconds.
    extract: Vec<f64>,
    /// Per-scene count, geometry, and crop-fidelity measurements.
    report: Value,
}

/// Runs the benchmark and reports invalid inputs separately from accuracy failures.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("benchmark failed: {error}");
            ExitCode::from(2)
        }
    }
}

/// Parses flags, preloads all inputs, and compares sequentially timed API calls.
///
/// # Errors
///
/// Returns an error for invalid flags or fixtures, unsupported configurations,
/// extraction errors, or an unwritable report destination.
fn run() -> Result<()> {
    let Some(options) = options()? else {
        return Ok(());
    };
    if cfg!(debug_assertions) {
        return Err(
            "timing requires a release build: cargo run --release --example benchmark".into(),
        );
    }
    let inputs = load_inputs(&options)?;
    let variants: Vec<_> = variants()
        .into_iter()
        .filter(|variant| options.variants.is_empty() || options.variants.contains(variant.name))
        .collect();
    if variants.is_empty()
        || options
            .variants
            .iter()
            .any(|name| !variants.iter().any(|variant| variant.name == name))
    {
        return Err("unknown --variant; use --help for the available names".into());
    }
    let mut results = Vec::new();
    for variant in &variants {
        eprintln!("{}: {} decoded scenes", variant.name, inputs.len());
        let mut measurements = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            measurements.push(measure(input, index, variant, &options)?);
        }
        results.push(measurements);
    }
    let baseline = variants
        .iter()
        .position(|variant| variant.name == "default");
    let reports: Vec<_> = variants.iter().zip(&results).map(|(variant, measurements)| {
        let eligible: Vec<_> = measurements.iter().filter(|m| m.eligible).collect();
        let baseline_paired: Vec<_> = baseline.map_or_else(Vec::new, |index| {
            results[index].iter().filter(|m| measurements[m.input].eligible).collect()
        });
        let detect = collect_samples(&eligible, false);
        let extract = collect_samples(&eligible, true);
        let baseline_detect = collect_samples(&baseline_paired, false);
        let baseline_extract = collect_samples(&baseline_paired, true);
        let passed = eligible.iter().filter(|m| m.report["accuracy"]["passed"] == true).count();
        json!({
            "name": variant.name,
            "settings": {"max_detection_dimension":variant.dimension,
                "detection_mode":if variant.fast {"fast"} else {"full"},
                "roi_normalized":(variant.roi && options.pixel_region.is_none()).then_some(options.roi),
                "roi_pixels":options.pixel_region.filter(|_|variant.roi).map(|r|[r.x,r.y,r.width,r.height]),
                "orientation_degrees":variant.angle.then_some(options.angle),
                "orientation_tolerance_degrees":variant.angle.then_some(options.angle_tolerance)},
            "eligible_summary": {"scenes":eligible.len(), "passed":passed,
                "failed":eligible.len()-passed,
                "detect_ms":statistics(&detect), "extract_ms":statistics(&extract),
                "paired_default_detect_ms":statistics(&baseline_detect),
                "paired_default_extract_ms":statistics(&baseline_extract),
                "paired_default_detect_speedup":speedup(&baseline_detect,&detect),
                "paired_default_extract_speedup":speedup(&baseline_extract,&extract)},
            "scenes":measurements.iter().map(|m| &m.report).collect::<Vec<_>>()
        })
    }).collect();
    let report = json!({
        "format_version":1,
        "metadata":metadata(),
        "methodology": {
            "iterations_per_scene_api":options.iterations,"warmup_per_scene_api":options.warmup,
            "timed_apis":["detect", "extract (includes detection and RGB crop allocation)"],
            "excluded_from_timing":["fixture hash validation", "image decoding", "scoring", "JSON output"],
            "execution":"sequential, single caller thread, alternating detect and extract; no concurrent benchmark processes",
            "percentiles":"nearest rank: sorted[ceil(p*n)-1]; samples equally weighted across scenes",
            "domain_eligibility":"all ground-truth cards fully inside the fixed ROI and mean short-edge angle inside the constraint; negatives are eligible",
            "speedup":"ratio of median durations against default on exactly the same eligible scenes",
            "phone_target_ms":1000.0/15.0,
            "scope":"decoded-frame CPU measurements on the reported host; handset capture, decoding, display, thermal behavior and sustained FPS require measurements on the target phone"
        },
        "variants":reports
    });
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = options.output {
        fs::write(path, format!("{json}\n"))?;
    } else {
        println!("{json}");
    }
    Ok(())
}

/// Parses benchmark controls without adding a command-line parsing dependency.
///
/// # Errors
///
/// Returns an error for missing values, invalid bounds, or unknown flags.
fn options() -> Result<Option<Options>> {
    let mut options = Options {
        manifests: Vec::new(),
        challenge_manifests: BTreeSet::new(),
        scenes: BTreeSet::new(),
        variants: BTreeSet::new(),
        iterations: 3,
        warmup: 1,
        challenges: false,
        output: None,
        roi: [0.1, 0.1, 0.8, 0.8],
        pixel_region: None,
        angle: 0.0,
        angle_tolerance: 15.0,
    };
    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--manifest" => options.manifests.push(PathBuf::from(
                arguments.next().ok_or("--manifest needs a path")?,
            )),
            "--challenge-manifest" => {
                let path = PathBuf::from(
                    arguments
                        .next()
                        .ok_or("--challenge-manifest needs a path")?,
                );
                options.challenge_manifests.insert(path.clone());
                options.manifests.push(path);
            }
            "--scene" => {
                options
                    .scenes
                    .insert(arguments.next().ok_or("--scene needs an ID")?);
            }
            "--variant" => {
                options
                    .variants
                    .insert(arguments.next().ok_or("--variant needs a name")?);
            }
            "--iterations" => {
                options.iterations = arguments
                    .next()
                    .ok_or("--iterations needs a count")?
                    .parse()?
            }
            "--warmup" => {
                options.warmup = arguments.next().ok_or("--warmup needs a count")?.parse()?
            }
            "--output" => {
                options.output = Some(PathBuf::from(
                    arguments.next().ok_or("--output needs a path")?,
                ))
            }
            "--angle" => {
                options.angle = arguments.next().ok_or("--angle needs degrees")?.parse()?
            }
            "--angle-tolerance" => {
                options.angle_tolerance = arguments
                    .next()
                    .ok_or("--angle-tolerance needs degrees")?
                    .parse()?
            }
            "--roi" => {
                let value = arguments
                    .next()
                    .ok_or("--roi needs normalized x,y,width,height")?;
                let values: Vec<f32> = value
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?;
                options.roi = values
                    .try_into()
                    .map_err(|_| "--roi requires four comma-separated values")?;
            }
            "--region" => {
                let value = arguments
                    .next()
                    .ok_or("--region needs pixel x,y,width,height")?;
                let values: Vec<u32> = value
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?;
                let [x, y, width, height]: [u32; 4] = values
                    .try_into()
                    .map_err(|_| "--region requires four comma-separated integers")?;
                options.pixel_region = Some(DetectionRegion {
                    x,
                    y,
                    width,
                    height,
                });
            }
            "--include-challenges" => options.challenges = true,
            "--help" | "-h" => {
                println!(
                    "Usage: cargo run --release --example benchmark -- [OPTIONS]\n\n\
                    --manifest PATH       Repeat to combine verified permanent corpora\n\
                    --challenge-manifest P Include challenges from only this manifest\n\
                    --scene ID            Repeat to select exact scene identifiers\n\
                    --variant NAME        Repeat to select variants (default: all)\n\
                    --iterations N        Measured calls per scene/API (default: 3)\n\
                    --warmup N            Unmeasured calls per scene/API (default: 1)\n\
                    --include-challenges  Include saved challenge cases\n\
                    --roi X,Y,W,H         Fixed normalized ROI (default: 0.1,0.1,0.8,0.8)\n\
                    --region X,Y,W,H      Fixed pixel ROI, overriding --roi\n\
                    --angle DEGREES       Short-edge angle modulo 180 (default: 0)\n\
                    --angle-tolerance D   Allowed angle deviation (default: 15)\n\
                    --output PATH         Write JSON report (default: stdout)\n\n\
                    Variants: {}\n\n\
                    The default corpus is tests/fixtures/scenes/manifest.json.\n\
                    Reports retain out-of-domain scene results separately from eligible summaries.\n\
                    Exit status: 0 completed (including scored failures); 2 invalid input/I/O.\n\
                    Stop other CPU workloads before collecting performance measurements.",
                    variants()
                        .iter()
                        .map(|v| v.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                return Ok(None);
            }
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    if options.iterations == 0 {
        return Err("--iterations must be greater than zero".into());
    }
    let [x, y, width, height] = options.roi;
    if options.roi.iter().any(|v| !v.is_finite())
        || x < 0.0
        || y < 0.0
        || width <= 0.0
        || height <= 0.0
        || x + width > 1.0
        || y + height > 1.0
    {
        return Err(
            "--roi must describe a positive rectangle inside normalized 0..1 coordinates".into(),
        );
    }
    if !options.angle.is_finite()
        || !options.angle_tolerance.is_finite()
        || !(0.0..=90.0).contains(&options.angle_tolerance)
    {
        return Err("angle must be finite and angle tolerance must be in 0..=90".into());
    }
    if options.manifests.is_empty() {
        options
            .manifests
            .push(scene_evaluation::default_manifest_path());
    }
    Ok(Some(options))
}

/// Enumerates independent resolution, algorithm, and restricted-domain comparisons.
fn variants() -> Vec<Variant> {
    [
        ("default", 1600, false, false, false),
        ("full-640", 640, false, false, false),
        ("full-480", 480, false, false, false),
        ("full-320", 320, false, false, false),
        ("fast-640", 640, true, false, false),
        ("fast-480", 480, true, false, false),
        ("fast-320", 320, true, false, false),
        ("full-640-roi", 640, false, true, false),
        ("full-640-angle", 640, false, false, true),
        ("full-640-roi-angle", 640, false, true, true),
        ("fast-480-roi", 480, true, true, false),
        ("fast-480-angle", 480, true, false, true),
        ("fast-480-roi-angle", 480, true, true, true),
    ]
    .into_iter()
    .map(|(name, dimension, fast, roi, angle)| Variant {
        name,
        dimension,
        fast,
        roi,
        angle,
    })
    .collect()
}

/// Verifies permanent fixtures and decodes images and references before timing.
///
/// # Errors
///
/// Returns an error for invalid fixtures, missing images, or an empty/unknown selection.
fn load_inputs(options: &Options) -> Result<Vec<Input>> {
    let mut inputs = Vec::new();
    for path in &options.manifests {
        scene_evaluation::validate_fixture_integrity(path)?;
        let manifest: Manifest = serde_json::from_slice(&fs::read(path)?)?;
        let root = path.parent().unwrap_or_else(|| Path::new("."));
        let mut decoded = BTreeMap::new();
        for scene in manifest.scenes {
            if (!options.challenges
                && !options.challenge_manifests.contains(path)
                && scene.tier != "regression")
                || (!options.scenes.is_empty() && !options.scenes.contains(&scene.id))
            {
                continue;
            }
            let image = image::open(root.join(&scene.file))?;
            let mut references = Vec::new();
            for card in &scene.cards {
                let reference = card
                    .reference
                    .as_ref()
                    .or_else(|| manifest.references.get(&card.source));
                if let Some(reference) = reference.filter(|_| scene.pixel_mae_limit.is_some()) {
                    if !decoded.contains_key(reference) {
                        decoded.insert(
                            reference.clone(),
                            image::open(root.join(reference))?.into_rgb8(),
                        );
                    }
                    references.push(decoded.get(reference).cloned());
                } else {
                    references.push(None);
                }
            }
            inputs.push(Input {
                manifest: path.display().to_string(),
                scene,
                image,
                references,
            });
        }
    }
    if inputs.is_empty()
        || options
            .scenes
            .iter()
            .any(|id| !inputs.iter().any(|input| input.scene.id == *id))
    {
        return Err(
            "no scenes selected or an unknown scene ID; challenge scenes need --include-challenges"
                .into(),
        );
    }
    Ok(inputs)
}

/// Builds a fixed normalized screen region in original source-image coordinates.
fn region(image: &DynamicImage, roi: [f32; 4]) -> DetectionRegion {
    let x = (image.width() as f32 * roi[0]).round() as u32;
    let y = (image.height() as f32 * roi[1]).round() as u32;
    let right = (image.width() as f32 * (roi[0] + roi[2])).round() as u32;
    let bottom = (image.height() as f32 * (roi[1] + roi[3])).round() as u32;
    DetectionRegion {
        x,
        y,
        width: right.saturating_sub(x).max(1),
        height: bottom.saturating_sub(y).max(1),
    }
}

/// Times both APIs with decoded inputs and scores separately produced detections.
///
/// # Errors
///
/// Returns an error for an invalid extractor configuration or API input.
fn measure(
    input: &Input,
    index: usize,
    variant: &Variant,
    options: &Options,
) -> Result<Measurement> {
    let roi = variant.roi.then(|| {
        options
            .pixel_region
            .unwrap_or_else(|| region(&input.image, options.roi))
    });
    let orientation = variant.angle.then_some(OrientationConstraint {
        angle_degrees: options.angle,
        tolerance_degrees: options.angle_tolerance,
    });
    let extractor = CardExtractor::new(ExtractorConfig {
        max_detection_dimension: variant.dimension,
        detection_mode: if variant.fast {
            DetectionMode::Fast
        } else {
            DetectionMode::Full
        },
        detection_region: roi,
        orientation,
        ..ExtractorConfig::default()
    })?;
    for _ in 0..options.warmup {
        black_box(extractor.detect(black_box(&input.image))?);
        black_box(extractor.extract(black_box(&input.image))?);
    }
    let mut detect = Vec::with_capacity(options.iterations);
    let mut extract = Vec::with_capacity(options.iterations);
    for _ in 0..options.iterations {
        let started = Instant::now();
        let result = extractor.detect(black_box(&input.image))?;
        detect.push(started.elapsed().as_secs_f64() * 1000.0);
        black_box(result);
        let started = Instant::now();
        let result = extractor.extract(black_box(&input.image))?;
        extract.push(started.elapsed().as_secs_f64() * 1000.0);
        black_box(result);
    }
    let cards = extractor.extract(&input.image)?;
    let detections = extractor.detect(&input.image)?;
    let eligibility: Vec<_> = input
        .scene
        .cards
        .iter()
        .map(|card| {
            let in_roi = roi.is_none_or(|roi| {
                card.corners.iter().all(|p| {
                    p[0] >= roi.x as f32
                        && p[1] >= roi.y as f32
                        && p[0] < roi.x.saturating_add(roi.width).min(input.image.width()) as f32
                        && p[1] < roi.y.saturating_add(roi.height).min(input.image.height()) as f32
                })
            });
            let corners = card.corners;
            let dx = (corners[1][0] - corners[0][0]) + (corners[2][0] - corners[3][0]);
            let dy = (corners[1][1] - corners[0][1]) + (corners[2][1] - corners[3][1]);
            let angle = dy.atan2(dx).to_degrees().rem_euclid(180.0);
            let distance = (angle - options.angle.rem_euclid(180.0)).rem_euclid(180.0);
            let distance = distance.min(180.0 - distance);
            let in_angle = !variant.angle || distance <= options.angle_tolerance;
            json!({"source":card.source,"mean_short_edge_angle_degrees":angle,
            "inside_roi":in_roi,"inside_angle_constraint":in_angle,"eligible":in_roi&&in_angle})
        })
        .collect();
    let eligible = eligibility.iter().all(|card| card["eligible"] == true);
    let report = json!({
        "manifest":input.manifest,"id":input.scene.id,"tier":input.scene.tier,
        "image_dimensions":[input.image.width(),input.image.height()],
        "detection_region":roi.map(|roi|json!({"x":roi.x,"y":roi.y,"width":roi.width,"height":roi.height})),
        "domain_eligible":eligible,"card_eligibility":eligibility,
        "detect_ms":statistics(&detect),"extract_ms":statistics(&extract),
        "detect_samples_ms":detect,"extract_samples_ms":extract,
        "detect_extract_geometry_agree":detections == cards.iter().map(|card|card.detection.clone()).collect::<Vec<_>>(),
        "accuracy":score(input,&cards)
    });
    Ok(Measurement {
        input: index,
        eligible,
        detect,
        extract,
        report,
    })
}

/// Scores exact counts and optimal unique corner matches using the regression rules.
fn score(input: &Input, cards: &[ExtractedCard]) -> Value {
    let scene = &input.scene;
    let size = scene.cards.len().max(cards.len());
    let mut costs = vec![vec![0.0; size]; size];
    for (i, expected) in scene.cards.iter().enumerate() {
        for (j, actual) in cards.iter().enumerate() {
            costs[i][j] = f64::from(corner_error(&actual.detection, &expected.corners));
        }
    }
    let penalty = (costs.iter().flatten().copied().fold(0.0, f64::max) + 1.0) * (size + 1) as f64;
    for row in costs.iter_mut().take(scene.cards.len()) {
        for value in row.iter_mut().take(cards.len()) {
            if *value > f64::from(scene.corner_tolerance) {
                *value += penalty;
            }
        }
    }
    let assignment = minimum_assignment(&costs);
    let mut used = BTreeSet::new();
    let mut matched = Vec::new();
    let mut unmatched = Vec::new();
    let mut passed = cards.len() == scene.cards.len()
        && cards.iter().all(|card| {
            card.image.dimensions() == (250, 350)
                && card.detection.area.is_finite()
                && card.detection.area > 0.0
                && card
                    .detection
                    .corners
                    .iter()
                    .all(|p| p.x.is_finite() && p.y.is_finite())
        });
    for (i, expected) in scene.cards.iter().enumerate() {
        let j = assignment[i];
        let Some(actual) = cards.get(j) else {
            unmatched.push(i);
            passed = false;
            continue;
        };
        used.insert(j);
        let error = corner_error(&actual.detection, &expected.corners);
        let geometry_ok = error <= scene.corner_tolerance;
        let pixels = input.references[i].as_ref().and_then(|reference| {
            (actual.image.dimensions() == (250, 350))
                .then(|| interior_error(&actual.image, reference))
        });
        let pixel_ok = scene
            .pixel_mae_limit
            .is_none_or(|limit| pixels.is_some_and(|(error, _)| error < limit));
        passed &= geometry_ok && pixel_ok;
        matched.push(
            json!({"expected_index":i,"source":expected.source,"detected_index":j,
            "max_corner_error":error,"corner_tolerance":scene.corner_tolerance,
            "geometry_passed":geometry_ok,"interior_rgb_mae":pixels.map(|p|p.0),
            "pixel_orientation":pixels.map(|p|if p.1 {"rotated180"} else {"original"}),
            "pixel_mae_limit":scene.pixel_mae_limit,"pixels_passed":pixel_ok}),
        );
    }
    json!({"passed":passed,"expected_count":scene.cards.len(),"detected_count":cards.len(),
        "count_passed":scene.cards.len()==cards.len(),"matches":matched,"unmatched_expected":unmatched,
        "unmatched_detected":(0..cards.len()).filter(|j|!used.contains(j)).collect::<Vec<_>>(),
        "detections":cards.iter().map(|card|json!({"corners":card.detection.corners.map(|p|[p.x,p.y]),
            "area":card.detection.area,"crop_dimensions":[card.image.width(),card.image.height()]})).collect::<Vec<_>>()})
}

/// Measures the best winding-preserving cyclic alignment of all four outline corners.
fn corner_error(detected: &CardDetection, expected: &[[f32; 2]; 4]) -> f32 {
    (0..4)
        .map(|start| {
            (0..4)
                .map(|index| {
                    let point = detected.corners[(start + index) % 4];
                    (point.x - expected[index][0]).hypot(point.y - expected[index][1])
                })
                .fold(0.0, f32::max)
        })
        .fold(f32::INFINITY, f32::min)
}

/// Measures canonical interior RGB MAE, permitting the same 180° ambiguity as tests.
fn interior_error(actual: &RgbImage, reference: &RgbImage) -> (f32, bool) {
    let error = |rotated| {
        let mut total = 0_u64;
        for y in 20..330 {
            for x in 20..230 {
                let expected = if rotated {
                    reference.get_pixel(249 - x, 349 - y)
                } else {
                    reference.get_pixel(x, y)
                };
                for (actual, expected) in actual.get_pixel(x, y).0.iter().zip(expected.0) {
                    total += u64::from(actual.abs_diff(expected));
                }
            }
        }
        total as f32 / (210.0 * 310.0 * 3.0)
    };
    let original = error(false);
    let rotated = error(true);
    if original <= rotated {
        (original, false)
    } else {
        (rotated, true)
    }
}

/// Assigns unique detections with the scorer's square Hungarian minimization.
fn minimum_assignment(costs: &[Vec<f64>]) -> Vec<usize> {
    let size = costs.len();
    let mut row_potential = vec![0.0; size + 1];
    let mut column_potential = vec![0.0; size + 1];
    let mut row_at_column = vec![0; size + 1];
    let mut predecessor = vec![0; size + 1];
    for row in 1..=size {
        row_at_column[0] = row;
        let mut column = 0;
        let mut minimum = vec![f64::INFINITY; size + 1];
        let mut visited = vec![false; size + 1];
        loop {
            visited[column] = true;
            let current_row = row_at_column[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0;
            for candidate in 1..=size {
                if visited[candidate] {
                    continue;
                }
                let reduced = costs[current_row - 1][candidate - 1]
                    - row_potential[current_row]
                    - column_potential[candidate];
                if reduced < minimum[candidate] {
                    minimum[candidate] = reduced;
                    predecessor[candidate] = column;
                }
                if minimum[candidate] < delta {
                    delta = minimum[candidate];
                    next_column = candidate;
                }
            }
            for candidate in 0..=size {
                if visited[candidate] {
                    row_potential[row_at_column[candidate]] += delta;
                    column_potential[candidate] -= delta;
                } else {
                    minimum[candidate] -= delta;
                }
            }
            column = next_column;
            if row_at_column[column] == 0 {
                break;
            }
        }
        loop {
            let previous = predecessor[column];
            row_at_column[column] = row_at_column[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }
    let mut result = vec![0; size];
    for column in 1..=size {
        result[row_at_column[column] - 1] = column - 1;
    }
    result
}

/// Flattens samples from an exact selected scene set for an API.
fn collect_samples(measurements: &[&Measurement], extraction: bool) -> Vec<f64> {
    measurements
        .iter()
        .flat_map(|m| if extraction { &m.extract } else { &m.detect })
        .copied()
        .collect()
}

/// Reports nearest-rank percentiles and the fraction below the 15 FPS budget.
fn statistics(samples: &[f64]) -> Value {
    if samples.is_empty() {
        return Value::Null;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let percentile = |p: f64| {
        sorted[((sorted.len() as f64 * p).ceil() as usize)
            .saturating_sub(1)
            .min(sorted.len() - 1)]
    };
    json!({"samples":samples.len(),"median":percentile(0.5),"p95":percentile(0.95),
        "max":sorted.last(),"samples_within_15fps_budget":samples.iter().filter(|v|**v<=1000.0/15.0).count()})
}

/// Compares medians only when both variants cover the identical eligible scene set.
fn speedup(baseline: &[f64], variant: &[f64]) -> Option<f64> {
    let a = statistics(baseline)["median"].as_f64()?;
    let b = statistics(variant)["median"].as_f64()?;
    Some(a / b)
}

/// Records hardware, toolchain, source, and runtime details alongside each report.
fn metadata() -> Value {
    let command = |program: &str, args: &[&str]| {
        Command::new(program)
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    let cpu = fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
        text.lines()
            .find(|line| line.starts_with("model name") || line.starts_with("Hardware"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, value)| value.trim().to_owned())
    });
    // Embed the compiled source, so moving this binary to a phone or editing the
    // checkout after building cannot silently misidentify the measured code.
    let sources: [(&str, &[u8]); 8] = [
        ("src/lib.rs", include_bytes!("../src/lib.rs")),
        ("src/detection.rs", include_bytes!("../src/detection.rs")),
        ("src/geometry.rs", include_bytes!("../src/geometry.rs")),
        ("src/fast_edges.rs", include_bytes!("../src/fast_edges.rs")),
        ("Cargo.toml", include_bytes!("../Cargo.toml")),
        ("Cargo.lock", include_bytes!("../Cargo.lock")),
        ("examples/benchmark.rs", include_bytes!("benchmark.rs")),
        (
            "tests/support/mod.rs",
            include_bytes!("../tests/support/mod.rs"),
        ),
    ];
    let source_sha256: BTreeMap<_, _> = sources
        .into_iter()
        .map(|(name, bytes)| {
            (
                name,
                Sha256::digest(bytes)
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>(),
            )
        })
        .collect();
    let executable_sha256 = env::current_exe()
        .ok()
        .and_then(|path| fs::read(path).ok())
        .map(|bytes| {
            Sha256::digest(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        });
    json!({"unix_time_seconds":SystemTime::now().duration_since(UNIX_EPOCH).ok().map(|d|d.as_secs()),
        "os":env::consts::OS,"architecture":env::consts::ARCH,"cpu":cpu,
        "logical_cpus":std::thread::available_parallelism().ok().map(usize::from),
        "kernel":command("uname",&["-a"]),"rustc":command("rustc",&["-Vv"]),
        "git_revision":command("git",&["rev-parse","HEAD"]),"git_status":command("git",&["status","--short"]),
        "rustflags":env::var("RUSTFLAGS").ok(),"release":!cfg!(debug_assertions),
        "command":env::args().collect::<Vec<_>>(),
        "compiled_source_sha256":source_sha256,"executable_sha256":executable_sha256})
}
