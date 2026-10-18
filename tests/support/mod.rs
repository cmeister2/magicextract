//! Load immutable annotated image fixtures and score extraction without rendering.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Component, Path, PathBuf},
};

use image::{DynamicImage, ImageFormat, RgbImage};
use magicextract::{CardDetection, CardExtractor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A fixture or scoring operation that can report a descriptive error.
pub type EvaluationResult<T> = Result<T, Box<dyn Error>>;

/// Whether a saved case is a required regression or an explicit challenge.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Required behavior included in the default test suite.
    Regression,
    /// A difficult case evaluated only when explicitly selected.
    Challenge,
}

/// A ground-truth card in original JPEG pixel coordinates.
#[derive(Clone, Debug, Deserialize)]
struct ExpectedCard {
    /// Stable source identifier used in diagnostics.
    source: String,
    /// Clockwise full-outline corners, including any hidden portion.
    corners: [[f32; 2]; 4],
    /// Annotated fraction of the physical card that remains visible.
    visible_fraction: f32,
    /// Optional scene-specific observed-appearance PNG instead of the source face.
    reference: Option<String>,
}

/// One immutable JPEG, its annotation, and its declared acceptance bounds.
#[derive(Debug, Deserialize)]
struct Scene {
    /// Unique identifier used for selection and reports.
    id: String,
    /// JPEG path relative to the manifest's directory.
    file: String,
    /// Whether the case is required by default.
    tier: Tier,
    /// Human-readable description of the saved case.
    description: String,
    /// Author-supplied scenario labels.
    tags: Vec<String>,
    /// Expected physical cards, excluding external occluders and printed panels.
    cards: Vec<ExpectedCard>,
    /// Maximum allowed displacement of any matched full-outline corner.
    corner_tolerance: f32,
    /// Optional explicit interior RGB MAE bound; null means geometry-only scoring.
    pixel_mae_limit: Option<f32>,
    /// Optional redundant author flag that must agree with the pixel bound.
    check_pixels: Option<bool>,
}

/// Versioned corpus annotations; extra author and provenance fields are accepted.
#[derive(Debug, Deserialize)]
struct Manifest {
    /// Supported schema version.
    format_version: u32,
    /// Clean source-face PNG paths keyed by source identifier.
    references: BTreeMap<String, String>,
    /// Lowercase SHA-256 pins keyed by relative PNG and JPEG paths.
    sha256: BTreeMap<String, String>,
    /// Saved scenes to validate and select for evaluation.
    scenes: Vec<Scene>,
}

/// A validated manifest and decoded appearance references.
struct Corpus {
    /// Directory against which fixture paths resolve.
    root: PathBuf,
    /// Parsed and validated ground truth.
    manifest: Manifest,
    /// Each appearance reference decoded once, keyed by its relative path.
    references: BTreeMap<String, RgbImage>,
}

/// Controls which saved scenes are scored; extractor settings always use defaults.
#[derive(Default)]
pub struct EvaluationOptions {
    /// Include challenge scenes alongside required regressions.
    pub include_challenges: bool,
    /// Select one exact scene identifier after applying the tier selection.
    pub scene_id: Option<String>,
}

/// Geometric orientation chosen when comparing the observed card to its reference.
#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelOrientation {
    /// The portrait reference's original orientation.
    Original,
    /// The permitted upside-down portrait orientation.
    Rotated180,
}

/// A single one-to-one assignment between an expected card and an extraction.
#[derive(Debug, Serialize)]
pub struct CardMatch {
    /// Index of the expected card in the scene annotation.
    pub expected_index: usize,
    /// Annotated source identifier.
    pub source: String,
    /// Index of the detected card in extractor output.
    pub detected_index: usize,
    /// Maximum corner error after the best winding-preserving cyclic alignment.
    pub corner_error: f32,
    /// Interior mean absolute error on the 0–255 RGB channel scale, when scored.
    pub pixel_mae: Option<f32>,
    /// Reference orientation used for the smaller pixel error, when scored.
    pub pixel_orientation: Option<PixelOrientation>,
}

/// Serializable detector output for visual diagnosis and later comparisons.
#[derive(Debug, Serialize)]
pub struct DetectionReport {
    /// Clockwise detector corners in original JPEG pixels.
    pub corners: [[f32; 2]; 4],
    /// Enclosed source-image area reported by the detector.
    pub area: f32,
}

/// Complete metrics and failures for one saved JPEG.
#[derive(Debug, Serialize)]
pub struct SceneReport {
    /// Stable scene identifier.
    pub id: String,
    /// Required or explicitly selected challenge classification.
    pub tier: Tier,
    /// Whether every declared count, corner, and pixel check passed.
    pub passed: bool,
    /// Ground-truth number of physical cards.
    pub expected_count: usize,
    /// Number of returned extracted cards.
    pub detected_count: usize,
    /// Original detector polygons, in extractor output order.
    pub detected: Vec<DetectionReport>,
    /// Unique expected-to-detected assignments.
    pub matches: Vec<CardMatch>,
    /// Ground-truth indexes that received no assignment.
    pub unmatched_expected: Vec<usize>,
    /// Detector indexes that received no assignment.
    pub unmatched_detected: Vec<usize>,
    /// Explanations of every failed acceptance criterion.
    pub failures: Vec<String>,
}

/// Aggregate counts and worst measured errors for a selected corpus.
#[derive(Debug, Default, Serialize)]
pub struct Summary {
    /// Number of evaluated JPEG scenes.
    pub scenes: usize,
    /// Number of scenes passing all acceptance criteria.
    pub passed: usize,
    /// Number of scenes with one or more failures.
    pub failed: usize,
    /// Total annotated physical cards.
    pub expected_cards: usize,
    /// Total extracted cards, including false positives.
    pub detected_cards: usize,
    /// Number of one-to-one card assignments, including assignments outside bounds.
    pub matched_cards: usize,
    /// Expected cards without an available detected counterpart.
    pub unmatched_expected: usize,
    /// Detected cards without an available expected counterpart.
    pub unmatched_detected: usize,
    /// Assignments with an explicit appearance-fidelity check.
    pub pixel_checks: usize,
    /// Largest assigned maximum-corner error, if any card was matched.
    pub max_corner_error: Option<f32>,
    /// Largest measured interior RGB MAE, if any pixels were checked.
    pub max_pixel_mae: Option<f32>,
}

/// Machine-readable batch result, including failed scenes rather than hiding them.
#[derive(Debug, Serialize)]
pub struct Report {
    /// Report schema version.
    pub format_version: u32,
    /// Input manifest path as supplied to the evaluator.
    pub manifest: String,
    /// Aggregate evaluation metrics.
    pub summary: Summary,
    /// Per-scene metrics in manifest order.
    pub scenes: Vec<SceneReport>,
}

impl Report {
    /// Formats only failed cases for concise assertion diagnostics.
    pub fn failure_summary(&self) -> String {
        self.scenes
            .iter()
            .filter(|scene| !scene.passed)
            .map(|scene| format!("{}: {}", scene.id, scene.failures.join("; ")))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Returns the source checkout's permanent primary corpus manifest.
pub fn default_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scenes/manifest.json")
}

/// Checks every saved fixture's hash, format, annotation, and reference dimensions.
///
/// # Errors
///
/// Returns an error for missing or changed files, invalid paths, unsupported
/// schema, malformed ground truth, or non-default per-scene extractor overrides.
pub fn validate_fixture_integrity(path: &Path) -> EvaluationResult<()> {
    load_corpus(path).map(|_| ())
}

/// Scores selected permanent JPEG scenes using the default card extractor.
///
/// # Errors
///
/// Returns an error for invalid or missing fixture data, an empty selection, or
/// an unknown selected identifier. Detection failures become failed scene reports.
pub fn evaluate_manifest(path: &Path, options: &EvaluationOptions) -> EvaluationResult<Report> {
    let corpus = load_corpus(path)?;
    let selected: Vec<_> = corpus
        .manifest
        .scenes
        .iter()
        .filter(|scene| {
            (scene.tier == Tier::Regression || options.include_challenges)
                && options.scene_id.as_ref().is_none_or(|id| id == &scene.id)
        })
        .collect();
    if selected.is_empty() {
        return Err("no scenes selected; check --scene and --include-challenges".into());
    }
    let mut scenes = Vec::with_capacity(selected.len());
    for scene in selected {
        let bytes = read_verified(&corpus.root, &scene.file, &corpus.manifest.sha256)?;
        let image = image::load_from_memory_with_format(&bytes, ImageFormat::Jpeg)?;
        scenes.push(score_scene(scene, &image, &corpus));
    }
    let summary = summarize(&scenes);
    Ok(Report {
        format_version: 1,
        manifest: path.display().to_string(),
        summary,
        scenes,
    })
}

/// Loads and validates the complete corpus, including unselected challenge files.
///
/// # Errors
///
/// Returns an error when schema, annotations, hashes, paths, or image formats fail validation.
fn load_corpus(path: &Path) -> EvaluationResult<Corpus> {
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let scene_values = value
        .get("scenes")
        .and_then(serde_json::Value::as_array)
        .ok_or("manifest scenes must be an array")?;
    for scene in scene_values {
        if scene.get("pixel_mae_limit").is_none() {
            return Err(
                "each scene must explicitly specify pixel_mae_limit, including null".into(),
            );
        }
        if scene
            .get("extractor_config")
            .is_some_and(|config| !config.is_null())
        {
            return Err(
                "per-scene extractor_config overrides are unsupported; score defaults".into(),
            );
        }
    }
    let manifest: Manifest = serde_json::from_value(value)?;
    if manifest.format_version != 1 || manifest.scenes.is_empty() {
        return Err("manifest requires format_version 1 and at least one scene".into());
    }
    let root = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let mut identifiers = BTreeSet::new();
    let mut reference_paths: BTreeSet<_> = manifest.references.values().cloned().collect();
    for scene in &manifest.scenes {
        if scene.id.trim().is_empty()
            || !identifiers.insert(scene.id.clone())
            || scene.description.trim().is_empty()
            || scene.tags.iter().any(|tag| tag.trim().is_empty())
        {
            return Err(format!("empty or duplicate scene metadata: {}", scene.id).into());
        }
        if !scene.corner_tolerance.is_finite()
            || scene.corner_tolerance <= 0.0
            || scene
                .pixel_mae_limit
                .is_some_and(|limit| !limit.is_finite() || limit <= 0.0 || limit > 255.0)
            || scene
                .check_pixels
                .is_some_and(|enabled| enabled != scene.pixel_mae_limit.is_some())
        {
            return Err(format!("invalid scoring bounds for {}", scene.id).into());
        }
        for card in &scene.cards {
            validate_card(card, &scene.id)?;
            let reference = card
                .reference
                .as_ref()
                .or_else(|| manifest.references.get(&card.source));
            if let Some(reference) = reference {
                reference_paths.insert(reference.clone());
            } else if scene.pixel_mae_limit.is_some() {
                return Err(format!(
                    "missing appearance reference for {} in {}",
                    card.source, scene.id
                )
                .into());
            }
        }
        let bytes = read_verified(&root, &scene.file, &manifest.sha256)?;
        validate_image_bytes(&bytes, ImageFormat::Jpeg, &scene.file, false)?;
    }
    let mut references = BTreeMap::new();
    for relative in reference_paths {
        let bytes = read_verified(&root, &relative, &manifest.sha256)?;
        let image = validate_image_bytes(&bytes, ImageFormat::Png, &relative, true)?.into_rgb8();
        references.insert(relative, image);
    }
    Ok(Corpus {
        root,
        manifest,
        references,
    })
}

/// Validates finite, clockwise, convex full outlines and meaningful visibility metadata.
///
/// # Errors
///
/// Returns an error for empty identifiers, nonfinite coordinates, degenerate or
/// reflected polygons, or visibility fractions outside zero through one.
fn validate_card(card: &ExpectedCard, scene_id: &str) -> EvaluationResult<()> {
    if card.source.trim().is_empty()
        || !card.visible_fraction.is_finite()
        || !(0.0..=1.0).contains(&card.visible_fraction)
        || card
            .corners
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(format!("invalid card metadata in {scene_id}").into());
    }
    for index in 0..4 {
        let first = card.corners[index];
        let second = card.corners[(index + 1) % 4];
        let third = card.corners[(index + 2) % 4];
        let turn = (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0]);
        if !turn.is_finite() || turn <= 0.0 {
            return Err(format!("card corners must be convex and clockwise in {scene_id}").into());
        }
    }
    Ok(())
}

/// Reads a normal relative fixture path and verifies its mandatory SHA-256 pin.
///
/// # Errors
///
/// Returns an error for absolute or parent-traversing paths, missing pins, I/O,
/// malformed hashes, or a digest mismatch.
fn read_verified(
    root: &Path,
    relative: &str,
    hashes: &BTreeMap<String, String>,
) -> EvaluationResult<Vec<u8>> {
    let path = Path::new(relative);
    if relative.is_empty()
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(format!("fixture paths must be normal relative paths: {relative}").into());
    }
    let expected = hashes
        .get(relative)
        .ok_or_else(|| format!("missing SHA-256 pin: {relative}"))?;
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("invalid lowercase SHA-256 pin: {relative}").into());
    }
    let bytes = fs::read(root.join(path))
        .map_err(|error| format!("cannot read fixture {relative}: {error}"))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if &actual != expected {
        return Err(
            format!("SHA-256 mismatch for {relative}: expected {expected}, got {actual}").into(),
        );
    }
    Ok(bytes)
}

/// Verifies the encoded format and dimensions before any pixel comparison.
///
/// # Errors
///
/// Returns an error for a wrong codec, corrupt image, zero dimension, or a
/// reference that is not the canonical 250 by 350 pixels.
fn validate_image_bytes(
    bytes: &[u8],
    format: ImageFormat,
    label: &str,
    reference: bool,
) -> EvaluationResult<DynamicImage> {
    if image::guess_format(bytes)? != format {
        return Err(format!("wrong encoded image format for {label}: expected {format:?}").into());
    }
    let image = image::load_from_memory_with_format(bytes, format)?;
    if image.width() == 0
        || image.height() == 0
        || (reference && (image.width(), image.height()) != (250, 350))
    {
        return Err(format!("invalid image dimensions for {label}").into());
    }
    Ok(image)
}

/// Scores all detections against their optimal unique annotated counterparts.
fn score_scene(scene: &Scene, image: &DynamicImage, corpus: &Corpus) -> SceneReport {
    let mut report = SceneReport {
        id: scene.id.clone(),
        tier: scene.tier,
        passed: false,
        expected_count: scene.cards.len(),
        detected_count: 0,
        detected: Vec::new(),
        matches: Vec::new(),
        unmatched_expected: Vec::new(),
        unmatched_detected: Vec::new(),
        failures: Vec::new(),
    };
    let cards = match CardExtractor::default().extract(image) {
        Ok(cards) => cards,
        Err(error) => {
            report.failures.push(format!("extraction failed: {error}"));
            report.unmatched_expected.extend(0..scene.cards.len());
            return report;
        }
    };
    report.detected_count = cards.len();
    report.detected = cards
        .iter()
        .map(|card| DetectionReport {
            corners: card.detection.corners.map(|point| [point.x, point.y]),
            area: card.detection.area,
        })
        .collect();
    for (index, card) in cards.iter().enumerate() {
        if card.image.dimensions() != (250, 350) {
            report.failures.push(format!(
                "detected card {index}: crop dimensions {:?} differ from 250x350",
                card.image.dimensions()
            ));
        }
        if !card.detection.area.is_finite()
            || card.detection.area <= 0.0
            || card
                .detection
                .corners
                .iter()
                .any(|point| !point.x.is_finite() || !point.y.is_finite())
        {
            report.failures.push(format!(
                "detected card {index}: invalid outline coordinates or area"
            ));
        }
    }
    if cards.len() != scene.cards.len() {
        report.failures.push(format!(
            "expected {} cards, detected {}",
            scene.cards.len(),
            cards.len()
        ));
    }
    let size = cards.len().max(scene.cards.len());
    let mut costs = vec![vec![0.0_f64; size]; size];
    for (expected_index, expected) in scene.cards.iter().enumerate() {
        for (detected_index, detected) in cards.iter().enumerate() {
            costs[expected_index][detected_index] =
                f64::from(corner_error(&detected.detection, &expected.corners));
        }
    }
    // First maximize pairs inside the acceptance bound; then minimize total error.
    let largest_error = costs.iter().flatten().copied().fold(0.0, f64::max);
    let outside_penalty = (largest_error + 1.0) * (size + 1) as f64;
    for row in costs.iter_mut().take(scene.cards.len()) {
        for cost in row.iter_mut().take(cards.len()) {
            if *cost > f64::from(scene.corner_tolerance) {
                *cost += outside_penalty;
            }
        }
    }
    let assignments = minimum_assignment(&costs);
    let mut used_detections = BTreeSet::new();
    for (expected_index, expected) in scene.cards.iter().enumerate() {
        let detected_index = assignments[expected_index];
        if detected_index >= cards.len() {
            report.unmatched_expected.push(expected_index);
            continue;
        }
        used_detections.insert(detected_index);
        let detected = &cards[detected_index];
        let error = corner_error(&detected.detection, &expected.corners);
        if error > scene.corner_tolerance {
            report.failures.push(format!(
                "card {expected_index} ({}): corner error {error:.3} > {}",
                expected.source, scene.corner_tolerance
            ));
        }
        let (pixel_mae, pixel_orientation) = if let Some(limit) = scene.pixel_mae_limit {
            let reference_path = expected
                .reference
                .as_ref()
                .or_else(|| corpus.manifest.references.get(&expected.source));
            let reference = reference_path.and_then(|path| corpus.references.get(path));
            match reference.filter(|_| detected.image.dimensions() == (250, 350)) {
                Some(reference) => {
                    let (mae, orientation) = interior_error(&detected.image, reference);
                    if mae >= limit {
                        report.failures.push(format!(
                            "card {expected_index} ({}): interior RGB MAE {mae:.3} >= {limit}",
                            expected.source
                        ));
                    }
                    (Some(mae), Some(orientation))
                }
                None => {
                    report.failures.push(format!(
                        "card {expected_index}: missing reference or noncanonical crop dimensions"
                    ));
                    (None, None)
                }
            }
        } else {
            (None, None)
        };
        report.matches.push(CardMatch {
            expected_index,
            source: expected.source.clone(),
            detected_index,
            corner_error: error,
            pixel_mae,
            pixel_orientation,
        });
    }
    report.unmatched_detected = (0..cards.len())
        .filter(|index| !used_detections.contains(index))
        .collect();
    report.passed = report.failures.is_empty();
    report
}

/// Finds the best cyclic corner alignment without permitting reflection.
fn corner_error(detected: &CardDetection, expected: &[[f32; 2]; 4]) -> f32 {
    if detected
        .corners
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return f32::MAX;
    }
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

/// Measures observed interior RGB fidelity, allowing a 180-degree portrait rotation.
fn interior_error(actual: &RgbImage, reference: &RgbImage) -> (f32, PixelOrientation) {
    let error = |rotated: bool| {
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
        (original, PixelOrientation::Original)
    } else {
        (rotated, PixelOrientation::Rotated180)
    }
}

/// Solves a square minimum-cost assignment with the Hungarian algorithm.
///
/// Returns one unique column for each row; zero-cost padding handles unequal counts.
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

/// Aggregates every selected scene, retaining failures and unmatched-card counts.
fn summarize(scenes: &[SceneReport]) -> Summary {
    let mut summary = Summary {
        scenes: scenes.len(),
        ..Summary::default()
    };
    for scene in scenes {
        summary.passed += usize::from(scene.passed);
        summary.failed += usize::from(!scene.passed);
        summary.expected_cards += scene.expected_count;
        summary.detected_cards += scene.detected_count;
        summary.matched_cards += scene.matches.len();
        summary.unmatched_expected += scene.unmatched_expected.len();
        summary.unmatched_detected += scene.unmatched_detected.len();
        for matched in &scene.matches {
            summary.max_corner_error = Some(
                summary
                    .max_corner_error
                    .unwrap_or(0.0)
                    .max(matched.corner_error),
            );
            if let Some(mae) = matched.pixel_mae {
                summary.pixel_checks += 1;
                summary.max_pixel_mae = Some(summary.max_pixel_mae.unwrap_or(0.0).max(mae));
            }
        }
    }
    summary
}

#[cfg(test)]
/// Focused scorer regressions that prevent false matches from appearing successful.
mod tests {
    use super::*;
    use magicextract::Point;

    /// Global one-to-one assignment avoids a tempting but worse greedy first match.
    #[test]
    fn assignment_is_unique_and_globally_minimal() {
        assert_eq!(
            minimum_assignment(&[vec![1.0, 2.0], vec![1.0, 10.0]]),
            vec![1, 0]
        );
        assert!(minimum_assignment(&[]).is_empty());
    }

    /// Cyclic starts preserve a polygon, whereas reversed winding is a reflected mapping.
    #[test]
    fn corner_alignment_allows_cyclic_starts_but_rejects_reflection() {
        let expected = [[0.0, 0.0], [10.0, 0.0], [10.0, 14.0], [0.0, 14.0]];
        let corners = expected.map(|[x, y]| Point { x, y });
        let mut detection = CardDetection {
            corners: [corners[2], corners[3], corners[0], corners[1]],
            area: 140.0,
        };
        assert_eq!(corner_error(&detection, &expected), 0.0);
        detection.corners.reverse();
        assert!(corner_error(&detection, &expected) >= 10.0);
    }
}
