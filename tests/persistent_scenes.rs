//! Evaluate saved JPEG scenes against independently authored annotations and PNGs.

/// Shared fixture validation and scoring, also used by the batch evaluator example.
#[path = "support/mod.rs"]
pub mod scene_evaluation;

use scene_evaluation::{
    EvaluationOptions, EvaluationResult, default_manifest_path, evaluate_manifest,
    validate_fixture_integrity,
};
use std::path::PathBuf;

/// Returns the permanent archive of counterexamples found by adversarial search.
fn adversarial_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/adversarial/manifest.json")
}

/// Returns the permanent corpus of busy and marked playmat backgrounds.
fn playmat_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/playmats/manifest.json")
}

/// Returns saved whole-scene pinhole-camera views and their projected annotations.
fn camera_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/camera/manifest.json")
}

/// Missing, changed, or malformed permanent fixtures fail without downloading replacements.
#[test]
fn permanent_fixture_hashes_formats_and_annotations_are_valid() -> EvaluationResult<()> {
    validate_fixture_integrity(&default_manifest_path())
}

/// Archived counterexamples remain required, immutable, and independently inspectable.
#[test]
fn adversarial_archive_hashes_formats_and_annotations_are_valid() -> EvaluationResult<()> {
    validate_fixture_integrity(&adversarial_manifest_path())
}

/// Saved playmat scenes and references must exist with their approved bytes and metadata.
#[test]
fn playmat_scene_hashes_formats_and_annotations_are_valid() -> EvaluationResult<()> {
    validate_fixture_integrity(&playmat_manifest_path())
}

/// Camera-view images and references retain their approved geometry and byte hashes.
#[test]
fn camera_scene_hashes_formats_and_annotations_are_valid() -> EvaluationResult<()> {
    validate_fixture_integrity(&camera_manifest_path())
}

/// Required scenes retain exact counts, full outlines, and declared appearance fidelity.
#[test]
fn regression_scenes_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(&default_manifest_path(), &EvaluationOptions::default())?;
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Required busy-background scenes satisfy the same default extraction contract.
#[test]
fn playmat_regression_scenes_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(&playmat_manifest_path(), &EvaluationOptions::default())?;
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Required camera poses preserve card counts, full outlines, and observed appearance.
#[test]
fn camera_regression_scenes_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(&camera_manifest_path(), &EvaluationOptions::default())?;
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Explicit challenge evaluation reports every unsupported count, outline, or pixel error.
#[test]
#[ignore = "includes difficult challenge fixtures; use evaluate_scenes for a complete JSON report"]
fn challenge_scenes_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(
        &default_manifest_path(),
        &EvaluationOptions {
            include_challenges: true,
            scene_id: None,
        },
    )?;
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Running the known counterexamples explicitly must surface remaining extraction failures.
#[test]
#[ignore = "known adversarial counterexamples; use evaluate_scenes with the adversarial manifest for JSON metrics"]
fn archived_adversarial_scenes_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(
        &adversarial_manifest_path(),
        &EvaluationOptions {
            include_challenges: true,
            scene_id: None,
        },
    )?;
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Explicit playmat challenge evaluation surfaces every count, outline, or fidelity failure.
#[test]
#[ignore = "includes busy and marked playmat challenges; use evaluate_scenes with the playmats manifest for JSON metrics"]
fn playmat_challenge_scenes_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(
        &playmat_manifest_path(),
        &EvaluationOptions {
            include_challenges: true,
            scene_id: None,
        },
    )?;
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Explicit camera challenge evaluation reports unresolved pose-related extraction errors.
#[test]
#[ignore = "includes whole-scene camera-pose challenges; use evaluate_scenes with the camera manifest for JSON metrics"]
fn camera_challenge_scenes_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(
        &camera_manifest_path(),
        &EvaluationOptions {
            include_challenges: true,
            scene_id: None,
        },
    )?;
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}
