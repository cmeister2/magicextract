//! Evaluate saved JPEG scenes against independently authored annotations and PNGs.

/// Shared fixture validation and scoring, also used by the batch evaluator example.
#[path = "support/mod.rs"]
pub mod scene_evaluation;

use scene_evaluation::{
    EvaluationOptions, EvaluationResult, default_manifest_path, evaluate_manifest,
    validate_fixture_integrity,
};
use std::path::PathBuf;

/// Checks one archived example's counts, full corners, and unchanged crop appearance.
fn assert_saved_scene_matches(manifest: PathBuf, scene_id: &str) -> EvaluationResult<()> {
    let report = evaluate_manifest(
        &manifest,
        &EvaluationOptions {
            include_challenges: true,
            scene_id: Some(scene_id.to_owned()),
        },
    )?;
    assert_eq!(report.summary.scenes, 1);
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

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

/// The dim counterexample retains its observed brightness after outline recovery.
#[test]
fn dim_card_matches_saved_ground_truth() -> EvaluationResult<()> {
    assert_saved_scene_matches(
        adversarial_manifest_path(),
        "seed-20261018-case-0048-dim-reduction-2",
    )
}

/// The physical card border is retained instead of its inner printed frame.
#[test]
fn busy_playmat_card_matches_saved_ground_truth() -> EvaluationResult<()> {
    assert_saved_scene_matches(
        playmat_manifest_path(),
        "playmat-fantasy-forest-lightning-bolt",
    )
}

/// A sleeve boundary does not replace the physical outline of the card inside it.
#[test]
fn sleeved_card_matches_saved_ground_truth() -> EvaluationResult<()> {
    assert_saved_scene_matches(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/materials/manifest.json"),
        "sleeve-lightning-bolt-1",
    )
}

/// Covered corners use the full observed card outlines without replacing occluding pixels.
#[test]
fn overlapping_cards_match_saved_ground_truth() -> EvaluationResult<()> {
    for scene_id in ["occlusion-two-cards-1", "occlusion-two-cards-2"] {
        assert_saved_scene_matches(default_manifest_path(), scene_id)?;
    }
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
