//! Validate saved border and art-treatment scenes without authoring or downloading.

/// Shared immutable fixture loader and scorer used by the gallery evaluator.
#[path = "support/mod.rs"]
pub mod scene_evaluation;

use scene_evaluation::{
    EvaluationOptions, EvaluationResult, evaluate_manifest, validate_fixture_integrity,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Returns the permanent white- and silver-border corpus manifest.
fn border_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/borders/manifest.json")
}

/// Returns the permanent borderless and extended-art corpus manifest.
fn art_treatment_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/art-treatments/manifest.json")
}

/// Checks immutable image assets and their recorded authoring inputs.
///
/// # Errors
///
/// Returns an error for missing or malformed fixtures and authoring metadata.
///
/// # Panics
///
/// Panics if authoring metadata differs from its pinned hash or recorded path.
fn saved_authoring_metadata(manifest_path: &Path) -> EvaluationResult<Value> {
    validate_fixture_integrity(manifest_path)?;
    let directory = manifest_path.parent().ok_or("missing fixture directory")?;
    let manifest: Value = serde_json::from_slice(&fs::read(manifest_path)?)?;
    for (filename, hash_key) in [
        ("recipes.json", "recipes_sha256"),
        ("sources.json", "source_manifest_sha256"),
    ] {
        assert_eq!(
            Sha256::digest(fs::read(directory.join(filename))?)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            manifest["generator"][hash_key]
                .as_str()
                .ok_or("missing authoring metadata checksum")?,
            "changed authoring metadata: {filename}"
        );
    }
    let corpus = directory
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("missing fixture corpus name")?;
    for (key, filename) in [
        ("recipes", "recipes.json"),
        ("source_manifest", "sources.json"),
    ] {
        assert_eq!(
            manifest["generator"][key],
            format!("tests/fixtures/{corpus}/{filename}"),
        );
    }
    assert_eq!(
        manifest["generator"]["script"],
        "scripts/build-border-fixtures.py",
    );
    assert_eq!(
        manifest["generator"]["renderer"],
        "scripts/build-scene-fixtures.py",
    );
    Ok(manifest)
}

/// All saved scenes, references, source credits, and scoring bounds remain pinned.
#[test]
fn border_scene_bytes_and_provenance_are_valid() -> EvaluationResult<()> {
    let manifest_path = border_manifest_path();
    let manifest = saved_authoring_metadata(&manifest_path)?;
    let directory = manifest_path.parent().expect("fixture directory");
    let scenes = manifest["scenes"].as_array().expect("saved scenes");
    assert_eq!(scenes.len(), 6);
    assert_eq!(
        scenes
            .iter()
            .filter(|scene| scene["tier"] == "regression")
            .count(),
        4
    );
    assert_eq!(
        scenes
            .iter()
            .filter(|scene| scene["tier"] == "challenge")
            .count(),
        2
    );
    for scene in scenes {
        let sleeve = scene["id"].as_str().expect("scene ID").ends_with("-sleeve");
        assert_eq!(
            scene["tier"],
            if sleeve { "challenge" } else { "regression" }
        );
        assert_eq!(scene["corner_tolerance"].as_f64(), Some(8.0));
        assert_eq!(scene["pixel_mae_limit"].as_f64(), Some(22.0));
        assert_eq!(scene["cards"].as_array().expect("physical card").len(), 1);
    }
    let sources: Value = serde_json::from_slice(&fs::read(directory.join("sources.json"))?)?;
    let records = sources["cards"].as_array().expect("credited source faces");
    assert_eq!(records.len(), 2);
    for (key, color) in [
        ("white-lightning-bolt", "white"),
        ("silver-booster-tutor", "silver"),
    ] {
        let record = records
            .iter()
            .find(|record| record["key"] == key)
            .expect("declared border source");
        assert_eq!(record["border_color"], color);
        assert_eq!(manifest["sources"][key], *record);
        assert!(!record["artist"].as_str().expect("artist credit").is_empty());
        assert!(
            record["scryfall_uri"]
                .as_str()
                .expect("source page")
                .starts_with("https://scryfall.com/card/")
        );
    }
    Ok(())
}

/// Full physical outlines and original treatment metadata remain pinned challenges.
#[test]
fn art_treatment_scene_bytes_and_provenance_are_valid() -> EvaluationResult<()> {
    let manifest_path = art_treatment_manifest_path();
    let manifest = saved_authoring_metadata(&manifest_path)?;
    let directory = manifest_path.parent().expect("fixture directory");
    let scenes = manifest["scenes"].as_array().expect("saved scenes");
    let recipes: Value = serde_json::from_slice(&fs::read(directory.join("recipes.json"))?)?;
    let recipes = recipes.as_array().expect("authored scene recipes");
    let sources: Value = serde_json::from_slice(&fs::read(directory.join("sources.json"))?)?;
    let records = sources["cards"].as_array().expect("credited source faces");
    assert_eq!(scenes.len(), 6);
    assert_eq!(recipes.len(), 6);
    assert_eq!(records.len(), 2);
    assert_eq!(
        manifest["sha256"].as_object().expect("asset pins").len(),
        12
    );
    for (treatment, key, border_color) in [
        ("borderless", "borderless-lightning-bolt", "borderless"),
        ("extended-art", "extended-once-upon-a-time", "black"),
    ] {
        let record = records
            .iter()
            .find(|record| record["key"] == key)
            .expect("declared treatment source");
        assert_eq!(record["border_color"], border_color);
        assert_eq!(record["layout"], "normal");
        assert_eq!(manifest["sources"][key], *record);
        assert!(!record["artist"].as_str().expect("artist credit").is_empty());
        assert!(
            record["scryfall_uri"]
                .as_str()
                .expect("source page")
                .starts_with("https://scryfall.com/card/")
        );
        assert!(
            !directory
                .join(record["file"].as_str().expect("cached source filename"))
                .exists(),
            "original source PNGs belong only in the offline authoring cache"
        );
        if treatment == "extended-art" {
            assert!(
                record["frame_effects"]
                    .as_array()
                    .expect("source frame treatments")
                    .iter()
                    .any(|effect| effect == "extendedart")
            );
        }
        for surface in ["plain", "playmat", "sleeve"] {
            let id = format!("{treatment}-{surface}");
            let scene = scenes
                .iter()
                .find(|scene| scene["id"] == id)
                .expect("saved treatment scene");
            let recipe = recipes
                .iter()
                .find(|recipe| recipe["id"] == id)
                .expect("pinned authoring recipe");
            assert_eq!(scene["tier"], "challenge");
            assert_eq!(scene["corner_tolerance"].as_f64(), Some(8.0));
            assert_eq!(scene["pixel_mae_limit"].as_f64(), Some(22.0));
            assert_eq!(scene["params"], *recipe);
            assert_eq!(recipe["appearance_reference"], true);
            let cards = scene["cards"].as_array().expect("physical card outline");
            assert_eq!(cards.len(), 1);
            assert_eq!(cards[0]["source"], key);
            assert_eq!(cards[0]["corners"], recipe["cards"][0]["corners"]);
            assert_eq!(cards[0]["visible_fraction"].as_f64(), Some(1.0));
        }
    }
    Ok(())
}

/// Required plain and busy scenes preserve full physical borders and observed faces.
#[test]
fn white_and_silver_cards_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(&border_manifest_path(), &EvaluationOptions::default())?;
    assert_eq!(report.summary.scenes, 4);
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Sleeve recovery must preserve both pale physical frames and their saved appearance.
#[test]
fn white_and_silver_sleeved_cards_match_saved_ground_truth() -> EvaluationResult<()> {
    for scene_id in ["white-border-sleeve", "silver-border-sleeve"] {
        let report = evaluate_manifest(
            &border_manifest_path(),
            &EvaluationOptions {
                include_challenges: true,
                scene_id: Some(scene_id.to_owned()),
            },
        )?;
        assert_eq!(report.summary.scenes, 1);
        assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    }
    Ok(())
}
