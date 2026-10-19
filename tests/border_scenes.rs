//! Validate saved white- and silver-border scenes without authoring or downloading.

/// Shared immutable fixture loader and scorer used by the gallery evaluator.
#[path = "support/mod.rs"]
pub mod scene_evaluation;

use scene_evaluation::{
    EvaluationOptions, EvaluationResult, evaluate_manifest, validate_fixture_integrity,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

/// Returns the permanent white- and silver-border corpus manifest.
fn border_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/borders/manifest.json")
}

/// All saved scenes, references, source credits, and scoring bounds remain pinned.
#[test]
fn border_scene_bytes_and_provenance_are_valid() -> EvaluationResult<()> {
    let manifest_path = border_manifest_path();
    validate_fixture_integrity(&manifest_path)?;
    let directory = manifest_path.parent().expect("fixture directory");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
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
                .expect("pinned authoring metadata"),
            "changed authoring metadata: {filename}"
        );
    }
    let scenes = manifest["scenes"].as_array().expect("saved scenes");
    assert_eq!(scenes.len(), 10);
    assert_eq!(
        scenes
            .iter()
            .filter(|scene| scene["tier"] == "regression")
            .count(),
        7
    );
    assert_eq!(
        scenes
            .iter()
            .filter(|scene| scene["tier"] == "challenge")
            .count(),
        3
    );
    for scene in scenes {
        let identifier = scene["id"].as_str().expect("scene ID");
        let challenge =
            identifier.ends_with("-sleeve") || identifier == "white-border-playmat-upper-right";
        assert_eq!(
            scene["tier"],
            if challenge { "challenge" } else { "regression" }
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

/// Required plain and busy scenes preserve full physical borders and observed faces.
#[test]
fn white_and_silver_cards_match_saved_ground_truth() -> EvaluationResult<()> {
    let report = evaluate_manifest(&border_manifest_path(), &EvaluationOptions::default())?;
    assert_eq!(report.summary.scenes, 7);
    assert_eq!(report.summary.failed, 0, "{}", report.failure_summary());
    Ok(())
}

/// Each border color uses the same three visibly different playmat placements.
#[test]
fn border_playmat_placements_cover_position_rotation_and_scale() -> EvaluationResult<()> {
    let manifest: Value = serde_json::from_slice(&fs::read(border_manifest_path())?)?;
    let scenes = manifest["scenes"].as_array().expect("saved scenes");
    let playmats: Vec<_> = scenes
        .iter()
        .filter(|scene| {
            scene["tags"]
                .as_array()
                .expect("scene tags")
                .iter()
                .any(|tag| tag == "playmat")
        })
        .collect();
    assert_eq!(playmats.len(), 6);
    for source in ["white-lightning-bolt", "silver-booster-tutor"] {
        let paired: Vec<_> = playmats
            .iter()
            .filter(|scene| scene["cards"][0]["source"] == source)
            .collect();
        assert_eq!(paired.len(), 3);
        let measurements: Vec<_> =
            paired
                .iter()
                .map(|scene| {
                    let corners = scene["cards"][0]["corners"]
                        .as_array()
                        .expect("full physical corners");
                    let points: Vec<_> = corners
                        .iter()
                        .map(|point| {
                            (
                                point[0].as_f64().expect("corner x"),
                                point[1].as_f64().expect("corner y"),
                            )
                        })
                        .collect();
                    assert!(points.iter().all(|&(x, y)| {
                        (10.0..630.0).contains(&x) && (10.0..510.0).contains(&y)
                    }));
                    let center_x = points.iter().map(|point| point.0).sum::<f64>() / 4.0;
                    let center_y = points.iter().map(|point| point.1).sum::<f64>() / 4.0;
                    let dx = points[1].0 - points[0].0;
                    let dy = points[1].1 - points[0].1;
                    assert!(playmats.iter().any(|other| {
                        other["cards"][0]["source"] != source
                            && other["cards"][0]["corners"] == scene["cards"][0]["corners"]
                    }));
                    [center_x, center_y, dx.hypot(dy), dy.atan2(dx).to_degrees()]
                })
                .collect();
        for (dimension, minimum_span) in [240.0, 170.0, 25.0, 50.0].into_iter().enumerate() {
            let minimum = measurements
                .iter()
                .map(|measurement| measurement[dimension])
                .fold(f64::INFINITY, f64::min);
            let maximum = measurements
                .iter()
                .map(|measurement| measurement[dimension])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                maximum - minimum >= minimum_span,
                "playmat position, rotation, or scale collapsed for {source}: dimension {dimension}"
            );
        }
    }
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
