//! Protect saved research scenes and source-face splits without running models.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

/// Read repository-owned fixture metadata without accessing any source cache.
fn manifest(name: &str) -> (std::path::PathBuf, Value) {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let bytes = fs::read(directory.join("manifest.json")).expect("saved fixture manifest");
    (
        directory,
        serde_json::from_slice(&bytes).expect("valid fixture metadata"),
    )
}

/// Check every permanent JPEG/reference and the strict new-corpus scoring rules.
#[test]
fn research_scene_bytes_and_scoring_rules_are_pinned() {
    for (name, expected_scenes) in [("learning", 768), ("temporal", 192), ("materials", 40)] {
        let (directory, data) = manifest(name);
        let hashes = data["sha256"].as_object().expect("asset checksums");
        for (filename, expected) in hashes {
            let relative = Path::new(filename);
            assert!(relative.is_relative(), "relative fixture paths required");
            assert!(
                relative
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_))),
                "fixture paths cannot escape their corpus"
            );
            let bytes = fs::read(directory.join(relative)).expect("saved image asset");
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                expected.as_str().expect("SHA256 string"),
                "changed saved asset in {name}: {filename}"
            );
        }
        let scenes = data["scenes"].as_array().expect("saved scenes");
        assert_eq!(scenes.len(), expected_scenes);
        let mut identifiers = BTreeSet::new();
        for scene in scenes {
            assert!(identifiers.insert(scene["id"].as_str().expect("scene ID")));
            let filename = scene["file"].as_str().expect("scene JPEG path");
            assert!(filename.ends_with(".jpg"));
            assert!(hashes.contains_key(filename), "scene bytes must be pinned");
            assert_eq!(scene["tier"], "challenge");
            assert_eq!(scene["corner_tolerance"].as_f64(), Some(8.0));
            let cards = scene["cards"]
                .as_array()
                .expect("expected physical outlines");
            if !cards.is_empty() {
                assert_eq!(scene["pixel_mae_limit"].as_f64(), Some(22.0));
            }
            for card in cards {
                assert_eq!(card["corners"].as_array().expect("full corners").len(), 4);
                let reference = card["reference"].as_str().expect("observed crop reference");
                assert!(
                    hashes.contains_key(reference),
                    "reference bytes must be pinned"
                );
            }
        }
    }
}

/// Keep unseen source artwork separate from training and preserve the old four.
#[test]
fn learning_source_faces_do_not_cross_partitions() {
    let (_, data) = manifest("learning");
    let (_, old) = manifest("scryfall");
    let old_ids = old["cards"]
        .as_array()
        .expect("original four source records")
        .iter()
        .map(|r| r["id"].as_str().expect("source ID"))
        .collect::<BTreeSet<_>>();
    let mut all_sources = BTreeSet::new();
    for (split, faces, scene_count) in [
        ("train", 36, 512),
        ("validation", 12, 128),
        ("test", 12, 128),
    ] {
        let sources = data["source_groups"][split]
            .as_array()
            .expect("source-face partition")
            .iter()
            .map(|v| v.as_str().expect("source ID"))
            .collect::<BTreeSet<_>>();
        assert_eq!(sources.len(), faces);
        assert!(sources.is_disjoint(&old_ids));
        for source in &sources {
            assert!(all_sources.insert(*source), "artwork crosses partitions");
        }
        let scenes = data["scenes"]
            .as_array()
            .expect("learning scenes")
            .iter()
            .filter(|s| s["split"] == split)
            .collect::<Vec<_>>();
        assert_eq!(scenes.len(), scene_count);
        let used = scenes
            .iter()
            .flat_map(|s| s["cards"].as_array().expect("cards"))
            .map(|c| c["source"].as_str().expect("source-face ID"))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            used, sources,
            "every scene must use its declared source split"
        );
    }
    let attribution: Value = serde_json::from_slice(
        &fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/learning/sources.json"),
        )
        .expect("permanent source attribution metadata"),
    )
    .expect("valid attribution metadata");
    let records = attribution["cards"]
        .as_array()
        .expect("credited source cards");
    assert_eq!(records.len(), 60);
    for field in ["id", "oracle_id", "illustration_id", "sha256"] {
        let mut identities = BTreeSet::new();
        for source in records {
            assert!(identities.insert(source[field].as_str().expect("source identity")));
        }
    }
    for source in records {
        let split = source["partition"].as_str().expect("source partition");
        assert!(
            data["source_groups"][split]
                .as_array()
                .expect("source-face partition")
                .contains(&source["id"])
        );
        assert!(!source["artist"].as_str().expect("artist credit").is_empty());
        assert!(
            source["source"]
                .as_str()
                .expect("source link")
                .starts_with("https://scryfall.com/card/")
        );
    }
}
