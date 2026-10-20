//! Compare complete extraction with rectification of already known corners.
//!
//! Build and run through `run.py`; fixture verification and build metadata live
//! in that wrapper. This helper decodes and checks every input before timing.

use std::{env, error::Error, hint::black_box, time::Instant};

use magicextract::{
    CardExtractor, DetectionMode, ExtractorConfig, Point,
    image::{self, DynamicImage},
};

/// Repeated calls per operation and scene; the five warmups are excluded.
const ITERATIONS: usize = 31;
/// Untimed warmup calls per operation and scene.
const WARMUPS: usize = 5;

/// A decoded RGB8 scene and the source-space corners obtained before timing.
struct Scene {
    /// Input path, used only for metadata emitted after measurement.
    path: String,
    /// Owned decoded pixels, shared by both operations.
    frame: DynamicImage,
    /// The single card's corners, found and checked before measurement.
    corners: [Point; 4],
}

/// Measure extraction while excluding output destruction and validation.
fn extract_ms(extractor: &CardExtractor, scene: &Scene) -> Result<f64, Box<dyn Error>> {
    let started = Instant::now();
    let result = black_box(extractor.extract(black_box(&scene.frame)));
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let cards = result?;
    black_box(&cards);
    Ok(elapsed)
}

/// Measure known-corner rectification, including its validation and allocation.
fn rectify_ms(extractor: &CardExtractor, scene: &Scene) -> Result<f64, Box<dyn Error>> {
    let started = Instant::now();
    let result = black_box(extractor.rectify(black_box(&scene.frame), black_box(&scene.corners)));
    let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
    let crop = result?;
    black_box(&crop);
    Ok(elapsed)
}

/// Serialize finite timings with enough precision for independent summaries.
fn samples_json(samples: &[f64]) -> String {
    samples
        .iter()
        .map(|sample| format!("{sample:.9}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Print numeric results only after all scenes finish their timing loops.
fn main() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("the crop benchmark requires an optimized release build".into());
    }
    let paths: Vec<_> = env::args().skip(1).collect();
    if paths.len() != 3 {
        return Err("expected exactly three saved fixture image paths".into());
    }
    let extractor = CardExtractor::new(ExtractorConfig {
        max_detection_dimension: 640,
        detection_mode: DetectionMode::Full,
        ..ExtractorConfig::default()
    })?;
    let mut scenes = Vec::new();
    for path in paths {
        // Decode once and require identical detected geometry and crop pixels before timing.
        let frame = DynamicImage::ImageRgb8(image::open(&path)?.into_rgb8());
        let detections = extractor.detect(&frame)?;
        if detections.len() != 1 {
            return Err(format!("expected exactly one detected card: {path}").into());
        }
        let corners = detections[0].corners;
        let extracted = extractor.extract(&frame)?;
        if extracted.len() != 1 || extracted[0].detection.corners != corners {
            return Err(format!("detection and extraction disagree: {path}").into());
        }
        let rectified = extractor.rectify(&frame, &corners)?;
        if rectified != extracted[0].image {
            return Err(format!("rectify and extract pixels differ: {path}").into());
        }
        scenes.push(Scene {
            path,
            frame,
            corners,
        });
    }

    let mut results = Vec::new();
    for (scene_index, scene) in scenes.iter().enumerate() {
        // Warm both operations, then alternate their order to balance timing effects.
        for iteration in 0..WARMUPS {
            if (scene_index + iteration) % 2 == 0 {
                extract_ms(&extractor, scene)?;
                rectify_ms(&extractor, scene)?;
            } else {
                rectify_ms(&extractor, scene)?;
                extract_ms(&extractor, scene)?;
            }
        }
        let mut extraction = Vec::with_capacity(ITERATIONS);
        let mut rectification = Vec::with_capacity(ITERATIONS);
        for iteration in 0..ITERATIONS {
            if (scene_index + iteration) % 2 == 0 {
                extraction.push(extract_ms(&extractor, scene)?);
                rectification.push(rectify_ms(&extractor, scene)?);
            } else {
                rectification.push(rectify_ms(&extractor, scene)?);
                extraction.push(extract_ms(&extractor, scene)?);
            }
        }
        results.push((extraction, rectification));
    }

    println!("[");
    for (index, (scene, (extraction, rectification))) in scenes.iter().zip(&results).enumerate() {
        // Input paths are printed to stderr, avoiding custom JSON string escaping.
        eprintln!("verified exact crop pixels: {}", scene.path);
        let corners = scene
            .corners
            .iter()
            .map(|point| format!("[{},{}]", point.x, point.y))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{{\"size\":[{},{}],\"corners\":[{}],\
             \"exact_crop_pixels\":true,\"extract_ms\":[{}],\"rectify_ms\":[{}]}}{}",
            scene.frame.width(),
            scene.frame.height(),
            corners,
            samples_json(extraction),
            samples_json(rectification),
            if index + 1 < scenes.len() { "," } else { "" }
        );
    }
    println!("]");
    Ok(())
}
