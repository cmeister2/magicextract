//! Render evidence with the production public API, independently of the timed harness.
//!
//! Compiled by `render-detection-examples.py` against the locked release library.

use std::{env, error::Error, fs, path::PathBuf};

use magicextract::{CardExtractor, DetectionMode, DetectionRegion, ExtractorConfig, image};

/// Extract real rectified crops and print their original-image corners as JSON.
fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    if arguments.len() != 5 {
        return Err("expected: input output-directory full|fast dimension roi|frame".into());
    }
    let frame = image::open(&arguments[0])?;
    let directory = PathBuf::from(&arguments[1]);
    let dimension = arguments[3].parse()?;
    let region = if arguments[4] == "roi" {
        let x = (frame.width() as f32 * 0.1).round() as u32;
        let y = (frame.height() as f32 * 0.1).round() as u32;
        let right = (frame.width() as f32 * 0.9).round() as u32;
        let bottom = (frame.height() as f32 * 0.9).round() as u32;
        Some(DetectionRegion {
            x,
            y,
            width: right - x,
            height: bottom - y,
        })
    } else {
        None
    };
    let extractor = CardExtractor::new(ExtractorConfig {
        max_detection_dimension: dimension,
        detection_mode: match arguments[2].as_str() {
            "full" => DetectionMode::Full,
            "fast" => DetectionMode::Fast,
            _ => return Err("expected full or fast".into()),
        },
        detection_region: region,
        ..ExtractorConfig::default()
    })?;
    fs::create_dir_all(&directory)?;
    let cards = extractor.extract(&frame)?;
    let mut serialized = Vec::new();
    for (index, card) in cards.iter().enumerate() {
        card.image
            .save(directory.join(format!("card-{index:03}.png")))?;
        let corners: Vec<_> = card
            .detection
            .corners
            .iter()
            .map(|point| format!("[{},{}]", point.x, point.y))
            .collect();
        serialized.push(format!("{{\"corners\":[{}]}}", corners.join(",")));
    }
    println!("[{}]", serialized.join(","));
    Ok(())
}
