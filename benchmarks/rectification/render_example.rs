//! Render actual public-API crops from supplied source-pixel corners.
//!
//! Build and invoke through `render-example.py`; no detection or timing runs.

use std::{env, error::Error, fs, path::PathBuf};

use magicextract::{
    CardExtractor, Point,
    image::{self, DynamicImage},
};

/// Save the decoded source and both caller-selected crop orientations.
fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = env::args().skip(1).collect();
    if arguments.len() != 10 {
        return Err("expected input, output directory, and eight corner coordinates".into());
    }
    let frame = DynamicImage::ImageRgb8(image::open(&arguments[0])?.into_rgb8());
    let output = PathBuf::from(&arguments[1]);
    fs::create_dir_all(&output)?;
    let coordinates: Vec<f32> = arguments[2..]
        .iter()
        .map(|value| value.parse())
        .collect::<Result<_, _>>()?;
    let corners = std::array::from_fn(|index| Point {
        x: coordinates[index * 2],
        y: coordinates[index * 2 + 1],
    });
    let extractor = CardExtractor::default();
    let crop = extractor.rectify(&frame, &corners)?;
    let turned = extractor.rectify(&frame, &[corners[2], corners[3], corners[0], corners[1]])?;
    frame.save(output.join("source.png"))?;
    crop.save(output.join("crop.png"))?;
    turned.save(output.join("crop-rotated.png"))?;

    // Compare the two actual API calls, without replacing either rendered crop.
    let rotated_reference = image::imageops::rotate180(&crop);
    let differences: Vec<_> = turned
        .as_raw()
        .iter()
        .zip(rotated_reference.as_raw())
        .map(|(&actual, &expected)| actual.abs_diff(expected))
        .collect();
    let maximum = differences.iter().copied().max().unwrap_or(0);
    let total: u64 = differences.iter().map(|&value| u64::from(value)).sum();
    println!(
        "{{\"source_dimensions\":[{},{}],\"crop_dimensions\":[{},{}],\
         \"half_turn_max_channel_difference\":{},\
         \"half_turn_mean_absolute_channel_difference\":{}}}",
        frame.width(),
        frame.height(),
        crop.width(),
        crop.height(),
        maximum,
        total as f64 / differences.len() as f64
    );
    Ok(())
}
