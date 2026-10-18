//! Save card crops from a photograph or an already decoded video frame.

use std::{env, error::Error, fs, path::PathBuf};

use magicextract::{CardExtractor, image};

/// Read one image and write numbered PNG crops into an output directory.
///
/// # Errors
///
/// Returns an error for missing arguments, unsupported image data, extraction
/// failure, or an output directory/file that cannot be created.
fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args_os().skip(1);
    let input = arguments
        .next()
        .ok_or("usage: cargo run --example extract -- <image> <output-directory>")?;
    let output = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("usage: cargo run --example extract -- <image> <output-directory>")?;
    if arguments.next().is_some() {
        return Err("expected exactly an input image and an output directory".into());
    }
    let frame = image::open(input)?;
    let cards = CardExtractor::default().extract(&frame)?;
    fs::create_dir_all(&output)?;
    for (index, card) in cards.iter().enumerate() {
        card.image
            .save(output.join(format!("card-{index:03}.png")))?;
        println!("card-{index:03}: {:?}", card.detection.corners);
    }
    println!("Extracted {} card(s).", cards.len());
    Ok(())
}
