//! Extract trading card sized outlines from photographs or decoded video frames.
//!
//! The detector uses visible outlines and a trading card aspect ratio of 5:7,
//! without identifying the card's game, artwork, rank, or suit. Each extraction
//! includes a portrait crop and corners in the original image's coordinates.
//!
//! ```no_run
//! use magicextract::{CardExtractor, image};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let frame = image::open("table.jpg")?;
//! for (index, card) in CardExtractor::default().extract(&frame)?.iter().enumerate() {
//!     card.image.save(format!("card-{index}.png"))?;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Image decoding and encoding are provided by your application's `image`
//! features. This crate itself operates on decoded images and enables no codecs.
//! Video decoding, tracking, and card recognition belong to the caller.

use std::fmt;

/// Contour detection and candidate filtering.
mod detection;
/// Projective correction of detected outlines.
mod geometry;

/// Image types shared with callers; enable codecs on your own `image` dependency.
pub use image;

/// Default short-side to long-side ratio used for trading card outlines.
pub const TRADING_CARD_ASPECT_RATIO: f32 = 5.0 / 7.0;

/// A position in the original image, measured in pixels from the top-left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// Horizontal coordinate, increasing to the right.
    pub x: f32,
    /// Vertical coordinate, increasing downward.
    pub y: f32,
}

/// A card-shaped outline, without any claim about its contents or physical size.
#[derive(Clone, Debug, PartialEq)]
pub struct CardDetection {
    /// Clockwise corners in original-image pixels.
    ///
    /// The first edge is the short edge, so rectification produces a portrait
    /// crop. Of the two possible starts, the uppermost (then leftmost) is used.
    /// This geometric orientation does not establish which way the artwork faces.
    pub corners: [Point; 4],
    /// Area enclosed by the corners, in original-image square pixels.
    pub area: f32,
}

/// A perspective-corrected RGB crop and its original-image outline.
#[derive(Clone, Debug)]
pub struct ExtractedCard {
    /// Portrait crop at the configured output resolution.
    pub image: image::RgbImage,
    /// Outline of the region used to produce the crop.
    pub detection: CardDetection,
}

/// Detection and output settings, validated by [`CardExtractor::new`].
///
/// Start with [`Self::default`] and change the settings relevant to your input.
#[derive(Clone, Debug, PartialEq)]
pub struct ExtractorConfig {
    /// Crop width in pixels, between 2 and 8192 inclusive. Default: 250.
    pub output_width: u32,
    /// Crop height in pixels, between 2 and 8192 inclusive. Default: 350.
    pub output_height: u32,
    /// Relative tolerance around the 5:7 short/long edge ratio.
    ///
    /// Must be finite and strictly between zero and one. Default: 0.20.
    /// Nearly frontal views use the apparent ratio. Oblique views also test a
    /// centered, square-pixel pinhole camera model; its physical-ratio tolerance
    /// is capped at 0.08 to limit implausible projective matches.
    pub aspect_ratio_tolerance: f32,
    /// Smallest candidate area as a fraction of the image. Default: 0.005.
    ///
    /// Must be finite, positive, and smaller than [`Self::max_area_fraction`].
    pub min_area_fraction: f32,
    /// Largest candidate area as a fraction of the image. Default: 0.95.
    ///
    /// Must be finite, at most one, and larger than [`Self::min_area_fraction`].
    pub max_area_fraction: f32,
    /// Longest side of the detection image, between 32 and 4096. Default: 1600.
    ///
    /// Larger images are reduced for detection; crops always sample the original.
    pub max_detection_dimension: u32,
    /// Positive, finite lower Canny gradient threshold. Default: 15.
    pub canny_low_threshold: f32,
    /// Finite upper Canny threshold, greater than the lower threshold. Default: 45.
    pub canny_high_threshold: f32,
    /// Maximum number of returned cards; must be nonzero. Default: 64.
    pub max_cards: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            output_width: 250,
            output_height: 350,
            aspect_ratio_tolerance: 0.20,
            min_area_fraction: 0.005,
            max_area_fraction: 0.95,
            max_detection_dimension: 1600,
            canny_low_threshold: 15.0,
            canny_high_threshold: 45.0,
            max_cards: 64,
        }
    }
}

impl ExtractorConfig {
    /// Check bounds before allowing this configuration into a detector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for out-of-range or nonfinite settings.
    fn validate(&self) -> Result<(), Error> {
        if !(2..=8192).contains(&self.output_width) || !(2..=8192).contains(&self.output_height) {
            return Err(Error::InvalidConfig(
                "output dimensions must be in 2..=8192",
            ));
        }
        if !self.aspect_ratio_tolerance.is_finite()
            || self.aspect_ratio_tolerance <= 0.0
            || self.aspect_ratio_tolerance >= 1.0
        {
            return Err(Error::InvalidConfig(
                "aspect ratio tolerance must be finite and in 0..1 (exclusive)",
            ));
        }
        if !self.min_area_fraction.is_finite()
            || !self.max_area_fraction.is_finite()
            || self.min_area_fraction <= 0.0
            || self.min_area_fraction >= self.max_area_fraction
            || self.max_area_fraction > 1.0
        {
            return Err(Error::InvalidConfig(
                "area fractions must satisfy 0 < min < max <= 1",
            ));
        }
        if !(32..=4096).contains(&self.max_detection_dimension) {
            return Err(Error::InvalidConfig(
                "maximum detection dimension must be in 32..=4096",
            ));
        }
        if !self.canny_low_threshold.is_finite()
            || !self.canny_high_threshold.is_finite()
            || self.canny_low_threshold <= 0.0
            || self.canny_high_threshold <= self.canny_low_threshold
        {
            return Err(Error::InvalidConfig(
                "Canny thresholds must be finite and satisfy 0 < low < high",
            ));
        }
        if self.max_cards == 0 {
            return Err(Error::InvalidConfig("maximum card count must be nonzero"));
        }
        Ok(())
    }
}

/// Invalid configuration or unsupported input to extraction.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A configuration value failed validation, with an explanation.
    InvalidConfig(&'static str),
    /// The input image has zero width or height.
    EmptyImage,
    /// The input has a dimension beyond the detector's signed coordinate range.
    ImageTooLarge,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => {
                write!(formatter, "invalid extractor configuration: {reason}")
            }
            Self::EmptyImage => formatter.write_str("the input image must not be empty"),
            Self::ImageTooLarge => {
                formatter.write_str("input dimensions must fit in a signed 32-bit coordinate")
            }
        }
    }
}

impl std::error::Error for Error {}

/// A reusable, immutable card extractor with validated settings.
///
/// Reuse an instance for frames from the same video. Calls are independent:
/// detections are neither tracked nor deduplicated between frames.
#[derive(Clone, Debug, Default)]
pub struct CardExtractor {
    /// Settings validated at construction and immutable thereafter.
    config: ExtractorConfig,
}

impl CardExtractor {
    /// Construct an extractor with custom settings.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for invalid ranges, nonfinite floating
    /// point values, or a zero card limit. See [`ExtractorConfig`] for bounds.
    pub fn new(config: ExtractorConfig) -> Result<Self, Error> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Read the settings used by this extractor.
    pub fn config(&self) -> &ExtractorConfig {
        &self.config
    }

    /// Find card-shaped outlines without allocating output crops.
    ///
    /// Results are sorted top-to-bottom, then left-to-right by their bounding box.
    /// When capped, the largest distinct candidates are retained first.
    /// An image with no accepted outlines returns an empty vector.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyImage`] for a zero dimension or
    /// [`Error::ImageTooLarge`] for a dimension larger than [`i32::MAX`].
    pub fn detect(&self, image: &image::DynamicImage) -> Result<Vec<CardDetection>, Error> {
        validate_image(image)?;
        Ok(detection::detect(image, &self.config))
    }

    /// Find outlines and rectify each into an RGB crop sampled from the input.
    ///
    /// Projectively degenerate candidates are skipped. RGB conversion discards
    /// alpha; callers should composite transparent input before extraction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EmptyImage`] for a zero dimension or
    /// [`Error::ImageTooLarge`] for a dimension larger than [`i32::MAX`].
    pub fn extract(&self, image: &image::DynamicImage) -> Result<Vec<ExtractedCard>, Error> {
        let detections = self.detect(image)?;
        if detections.is_empty() {
            return Ok(Vec::new());
        }
        let source = image.to_rgb8();
        Ok(detections
            .into_iter()
            .filter_map(|detection| {
                let image = geometry::rectify(
                    &source,
                    &detection.corners,
                    self.config.output_width,
                    self.config.output_height,
                )?;
                Some(ExtractedCard { image, detection })
            })
            .collect())
    }
}

/// Extract card crops using the default settings.
///
/// Use [`CardExtractor`] to customize detection or reuse settings between frames.
///
/// # Errors
///
/// Returns [`Error::EmptyImage`] for a zero dimension or [`Error::ImageTooLarge`]
/// for a dimension larger than [`i32::MAX`].
pub fn extract_cards(image: &image::DynamicImage) -> Result<Vec<ExtractedCard>, Error> {
    CardExtractor::default().extract(image)
}

/// Reject image sizes incompatible with the contour coordinate representation.
///
/// # Errors
///
/// Returns [`Error::EmptyImage`] for empty input and [`Error::ImageTooLarge`]
/// when an input dimension exceeds [`i32::MAX`].
fn validate_image(image: &image::DynamicImage) -> Result<(), Error> {
    if image.width() == 0 || image.height() == 0 {
        return Err(Error::EmptyImage);
    }
    if image.width() > i32::MAX as u32 || image.height() > i32::MAX as u32 {
        return Err(Error::ImageTooLarge);
    }
    Ok(())
}
