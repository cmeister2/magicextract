//! Map detected quadrilaterals to fixed-size portrait crops.

use image::RgbImage;
use imageproc::geometric_transformations::{
    Border, Interpolation, Projection, warp_into, warp_into_with,
};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::Point;

/// Sample the original image using a projective map from the detected corners.
///
/// Returns `None` when the four points cannot define an invertible projection.
/// The caller guarantees nonempty images and output dimensions of at least two.
pub(crate) fn rectify(
    source: &RgbImage,
    corners: &[Point; 4],
    width: u32,
    height: u32,
) -> Option<RgbImage> {
    let from = corners.map(|point| (point.x, point.y));
    let right = (width - 1) as f32;
    let bottom = (height - 1) as f32;
    let to = [(0.0, 0.0), (right, 0.0), (right, bottom), (0.0, bottom)];
    let projection = Projection::from_control_points(from, to)?;
    let mut crop = RgbImage::new(width, height);
    warp_into(
        source,
        projection,
        Interpolation::Bilinear,
        Border::Replicate,
        &mut crop,
    );
    Some(crop)
}

/// Rectify externally supplied corners with checked floating-point projection.
///
/// The caller checks clockwise convex geometry, bounds, and image dimensions.
/// Returns `None` if the f32 projection loses control-point accuracy or produces
/// invalid sample coordinates. This catches almost-collapsed quads that are
/// mathematically convex but cannot be reliably represented by the warp.
pub(crate) fn rectify_checked(
    source: &RgbImage,
    corners: &[Point; 4],
    width: u32,
    height: u32,
) -> Option<RgbImage> {
    let from = corners.map(|point| (point.x, point.y));
    let right = (width - 1) as f32;
    let bottom = (height - 1) as f32;
    let to = [(0.0, 0.0), (right, 0.0), (right, bottom), (0.0, bottom)];
    let projection = Projection::from_control_points(from, to)?;
    let inverse = projection.invert();
    for (source_point, destination_point) in from.into_iter().zip(to) {
        if !accurate_control_point(projection * source_point, destination_point)
            || !accurate_control_point(inverse * destination_point, source_point)
        {
            return None;
        }
    }

    // Even a finite control-point fit must not expose the dependency's sampler
    // to nonfinite or overflowing coordinates elsewhere in the output. A valid
    // convex map stays within the source bounds, allowing half a pixel for
    // floating-point drift at the boundary. Keep the original coordinates on
    // the valid path so interpolation remains identical to ordinary extraction.
    let invalid_sample = AtomicBool::new(false);
    let source_right = (source.width() - 1) as f32;
    let source_bottom = (source.height() - 1) as f32;
    let mut crop = RgbImage::new(width, height);
    warp_into_with(
        source,
        |x, y| {
            let (source_x, source_y) = inverse * (x, y);
            if source_x.is_finite()
                && source_y.is_finite()
                && (-0.5..=source_right + 0.5).contains(&source_x)
                && (-0.5..=source_bottom + 0.5).contains(&source_y)
            {
                (source_x, source_y)
            } else {
                // This placeholder is never returned to the caller: any
                // invalid sample rejects the whole crop after the warp.
                invalid_sample.store(true, Ordering::Relaxed);
                (0.0, 0.0)
            }
        },
        Interpolation::Bilinear,
        Border::Replicate,
        &mut crop,
    );
    (!invalid_sample.load(Ordering::Relaxed)).then_some(crop)
}

/// Require finite control-point mappings with at most half a pixel of error.
fn accurate_control_point(actual: (f32, f32), expected: (f32, f32)) -> bool {
    actual.0.is_finite()
        && actual.1.is_finite()
        && (actual.0 - expected.0).abs() <= 0.5
        && (actual.1 - expected.1).abs() <= 0.5
}
