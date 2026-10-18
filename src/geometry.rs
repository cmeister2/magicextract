//! Map detected quadrilaterals to fixed-size portrait crops.

use image::RgbImage;
use imageproc::geometric_transformations::{Border, Interpolation, Projection, warp_into};

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
