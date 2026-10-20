//! Detection windows and angle restrictions retain source-image semantics.

use image::{DynamicImage, Rgb, RgbImage};
use imageproc::{
    drawing::{draw_filled_circle_mut, draw_polygon_mut},
    point::Point as PixelPoint,
};
use magicextract::{
    CardDetection, CardExtractor, DetectionMode, DetectionRegion, Error, ExtractorConfig,
    OrientationConstraint, Point,
};

/// Dark tabletop used to isolate the geometric behavior under test.
const TABLE: Rgb<u8> = Rgb([24, 31, 37]);

/// A high-contrast card face with no inner contour.
const CARD: Rgb<u8> = Rgb([238, 235, 225]);

/// The fast path locates a clear card and retains source pixels through a window.
#[test]
fn fast_mode_preserves_geometry_and_pixels_inside_region() {
    let mut source = RgbImage::from_pixel(800, 600, TABLE);
    let expected = rectangle(360.0, 140.0, 180.0, 252.0);
    paint_card(&mut source, expected);
    paint_card(&mut source, rectangle(20.0, 20.0, 180.0, 252.0));
    let extractor = CardExtractor::new(ExtractorConfig {
        detection_mode: DetectionMode::Fast,
        max_detection_dimension: 320,
        detection_region: Some(DetectionRegion {
            x: 300,
            y: 80,
            width: 320,
            height: 400,
        }),
        orientation: Some(OrientationConstraint {
            angle_degrees: 0.0,
            tolerance_degrees: 10.0,
        }),
        ..ExtractorConfig::default()
    })
    .unwrap();
    let cards = extractor.extract(&DynamicImage::ImageRgb8(source)).unwrap();
    assert_eq!(cards.len(), 1);
    assert_portrait_corners(&cards[0].detection, expected, 4.0);
    assert_eq!(*cards[0].image.get_pixel(125, 175), CARD);
    let empty = DynamicImage::ImageRgb8(RgbImage::from_pixel(800, 600, TABLE));
    assert!(extractor.detect(&empty).unwrap().is_empty());
}

/// Paints a source-space quadrilateral without altering its expected corners.
fn paint_card(image: &mut RgbImage, corners: [Point; 4]) {
    let polygon: Vec<_> = corners
        .map(|point| PixelPoint::new(point.x.round() as i32, point.y.round() as i32))
        .into_iter()
        .collect();
    draw_polygon_mut(image, &polygon, CARD);
}

/// Constructs corners for a portrait rectangle in original-image pixels.
fn rectangle(x: f32, y: f32, width: f32, height: f32) -> [Point; 4] {
    [
        Point { x, y },
        Point { x: x + width, y },
        Point {
            x: x + width,
            y: y + height,
        },
        Point { x, y: y + height },
    ]
}

/// Verifies the physical portrait order, allowing the equivalent 180-degree start.
fn assert_portrait_corners(detection: &CardDetection, expected: [Point; 4], tolerance: f32) {
    let error = [0, 2]
        .into_iter()
        .map(|offset| {
            (0..4)
                .map(|index| {
                    let actual = detection.corners[index];
                    let target = expected[(index + offset) % 4];
                    (actual.x - target.x).hypot(actual.y - target.y)
                })
                .fold(0.0, f32::max)
        })
        .fold(f32::INFINITY, f32::min);
    assert!(
        error <= tolerance,
        "source corner error {error} exceeds {tolerance}: {detection:?}"
    );
}

/// Zero-area regions are invalid regardless of the future input dimensions.
#[test]
fn rejects_zero_sized_regions_at_construction() {
    for (width, height) in [(0, 1), (1, 0), (0, 0)] {
        let result = CardExtractor::new(ExtractorConfig {
            detection_region: Some(DetectionRegion {
                x: 0,
                y: 0,
                width,
                height,
            }),
            ..ExtractorConfig::default()
        });
        assert!(matches!(result, Err(Error::InvalidConfig(_))));
    }
}

/// Bounds are validated per frame without overflowing unsigned coordinates.
#[test]
fn rejects_regions_outside_the_input() {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(480, 420, TABLE));
    for region in [
        DetectionRegion {
            x: 480,
            y: 0,
            width: 1,
            height: 1,
        },
        DetectionRegion {
            x: 0,
            y: 420,
            width: 1,
            height: 1,
        },
        DetectionRegion {
            x: 470,
            y: 410,
            width: 11,
            height: 10,
        },
        DetectionRegion {
            x: 470,
            y: 410,
            width: 10,
            height: 11,
        },
        DetectionRegion {
            x: u32::MAX,
            y: 0,
            width: 2,
            height: 1,
        },
        DetectionRegion {
            x: 1,
            y: 1,
            width: u32::MAX,
            height: u32::MAX,
        },
    ] {
        let extractor = CardExtractor::new(ExtractorConfig {
            detection_region: Some(region),
            ..ExtractorConfig::default()
        })
        .unwrap();
        assert_eq!(extractor.detect(&image), Err(Error::InvalidRegion));
        assert!(matches!(
            extractor.extract(&image),
            Err(Error::InvalidRegion)
        ));
    }
}

/// A window ending exactly at the image boundary is valid.
#[test]
fn accepts_regions_touching_the_input_boundary() {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(480, 420, TABLE));
    for region in [
        DetectionRegion {
            x: 0,
            y: 0,
            width: 480,
            height: 420,
        },
        DetectionRegion {
            x: 470,
            y: 410,
            width: 10,
            height: 10,
        },
    ] {
        let extractor = CardExtractor::new(ExtractorConfig {
            detection_region: Some(region),
            ..ExtractorConfig::default()
        })
        .unwrap();
        assert!(extractor.detect(&image).unwrap().is_empty());
    }
}

/// Cropping and downscaling preserve source corners and sample full-resolution RGB.
#[test]
fn region_limits_detection_and_preserves_source_crops() {
    let mut source = RgbImage::from_pixel(1_280, 960, TABLE);
    let expected = rectangle(720.0, 220.0, 250.0, 350.0);
    paint_card(&mut source, expected);
    paint_card(&mut source, rectangle(100.0, 100.0, 180.0, 252.0));
    let landmark = Rgb([210, 30, 40]);
    draw_filled_circle_mut(&mut source, (845, 395), 30, landmark);
    let extractor = CardExtractor::new(ExtractorConfig {
        detection_region: Some(DetectionRegion {
            x: 660,
            y: 160,
            width: 400,
            height: 500,
        }),
        max_detection_dimension: 200,
        ..ExtractorConfig::default()
    })
    .unwrap();
    let cards = extractor.extract(&DynamicImage::ImageRgb8(source)).unwrap();
    assert_eq!(cards.len(), 1);
    assert_portrait_corners(&cards[0].detection, expected, 8.0);
    assert!((cards[0].detection.area - 250.0 * 350.0).abs() < 4_000.0);
    assert_eq!(cards[0].image.dimensions(), (250, 350));
    assert_eq!(*cards[0].image.get_pixel(125, 175), landmark);
}

/// Area limits refer to the window rather than the surrounding camera frame.
#[test]
fn area_fractions_are_relative_to_the_region() {
    let mut source = RgbImage::from_pixel(1_000, 800, TABLE);
    let expected = rectangle(400.0, 200.0, 200.0, 280.0);
    paint_card(&mut source, expected);
    let source = DynamicImage::ImageRgb8(source);
    let config = ExtractorConfig {
        min_area_fraction: 0.4,
        max_area_fraction: 0.7,
        ..ExtractorConfig::default()
    };
    assert!(
        CardExtractor::new(config.clone())
            .unwrap()
            .detect(&source)
            .unwrap()
            .is_empty()
    );
    let extractor = CardExtractor::new(ExtractorConfig {
        detection_region: Some(DetectionRegion {
            x: 360,
            y: 160,
            width: 280,
            height: 360,
        }),
        ..config
    })
    .unwrap();
    let detections = extractor.detect(&source).unwrap();
    assert_eq!(detections.len(), 1);
    assert_portrait_corners(&detections[0], expected, 4.0);
}

/// An asymmetric crop must retain the original pinhole camera's principal point.
#[test]
fn off_axis_region_preserves_oblique_physical_orientation() {
    for horizontal_axis in [false, true] {
        let (sin, cos) = 62.0_f32.to_radians().sin_cos();
        let expected = [
            (-90.0, -126.0),
            (90.0, -126.0),
            (90.0, 126.0),
            (-90.0, 126.0),
        ]
        .map(|(x, y)| {
            let (rotated_x, rotated_y, depth) = if horizontal_axis {
                (x, y * cos, 1_200.0 + y * sin)
            } else {
                (x * cos, y, 1_200.0 - x * sin)
            };
            Point {
                x: 639.5 + 1_100.0 * (rotated_x + 200.0) / depth,
                y: 479.5 + 1_100.0 * (rotated_y - 160.0) / depth,
            }
        });
        let mut source = RgbImage::from_pixel(1_280, 960, TABLE);
        paint_card(&mut source, expected);
        let extractor = CardExtractor::new(ExtractorConfig {
            detection_region: Some(DetectionRegion {
                x: 600,
                y: 100,
                width: 400,
                height: 420,
            }),
            ..ExtractorConfig::default()
        })
        .unwrap();
        let detections = extractor.detect(&DynamicImage::ImageRgb8(source)).unwrap();
        assert_eq!(detections.len(), 1, "horizontal_axis={horizontal_axis}");
        assert_portrait_corners(&detections[0], expected, 5.0);
    }
}

/// Orientation restrictions reject nonfinite values and invalid angle widths.
#[test]
fn validates_orientation_constraint() {
    for (angle_degrees, tolerance_degrees) in [
        (f32::NAN, 10.0),
        (f32::INFINITY, 10.0),
        (0.0, f32::NAN),
        (0.0, f32::INFINITY),
        (0.0, -0.01),
        (0.0, 90.01),
    ] {
        assert!(matches!(
            CardExtractor::new(ExtractorConfig {
                orientation: Some(OrientationConstraint {
                    angle_degrees,
                    tolerance_degrees,
                }),
                ..ExtractorConfig::default()
            }),
            Err(Error::InvalidConfig(_))
        ));
    }
    for (angle_degrees, tolerance_degrees) in [(0.0, 0.0), (-180.0, 90.0), (360.0, 10.0)] {
        assert!(
            CardExtractor::new(ExtractorConfig {
                orientation: Some(OrientationConstraint {
                    angle_degrees,
                    tolerance_degrees,
                }),
                ..ExtractorConfig::default()
            })
            .is_ok()
        );
    }
}

/// Rotates a physical portrait card clockwise around a source-space center.
fn rotated_card(center: Point, width: f32, height: f32, angle_degrees: f32) -> [Point; 4] {
    let (sin, cos) = angle_degrees.to_radians().sin_cos();
    [
        (-width * 0.5, -height * 0.5),
        (width * 0.5, -height * 0.5),
        (width * 0.5, height * 0.5),
        (-width * 0.5, height * 0.5),
    ]
    .map(|(x, y)| Point {
        x: center.x + x * cos - y * sin,
        y: center.y + x * sin + y * cos,
    })
}

/// Short-edge angle restrictions wrap at 180 degrees and apply before card limits.
#[test]
fn orientation_filters_upright_landscape_and_diagonal_inside_region() {
    let mut source = RgbImage::from_pixel(900, 720, TABLE);
    let portrait = rectangle(80.0, 80.0, 100.0, 140.0);
    let landscape = rotated_card(Point { x: 450.0, y: 180.0 }, 180.0, 252.0, 90.0);
    let diagonal = rotated_card(Point { x: 700.0, y: 350.0 }, 130.0, 182.0, 45.0);
    for corners in [portrait, landscape, diagonal] {
        paint_card(&mut source, corners);
    }
    let source = DynamicImage::ImageRgb8(source);
    assert_eq!(CardExtractor::default().detect(&source).unwrap().len(), 3);
    for (angle_degrees, expected) in [
        (0.0, portrait),
        (180.0, portrait),
        (-180.0, portrait),
        (90.0, landscape),
        (270.0, landscape),
        (45.0, diagonal),
        (225.0, diagonal),
    ] {
        let extractor = CardExtractor::new(ExtractorConfig {
            detection_region: Some(DetectionRegion {
                x: 40,
                y: 40,
                width: 820,
                height: 600,
            }),
            max_cards: 1,
            max_detection_dimension: 450,
            orientation: Some(OrientationConstraint {
                angle_degrees,
                tolerance_degrees: 8.0,
            }),
            ..ExtractorConfig::default()
        })
        .unwrap();
        let detections = extractor.detect(&source).unwrap();
        assert_eq!(detections.len(), 1, "angle={angle_degrees}");
        assert_portrait_corners(&detections[0], expected, 7.0);
    }
}
