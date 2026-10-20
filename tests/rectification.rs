//! Caller-supplied outlines retain their order and sample original image pixels.

use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage, imageops};
use imageproc::{
    drawing::{draw_filled_circle_mut, draw_polygon_mut},
    point::Point as PixelPoint,
};
use magicextract::{
    CardExtractor, DetectionMode, DetectionRegion, Error, ExtractorConfig, OrientationConstraint,
    Point,
};

/// A dark surface makes the synthetic card's outer boundary unambiguous.
const TABLE: Rgb<u8> = Rgb([24, 31, 37]);

/// The card's neutral face separates colored landmarks from its outline.
const FACE: Rgb<u8> = Rgb([238, 235, 225]);

/// Constructs a clockwise rectangle using inclusive source pixel coordinates.
fn rectangle(left: f32, top: f32, right: f32, bottom: f32) -> [Point; 4] {
    [
        Point { x: left, y: top },
        Point { x: right, y: top },
        Point {
            x: right,
            y: bottom,
        },
        Point { x: left, y: bottom },
    ]
}

/// Paints one quadrilateral without modifying the supplied source coordinates.
fn paint_face(image: &mut RgbImage, corners: [Point; 4]) {
    let polygon: Vec<_> = corners
        .map(|point| PixelPoint::new(point.x as i32, point.y as i32))
        .into_iter()
        .collect();
    draw_polygon_mut(image, &polygon, FACE);
}

/// Builds an extractor with an explicit output size for small synthetic tests.
fn sized_extractor(width: u32, height: u32) -> CardExtractor {
    CardExtractor::new(ExtractorConfig {
        output_width: width,
        output_height: height,
        ..ExtractorConfig::default()
    })
    .unwrap()
}

/// Measures horizontal contrast in the central third of the rectified face.
fn central_detail(image: &RgbImage) -> f32 {
    let mut difference = 0_u64;
    let mut count = 0_u64;
    for y in image.height() / 3..image.height() * 2 / 3 {
        for x in image.width() / 3..image.width() * 2 / 3 - 1 {
            difference +=
                u64::from(image.get_pixel(x, y).0[0].abs_diff(image.get_pixel(x + 1, y).0[0]));
            count += 1;
        }
    }
    difference as f32 / count as f32
}

/// Rectification preserves perspective landmarks and matches extraction pixels.
#[test]
fn rectifies_perspective_landmarks_and_matches_extraction() {
    let corners = [
        Point { x: 90.0, y: 60.0 },
        Point { x: 270.0, y: 80.0 },
        Point { x: 280.0, y: 340.0 },
        Point { x: 70.0, y: 320.0 },
    ];
    let mut source = RgbImage::from_pixel(420, 420, TABLE);
    paint_face(&mut source, corners);
    let landmarks = [
        ((130, 110), Rgb([220, 30, 30]), (false, false)),
        ((230, 120), Rgb([30, 180, 30]), (true, false)),
        ((240, 290), Rgb([220, 180, 30]), (true, true)),
        ((120, 280), Rgb([30, 30, 220]), (false, true)),
    ];
    for (center, color, _) in landmarks {
        draw_filled_circle_mut(&mut source, center, 13, color);
    }
    let source = DynamicImage::ImageRgb8(source);
    let extractor = sized_extractor(150, 210);
    let output = extractor.rectify(&source, &corners).unwrap();
    assert_eq!(output.dimensions(), (150, 210));
    for (_, color, expected_quadrant) in landmarks {
        let matching: Vec<_> = output
            .enumerate_pixels()
            .filter(|(_, _, pixel)| **pixel == color)
            .map(|(x, y, _)| (x, y))
            .collect();
        assert!(matching.len() > 50, "lost landmark {color:?}");
        assert!(matching.iter().all(|&(x, y)| {
            (x >= output.width() / 2, y >= output.height() / 2) == expected_quadrant
        }));
    }

    let cards = extractor.extract(&source).unwrap();
    assert_eq!(cards.len(), 1);
    let supplied = extractor
        .rectify(&source, &cards[0].detection.corners)
        .unwrap();
    assert_eq!(supplied, cards[0].image);
}

/// Caller order selects output orientation instead of canonicalizing the outline.
#[test]
fn preserves_corner_order_and_half_turn_orientation() {
    let source = DynamicImage::ImageRgb8(RgbImage::from_fn(128, 160, |x, y| {
        Rgb([x as u8, y as u8, ((x + y) / 2) as u8])
    }));
    let corners = rectangle(30.0, 40.0, 90.0, 124.0);
    let extractor = sized_extractor(61, 85);
    let upright = extractor.rectify(&source, &corners).unwrap();
    let rotated = extractor
        .rectify(&source, &[corners[2], corners[3], corners[0], corners[1]])
        .unwrap();
    let expected = imageops::rotate180(&upright);
    assert!(
        rotated
            .as_raw()
            .iter()
            .zip(expected.as_raw())
            .all(|(&actual, &expected)| actual.abs_diff(expected) <= 1),
        "changing the starting corner by two must rotate the crop 180 degrees"
    );
    for (x, y, expected) in [
        (0, 0, Rgb([30, 40, 35])),
        (60, 0, Rgb([90, 40, 65])),
        (60, 84, Rgb([90, 124, 107])),
        (0, 84, Rgb([30, 124, 77])),
    ] {
        assert!(
            upright
                .get_pixel(x, y)
                .0
                .iter()
                .zip(expected.0)
                .all(|(&actual, expected)| actual.abs_diff(expected) <= 1)
        );
    }
}

/// Explicit outlines work without detectable edges or a matching detection domain.
#[test]
fn bypasses_detection_region_orientation_aspect_and_area_filters() {
    let source = DynamicImage::ImageRgb8(RgbImage::from_fn(160, 120, |x, y| {
        Rgb([80 + (x / 20) as u8, 90 + (y / 20) as u8, 100])
    }));
    assert!(CardExtractor::default().detect(&source).unwrap().is_empty());
    let reference = sized_extractor(80, 40);
    let small = rectangle(30.0, 30.0, 110.0, 60.0);
    let full = rectangle(0.0, 0.0, 159.0, 119.0);
    for (min_area_fraction, max_area_fraction, corners) in [(0.9, 1.0, small), (0.001, 0.01, full)]
    {
        let extractor = CardExtractor::new(ExtractorConfig {
            output_width: 80,
            output_height: 40,
            min_area_fraction,
            max_area_fraction,
            detection_region: Some(DetectionRegion {
                x: 1_000,
                y: 1_000,
                width: 1,
                height: 1,
            }),
            orientation: Some(OrientationConstraint {
                angle_degrees: 90.0,
                tolerance_degrees: 0.0,
            }),
            detection_mode: DetectionMode::Fast,
            max_detection_dimension: 32,
            ..ExtractorConfig::default()
        })
        .unwrap();
        assert_eq!(extractor.detect(&source), Err(Error::InvalidRegion));
        assert_eq!(
            extractor.rectify(&source, &corners).unwrap(),
            reference.rectify(&source, &corners).unwrap()
        );
    }
}

/// Rescaled preview detections sample source detail after a cropped detection pass.
#[test]
fn samples_original_pixels_using_rescaled_preview_region_corners() {
    let mut original = RgbImage::from_pixel(800, 600, TABLE);
    paint_face(&mut original, rectangle(360.0, 140.0, 540.0, 392.0));
    for y in 180..350 {
        for x in 380..520 {
            let value = if (x / 2 + y / 2) % 2 == 0 { 0 } else { 255 };
            original.put_pixel(x, y, Rgb([value; 3]));
        }
    }
    let preview = DynamicImage::ImageRgb8(imageops::resize(
        &original,
        200,
        150,
        imageops::FilterType::Triangle,
    ));
    let original = DynamicImage::ImageRgb8(original);
    let extractor = CardExtractor::new(ExtractorConfig {
        detection_region: Some(DetectionRegion {
            x: 75,
            y: 20,
            width: 80,
            height: 100,
        }),
        max_detection_dimension: 100,
        max_cards: 1,
        ..ExtractorConfig::default()
    })
    .unwrap();
    let detections = extractor.detect(&preview).unwrap();
    assert_eq!(detections.len(), 1);
    let preview_corners = detections[0].corners;
    let source_corners = preview_corners.map(|point| Point {
        x: point.x * 4.0,
        y: point.y * 4.0,
    });
    for (actual, expected) in source_corners
        .iter()
        .zip(rectangle(360.0, 140.0, 540.0, 392.0))
    {
        assert!((actual.x - expected.x).hypot(actual.y - expected.y) < 12.0);
    }
    let high_resolution = extractor.rectify(&original, &source_corners).unwrap();
    let low_resolution = extractor.rectify(&preview, &preview_corners).unwrap();
    assert!(
        central_detail(&high_resolution) > 30.0,
        "the original checkerboard must retain fine texture"
    );
    assert!(
        central_detail(&low_resolution) < 8.0,
        "the preview control must demonstrate loss of fine texture"
    );
}

/// Inclusive image boundaries remain valid, including the smallest valid square.
#[test]
fn accepts_inclusive_source_boundaries() {
    for (width, height) in [(2, 2), (17, 23)] {
        let source = DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, FACE));
        let corners = rectangle(0.0, 0.0, (width - 1) as f32, (height - 1) as f32);
        let output = sized_extractor(width, height)
            .rectify(&source, &corners)
            .unwrap();
        assert_eq!(output, source.to_rgb8());
    }
}

/// Invalid winding and geometry fail explicitly rather than being reordered.
#[test]
fn rejects_malformed_quadrilaterals() {
    let source = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, FACE));
    let valid = rectangle(10.0, 10.0, 80.0, 90.0);
    let mut duplicate = valid;
    duplicate[1] = duplicate[0];
    let mut collinear_edge = valid;
    collinear_edge[1] = Point { x: 45.0, y: 50.0 };
    let mut concave = valid;
    concave[2] = Point { x: 30.0, y: 30.0 };
    for (name, corners) in [
        ("duplicate", duplicate),
        ("collinear edge", collinear_edge),
        ("concave", concave),
        ("crossed", [valid[0], valid[2], valid[1], valid[3]]),
        ("counterclockwise", [valid[0], valid[3], valid[2], valid[1]]),
        (
            "collinear",
            [
                Point { x: 10.0, y: 10.0 },
                Point { x: 20.0, y: 20.0 },
                Point { x: 30.0, y: 30.0 },
                Point { x: 40.0, y: 40.0 },
            ],
        ),
    ] {
        assert_eq!(
            CardExtractor::default().rectify(&source, &corners),
            Err(Error::InvalidQuadrilateral),
            "accepted {name}"
        );
    }
}

/// Nearly collapsed projective edges fail before unstable bilinear sampling.
#[test]
fn rejects_numerically_unstable_projection() {
    let source = DynamicImage::ImageRgb8(RgbImage::from_fn(100, 100, |x, y| {
        Rgb([x as u8, y as u8, 180])
    }));
    let extractor = sized_extractor(4, 4);
    for epsilon in [1e-6_f32, 1e-9, 1e-15] {
        let corners = [
            Point { x: 0.0, y: 0.0 },
            Point { x: epsilon, y: 0.0 },
            Point { x: 99.0, y: 99.0 },
            Point { x: 0.0, y: 99.0 },
        ];
        assert_eq!(
            extractor.rectify(&source, &corners),
            Err(Error::InvalidQuadrilateral),
            "accepted numerically unstable projection with edge length {epsilon}"
        );
    }
}

/// Both coordinate axes reject nonfinite values and pixels outside the source.
#[test]
fn rejects_nonfinite_and_out_of_frame_corners() {
    let source = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, FACE));
    for value in [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.01,
        99.01,
        100.0,
    ] {
        for index in 0..4 {
            for horizontal in [true, false] {
                let mut corners = rectangle(10.0, 10.0, 80.0, 90.0);
                if horizontal {
                    corners[index].x = value;
                } else {
                    corners[index].y = value;
                }
                assert_eq!(
                    CardExtractor::default().rectify(&source, &corners),
                    Err(Error::InvalidQuadrilateral),
                    "accepted value {value} at corner {index}, horizontal={horizontal}"
                );
            }
        }
    }
}

/// Empty frames retain their existing image-validation error before corner work.
#[test]
fn rejects_empty_input() {
    for (width, height) in [(0, 0), (0, 20), (20, 0)] {
        let source = DynamicImage::ImageRgb8(RgbImage::new(width, height));
        assert_eq!(
            CardExtractor::default().rectify(&source, &rectangle(0.0, 0.0, 5.0, 7.0)),
            Err(Error::EmptyImage)
        );
    }
}

/// Other decoded formats use the same RGB conversion as existing extraction.
#[test]
fn converts_luma_and_rgba_using_existing_rgb_semantics() {
    let luma =
        DynamicImage::ImageLuma8(GrayImage::from_fn(32, 40, |x, y| Luma([(x + y * 2) as u8])));
    let rgba = DynamicImage::ImageRgba8(RgbaImage::from_fn(32, 40, |x, y| {
        Rgba([x as u8 * 3, y as u8 * 2, 180, (x % 2) as u8 * 255])
    }));
    let corners = rectangle(3.0, 5.0, 27.0, 35.0);
    let extractor = sized_extractor(25, 31);
    for source in [luma, rgba] {
        let rgb = DynamicImage::ImageRgb8(source.to_rgb8());
        assert_eq!(
            extractor.rectify(&source, &corners).unwrap(),
            extractor.rectify(&rgb, &corners).unwrap()
        );
    }
}
