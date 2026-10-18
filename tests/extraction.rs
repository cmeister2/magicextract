//! Synthetic photographs exercise card detection without downloaded fixtures.

use image::{DynamicImage, Rgb, RgbImage, codecs::jpeg::JpegEncoder, imageops};
use imageproc::{
    drawing::{draw_filled_circle_mut, draw_filled_rect_mut, draw_polygon_mut},
    filter::gaussian_blur_f32,
    point::Point as PixelPoint,
    rect::Rect,
};
use magicextract::{CardDetection, CardExtractor, Error, ExtractorConfig, extract_cards};

/// A dark tabletop provides contrast around light card silhouettes.
const TABLE: Rgb<u8> = Rgb([24, 31, 37]);

/// An off-white card face avoids depending on saturated highlights.
const CARD: Rgb<u8> = Rgb([238, 235, 225]);

/// A diagnostic label and a mutation that should invalidate default settings.
type InvalidConfigCase = (&'static str, fn(&mut ExtractorConfig));

/// Paints a filled quadrilateral in source-image coordinates.
fn paint_quad(image: &mut RgbImage, corners: [(i32, i32); 4], color: Rgb<u8>) {
    let polygon: Vec<_> = corners
        .into_iter()
        .map(|(x, y)| PixelPoint::new(x, y))
        .collect();
    draw_polygon_mut(image, &polygon, color);
}

/// Constructs a tabletop containing one portrait card with square corners.
fn single_card() -> DynamicImage {
    let mut image = RgbImage::from_pixel(480, 420, TABLE);
    paint_quad(
        &mut image,
        [(120, 80), (300, 80), (300, 332), (120, 332)],
        CARD,
    );
    DynamicImage::ImageRgb8(image)
}

/// Verifies corners in source coordinates without prescribing cyclic ordering.
fn assert_corners_near(detection: &CardDetection, expected: [(i32, i32); 4], tolerance: f32) {
    for (x, y) in expected {
        let distance = detection
            .corners
            .iter()
            .map(|point| (point.x - x as f32).hypot(point.y - y as f32))
            .fold(f32::INFINITY, f32::min);
        assert!(
            distance <= tolerance,
            "expected corner ({x}, {y}) within {tolerance} pixels; got {:?}",
            detection.corners
        );
    }
}

/// Checks clockwise winding, portrait edge choice, and the canonical start point.
fn assert_canonical_corners(detection: &CardDetection) {
    let corners = &detection.corners;
    let signed_double_area: f32 = (0..4)
        .map(|index| {
            let next = (index + 1) % 4;
            corners[index].x * corners[next].y - corners[next].x * corners[index].y
        })
        .sum();
    assert!(
        signed_double_area > 0.0,
        "corners must wind clockwise: {corners:?}"
    );
    let lengths = std::array::from_fn::<_, 4, _>(|index| {
        let next = (index + 1) % 4;
        (corners[index].x - corners[next].x).hypot(corners[index].y - corners[next].y)
    });
    assert!(
        lengths[0] + lengths[2] < lengths[1] + lengths[3],
        "the first pair of opposite edges must be the short pair: {corners:?}"
    );
    assert!(
        (corners[0].y, corners[0].x) <= (corners[2].y, corners[2].x),
        "the uppermost, then leftmost portrait start must come first: {corners:?}"
    );
}

/// Measures neighboring-pixel differences in the center third of a grayscale face.
fn central_horizontal_detail(image: &RgbImage) -> f32 {
    let mut differences = 0_u64;
    let mut samples = 0_u64;
    for y in image.height() / 3..image.height() * 2 / 3 {
        for x in image.width() / 3..image.width() * 2 / 3 - 1 {
            differences +=
                u64::from(image.get_pixel(x, y).0[0].abs_diff(image.get_pixel(x + 1, y).0[0]));
            samples += 1;
        }
    }
    differences as f32 / samples as f32
}

/// A featureless image contains no card candidates.
#[test]
fn blank_image_has_no_cards() {
    let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(480, 420, TABLE));
    assert!(CardExtractor::default().detect(&image).unwrap().is_empty());
    assert!(extract_cards(&image).unwrap().is_empty());
}

/// An empty image is invalid input rather than a successful empty detection.
#[test]
fn empty_images_are_rejected() {
    for (width, height) in [(0, 0), (0, 20), (20, 0)] {
        let image = DynamicImage::ImageRgb8(RgbImage::new(width, height));
        assert_eq!(
            CardExtractor::default().detect(&image),
            Err(Error::EmptyImage)
        );
        assert!(matches!(extract_cards(&image), Err(Error::EmptyImage)));
    }
}

/// Nonempty images too small to contain an outline remain valid empty results.
#[test]
fn tiny_images_have_no_cards() {
    for (width, height) in [(1, 1), (1, 20), (20, 1), (2, 2)] {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, TABLE));
        assert!(CardExtractor::default().detect(&image).unwrap().is_empty());
        assert!(extract_cards(&image).unwrap().is_empty());
    }
}

/// A simple card produces one source-space quadrilateral and one portrait crop.
#[test]
fn extracts_single_card_at_default_resolution() {
    let cards = extract_cards(&single_card()).unwrap();
    assert_eq!(cards.len(), 1);
    assert_eq!(cards[0].image.dimensions(), (250, 350));
    assert_corners_near(
        &cards[0].detection,
        [(120, 80), (300, 80), (300, 332), (120, 332)],
        4.0,
    );
    assert!((cards[0].detection.area - 180.0 * 252.0).abs() < 2_000.0);
    assert_eq!(*cards[0].image.get_pixel(125, 175), CARD);
}

/// Both landscape and portrait silhouettes are retained as separate cards.
#[test]
fn detects_multiple_cards_in_different_orientations() {
    let mut image = RgbImage::from_pixel(720, 420, TABLE);
    let portrait = [(60, 70), (200, 70), (200, 266), (60, 266)];
    let landscape = [(400, 100), (596, 100), (596, 240), (400, 240)];
    paint_quad(&mut image, portrait, CARD);
    paint_quad(&mut image, landscape, CARD);
    let cards = extract_cards(&DynamicImage::ImageRgb8(image)).unwrap();
    assert_eq!(cards.len(), 2);
    for card in &cards {
        assert_eq!(card.image.dimensions(), (250, 350));
    }
    for expected in [portrait, landscape] {
        let nearest = cards
            .iter()
            .min_by(|left, right| {
                let distance = |card: &magicextract::ExtractedCard| {
                    card.detection
                        .corners
                        .iter()
                        .map(|point| {
                            (point.x - expected[0].0 as f32).hypot(point.y - expected[0].1 as f32)
                        })
                        .fold(f32::INFINITY, f32::min)
                };
                distance(left).total_cmp(&distance(right))
            })
            .unwrap();
        assert_corners_near(&nearest.detection, expected, 4.0);
    }
}

/// Detection does not depend on the card being brighter than its surroundings.
#[test]
fn detects_dark_card_on_light_background() {
    let mut image = RgbImage::from_pixel(480, 420, Rgb([240, 240, 240]));
    let corners = [(120, 80), (300, 80), (300, 332), (120, 332)];
    paint_quad(&mut image, corners, Rgb([35, 35, 35]));
    let detections = CardExtractor::default()
        .detect(&DynamicImage::ImageRgb8(image))
        .unwrap();
    assert_eq!(detections.len(), 1);
    assert_corners_near(&detections[0], corners, 4.0);
}

/// Rotation preserves the card shape and its reported source-image corners.
#[test]
fn detects_rotated_card() {
    let mut image = RgbImage::from_pixel(420, 420, TABLE);
    let corners = [(178, 60), (338, 128), (242, 352), (82, 284)];
    paint_quad(&mut image, corners, CARD);
    let detections = CardExtractor::default()
        .detect(&DynamicImage::ImageRgb8(image))
        .unwrap();
    assert_eq!(detections.len(), 1);
    assert_corners_near(&detections[0], corners, 5.0);
}

/// Perspective rectification retains four colored landmarks inside the card.
#[test]
fn rectifies_perspective_and_preserves_face_pixels() {
    let mut image = RgbImage::from_pixel(420, 420, TABLE);
    let corners = [(90, 60), (270, 80), (280, 340), (70, 320)];
    paint_quad(&mut image, corners, CARD);
    let landmarks = [
        ((130, 110), Rgb([220, 30, 30])),
        ((230, 120), Rgb([30, 180, 30])),
        ((120, 280), Rgb([30, 30, 220])),
        ((240, 290), Rgb([220, 180, 30])),
    ];
    for (center, color) in landmarks {
        draw_filled_circle_mut(&mut image, center, 13, color);
    }
    let cards = extract_cards(&DynamicImage::ImageRgb8(image)).unwrap();
    assert_eq!(cards.len(), 1);
    let output = &cards[0].image;
    let mut occupied_quadrants = Vec::new();
    for (_, color) in landmarks {
        let matching: Vec<_> = output
            .enumerate_pixels()
            .filter(|(_, _, pixel)| {
                pixel
                    .0
                    .iter()
                    .zip(color.0)
                    .all(|(&actual, expected)| actual.abs_diff(expected) < 15)
            })
            .map(|(x, y, _)| (x as f32, y as f32))
            .collect();
        assert!(matching.len() > 200, "lost landmark {color:?}");
        let count = matching.len() as f32;
        let center_x = matching.iter().map(|point| point.0).sum::<f32>() / count;
        let center_y = matching.iter().map(|point| point.1).sum::<f32>() / count;
        assert!(
            (30.0..85.0).contains(&center_x) || (165.0..220.0).contains(&center_x),
            "landmark {color:?} has unexpected horizontal location {center_x}"
        );
        assert!(
            (35.0..100.0).contains(&center_y) || (250.0..315.0).contains(&center_y),
            "landmark {color:?} has unexpected vertical location {center_y}"
        );
        occupied_quadrants.push((center_x > 125.0, center_y > 175.0));
    }
    let expected_quadrants = [(false, false), (true, false), (false, true), (true, true)];
    let rotated_quadrants = expected_quadrants.map(|(right, bottom)| (!right, !bottom));
    assert!(
        occupied_quadrants == expected_quadrants || occupied_quadrants == rotated_quadrants,
        "landmark layout must match the source or its 180-degree rotation: {occupied_quadrants:?}"
    );
    occupied_quadrants.sort_unstable();
    occupied_quadrants.dedup();
    assert_eq!(occupied_quadrants.len(), 4);
    assert_corners_near(&cards[0].detection, corners, 6.0);
}

/// Rounded physical corners still produce a single rectangular card crop.
#[test]
fn detects_rounded_card_corners() {
    let mut image = RgbImage::from_pixel(400, 400, TABLE);
    draw_filled_rect_mut(&mut image, Rect::at(94, 70).of_size(122, 210), CARD);
    draw_filled_rect_mut(&mut image, Rect::at(80, 84).of_size(150, 182), CARD);
    for center in [(94, 84), (215, 84), (94, 265), (215, 265)] {
        draw_filled_circle_mut(&mut image, center, 14, CARD);
    }
    let cards = extract_cards(&DynamicImage::ImageRgb8(image)).unwrap();
    assert_eq!(cards.len(), 1);
    assert_corners_near(
        &cards[0].detection,
        [(80, 70), (229, 70), (229, 279), (80, 279)],
        15.0,
    );
}

/// Square and very narrow objects do not pass the trading card shape filter.
#[test]
fn rejects_objects_with_wrong_aspect_ratios() {
    let mut image = RgbImage::from_pixel(620, 440, TABLE);
    paint_quad(
        &mut image,
        [(40, 40), (180, 40), (180, 180), (40, 180)],
        CARD,
    );
    paint_quad(
        &mut image,
        [(270, 70), (550, 70), (550, 120), (270, 120)],
        CARD,
    );
    assert!(
        extract_cards(&DynamicImage::ImageRgb8(image))
            .unwrap()
            .is_empty()
    );
}

/// Printed borders and the two sides of an edge do not duplicate a card.
#[test]
fn suppresses_duplicate_nested_card_borders() {
    let mut image = single_card().into_rgb8();
    paint_quad(
        &mut image,
        [(130, 90), (290, 90), (290, 322), (130, 322)],
        Rgb([45, 45, 45]),
    );
    paint_quad(
        &mut image,
        [(134, 94), (286, 94), (286, 318), (134, 318)],
        CARD,
    );
    assert_eq!(
        extract_cards(&DynamicImage::ImageRgb8(image))
            .unwrap()
            .len(),
        1
    );
}

/// Tiny card-like image details are excluded by the configured area floor.
#[test]
fn ignores_shapes_below_minimum_area() {
    let mut image = RgbImage::from_pixel(600, 600, TABLE);
    paint_quad(&mut image, [(30, 30), (50, 30), (50, 58), (30, 58)], CARD);
    let large = [(200, 100), (420, 100), (420, 408), (200, 408)];
    paint_quad(&mut image, large, CARD);
    let detections = CardExtractor::default()
        .detect(&DynamicImage::ImageRgb8(image))
        .unwrap();
    assert_eq!(detections.len(), 1);
    assert_corners_near(&detections[0], large, 4.0);
}

/// Detection resizing must not change the coordinate space of returned corners.
#[test]
fn reports_original_coordinates_after_detection_downscaling() {
    let mut image = RgbImage::from_pixel(1_200, 1_000, TABLE);
    let corners = [(300, 220), (600, 220), (600, 640), (300, 640)];
    paint_quad(&mut image, corners, CARD);
    let extractor = CardExtractor::new(ExtractorConfig {
        max_detection_dimension: 400,
        ..ExtractorConfig::default()
    })
    .unwrap();
    let cards = extractor.extract(&DynamicImage::ImageRgb8(image)).unwrap();
    assert_eq!(cards.len(), 1);
    assert_corners_near(&cards[0].detection, corners, 10.0);
    assert!((cards[0].detection.area - 300.0 * 420.0).abs() < 8_000.0);
    assert_eq!(*cards[0].image.get_pixel(125, 175), CARD);
}

/// Fine face texture survives detection downscaling because crops sample the source.
#[test]
fn preserves_original_face_detail_after_detection_downscaling() {
    let mut source = RgbImage::from_pixel(1_200, 1_000, TABLE);
    paint_quad(
        &mut source,
        [(300, 220), (600, 220), (600, 640), (300, 640)],
        CARD,
    );
    for y in 250..610 {
        for x in 330..570 {
            let intensity = if (x / 2 + y / 2) % 2 == 0 { 0 } else { 255 };
            source.put_pixel(x, y, Rgb([intensity; 3]));
        }
    }
    let reduced = imageops::resize(&source, 400, 333, imageops::FilterType::Triangle);
    let extractor = CardExtractor::new(ExtractorConfig {
        output_width: 600,
        output_height: 840,
        max_detection_dimension: 400,
        ..ExtractorConfig::default()
    })
    .unwrap();
    let original_cards = extractor.extract(&DynamicImage::ImageRgb8(source)).unwrap();
    let reduced_cards = extractor
        .extract(&DynamicImage::ImageRgb8(reduced))
        .unwrap();
    assert_eq!(original_cards.len(), 1);
    assert_eq!(reduced_cards.len(), 1);
    assert_eq!(original_cards[0].image.dimensions(), (600, 840));
    let original_detail = central_horizontal_detail(&original_cards[0].image);
    let reduced_detail = central_horizontal_detail(&reduced_cards[0].image);
    assert!(
        original_detail > 30.0,
        "source checkerboard should retain strong local contrast: {original_detail}"
    );
    assert!(
        reduced_detail < 8.0,
        "the low-resolution control should lose the fine checkerboard: {reduced_detail}"
    );
}

/// Callers can choose output dimensions independently of detection resizing.
#[test]
fn honors_custom_output_dimensions() {
    let extractor = CardExtractor::new(ExtractorConfig {
        output_width: 100,
        output_height: 140,
        ..ExtractorConfig::default()
    })
    .unwrap();
    let cards = extractor.extract(&single_card()).unwrap();
    assert_eq!(cards.len(), 1);
    assert_eq!(cards[0].image.dimensions(), (100, 140));
}

/// Both configured area bounds apply to the fraction of the full input image.
#[test]
fn honors_custom_area_bounds() {
    for config in [
        ExtractorConfig {
            min_area_fraction: 0.30,
            ..ExtractorConfig::default()
        },
        ExtractorConfig {
            max_area_fraction: 0.10,
            ..ExtractorConfig::default()
        },
    ] {
        assert!(
            CardExtractor::new(config)
                .unwrap()
                .detect(&single_card())
                .unwrap()
                .is_empty()
        );
    }
}

/// The result cap is applied to distinct cards.
#[test]
fn honors_maximum_card_count() {
    let mut image = RgbImage::from_pixel(600, 300, TABLE);
    for x in [40, 230, 420] {
        paint_quad(
            &mut image,
            [(x, 60), (x + 90, 60), (x + 90, 186), (x, 186)],
            CARD,
        );
    }
    let extractor = CardExtractor::new(ExtractorConfig {
        max_cards: 2,
        ..ExtractorConfig::default()
    })
    .unwrap();
    let detections = extractor.detect(&DynamicImage::ImageRgb8(image)).unwrap();
    assert_eq!(detections.len(), 2);
}

/// A cap retains the largest outlines, then returns them in stable spatial order.
#[test]
fn caps_largest_cards_and_returns_deterministic_spatial_order() {
    let mut image = RgbImage::from_pixel(800, 600, TABLE);
    let upper_left = [(60, 70), (200, 70), (200, 266), (60, 266)];
    let upper_right = [(300, 70), (420, 70), (420, 238), (300, 238)];
    let lower_large = [(480, 300), (660, 300), (660, 552), (480, 552)];
    let lower_small = [(60, 400), (120, 400), (120, 484), (60, 484)];
    for corners in [upper_left, upper_right, lower_large, lower_small] {
        paint_quad(&mut image, corners, CARD);
    }
    let image = DynamicImage::ImageRgb8(image);
    let extractor = CardExtractor::new(ExtractorConfig {
        max_cards: 3,
        ..ExtractorConfig::default()
    })
    .unwrap();
    let detections = extractor.detect(&image).unwrap();
    assert_eq!(detections.len(), 3);
    assert_eq!(extractor.detect(&image).unwrap(), detections);
    for (detection, expected) in detections
        .iter()
        .zip([upper_left, upper_right, lower_large])
    {
        assert_corners_near(detection, expected, 4.0);
        assert_canonical_corners(detection);
    }
    assert!(detections[2].area > detections[0].area);
    assert!(detections[0].area > detections[1].area);
}

/// A spread of angles exercises portrait normalization across quadrant changes.
#[test]
fn detects_cards_across_rotation_angles() {
    for degrees in [-75.0_f32, -40.0, -12.0, 0.0, 28.0, 65.0, 110.0, 175.0] {
        let angle = degrees.to_radians();
        let (sin, cos) = angle.sin_cos();
        let corners = [(-70.0, -98.0), (70.0, -98.0), (70.0, 98.0), (-70.0, 98.0)].map(|(x, y)| {
            (
                (320.0 + x * cos - y * sin).round() as i32,
                (260.0 + x * sin + y * cos).round() as i32,
            )
        });
        let mut image = RgbImage::from_pixel(640, 520, TABLE);
        paint_quad(&mut image, corners, CARD);
        let cards = extract_cards(&DynamicImage::ImageRgb8(image)).unwrap();
        assert_eq!(cards.len(), 1, "rotation {degrees}");
        assert_corners_near(&cards[0].detection, corners, 6.0);
        assert_canonical_corners(&cards[0].detection);
        assert_eq!(cards[0].image.dimensions(), (250, 350));
    }
}

/// Size and tabletop hue do not prescribe a specific photographic setup.
#[test]
fn detects_varied_card_scales_and_background_colors() {
    for (width, background, face) in [
        (60, Rgb([24, 85, 48]), CARD),
        (110, Rgb([95, 42, 30]), CARD),
        (180, Rgb([55, 40, 105]), CARD),
        (240, Rgb([210, 210, 205]), Rgb([35, 40, 45])),
    ] {
        let height = width * 7 / 5;
        let left = 320 - width / 2;
        let top = 260 - height / 2;
        let corners = [
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
        ];
        let mut image = RgbImage::from_pixel(640, 520, background);
        paint_quad(&mut image, corners, face);
        let cards = extract_cards(&DynamicImage::ImageRgb8(image)).unwrap();
        assert_eq!(
            cards.len(),
            1,
            "card width {width}, background {background:?}"
        );
        assert_corners_near(&cards[0].detection, corners, 5.0);
    }
}

/// Mild camera blur, sensor noise, and JPEG loss retain the isolated outline.
#[test]
fn detects_cards_after_mild_photographic_degradation() {
    let original = single_card().into_rgb8();
    let blurred = gaussian_blur_f32(&original, 0.8);
    let mut noisy = original.clone();
    for (x, y, pixel) in noisy.enumerate_pixels_mut() {
        let offset = ((x.wrapping_mul(73) ^ y.wrapping_mul(151)) % 9) as i16 - 4;
        for channel in &mut pixel.0 {
            *channel = (i16::from(*channel) + offset).clamp(0, 255) as u8;
        }
    }
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 75)
        .encode_image(&DynamicImage::ImageRgb8(original))
        .unwrap();
    let compressed = image::load_from_memory(&jpeg).unwrap().into_rgb8();
    for (name, image) in [("blur", blurred), ("noise", noisy), ("jpeg", compressed)] {
        let cards = extract_cards(&DynamicImage::ImageRgb8(image)).unwrap();
        assert_eq!(cards.len(), 1, "degradation {name}");
        assert_corners_near(
            &cards[0].detection,
            [(120, 80), (300, 80), (300, 332), (120, 332)],
            6.0,
        );
    }
}

/// Invalid numeric settings are rejected at construction before image work.
#[test]
fn rejects_invalid_configuration() {
    let invalid: &[InvalidConfigCase] = &[
        ("zero output width", |config| config.output_width = 0),
        ("zero output height", |config| config.output_height = 0),
        ("output width below minimum", |config| {
            config.output_width = 1
        }),
        ("output height above maximum", |config| {
            config.output_height = 8193
        }),
        ("zero detection dimension", |config| {
            config.max_detection_dimension = 0
        }),
        ("detection dimension below minimum", |config| {
            config.max_detection_dimension = 31
        }),
        ("detection dimension above maximum", |config| {
            config.max_detection_dimension = 4097
        }),
        ("zero maximum cards", |config| config.max_cards = 0),
        ("zero aspect tolerance", |config| {
            config.aspect_ratio_tolerance = 0.0
        }),
        ("unit aspect tolerance", |config| {
            config.aspect_ratio_tolerance = 1.0
        }),
        ("negative aspect tolerance", |config| {
            config.aspect_ratio_tolerance = -0.1
        }),
        ("NaN aspect tolerance", |config| {
            config.aspect_ratio_tolerance = f32::NAN
        }),
        ("infinite aspect tolerance", |config| {
            config.aspect_ratio_tolerance = f32::INFINITY
        }),
        ("negative area floor", |config| {
            config.min_area_fraction = -0.1
        }),
        ("zero area floor", |config| config.min_area_fraction = 0.0),
        ("NaN area floor", |config| {
            config.min_area_fraction = f32::NAN
        }),
        ("area ceiling above one", |config| {
            config.max_area_fraction = 1.1
        }),
        ("NaN area ceiling", |config| {
            config.max_area_fraction = f32::NAN
        }),
        ("reversed area bounds", |config| {
            config.min_area_fraction = 0.5;
            config.max_area_fraction = 0.1;
        }),
        ("equal area bounds", |config| {
            config.max_area_fraction = config.min_area_fraction
        }),
        ("zero edge threshold", |config| {
            config.canny_low_threshold = 0.0
        }),
        ("negative edge threshold", |config| {
            config.canny_low_threshold = -1.0
        }),
        ("NaN low edge threshold", |config| {
            config.canny_low_threshold = f32::NAN
        }),
        ("infinite high edge threshold", |config| {
            config.canny_high_threshold = f32::INFINITY
        }),
        ("reversed edge thresholds", |config| {
            config.canny_low_threshold = 100.0;
            config.canny_high_threshold = 30.0;
        }),
        ("equal edge thresholds", |config| {
            config.canny_high_threshold = config.canny_low_threshold
        }),
    ];
    for &(description, invalidate) in invalid {
        let mut config = ExtractorConfig::default();
        invalidate(&mut config);
        assert!(
            CardExtractor::new(config).is_err(),
            "accepted {description}"
        );
    }
}
