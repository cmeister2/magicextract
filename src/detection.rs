//! Contour-based detection and conservative recovery of trading card outlines.

use image::{DynamicImage, GrayImage, imageops};
use imageproc::{
    contours::find_contours,
    contrast::{ThresholdType, otsu_level, threshold},
    edges::canny,
};

use crate::{CardDetection, ExtractorConfig, Point, TRADING_CARD_ASPECT_RATIO};

/// Douglas–Peucker tolerances, expressed as fractions of a contour's perimeter.
const SIMPLIFICATION_FRACTIONS: [f32; 6] = [0.008, 0.012, 0.018, 0.025, 0.035, 0.05];

/// Maximum intersection-over-union for two candidates to remain separate.
const DUPLICATE_IOU: f32 = 0.6;

/// Fraction of a smaller outline inside another that identifies an interior panel.
const INTERIOR_CONTAINMENT: f32 = 0.95;

/// Minimum contour solidity for the primary four-corner approximation path.
const MINIMUM_CONTOUR_SOLIDITY: f32 = 0.85;

/// Fraction of every inferred edge that must have an observed contour nearby.
const MINIMUM_EDGE_COVERAGE: f32 = 0.6;

/// Bounded line pool that leaves room for short visible sides beside printed panels.
const MAXIMUM_OBSERVED_EDGES: usize = 12;

/// Number of distributed luminance profiles used to verify an enclosing border.
const BORDER_PROFILE_SAMPLES: usize = 48;

/// Contrast statistics across a proposed edge, measured along its middle section.
struct BorderProfile {
    /// Signed outward contrast exceeded by three quarters of the samples.
    lower_quartile: f32,
    /// Absolute outward contrast exceeded by three quarters of the samples.
    absolute_lower_quartile: f32,
    /// Signed outward contrast exceeded by one quarter of the samples.
    upper_quartile: f32,
    /// Minimum coverage of frame continuity and persistent outer contrast.
    band_samples: usize,
}

/// A straight observed section of a contour, used to infer missing corner joins.
#[derive(Clone, Copy)]
struct ObservedEdge {
    /// First observed endpoint in detection-image coordinates.
    start: Point,
    /// Last observed endpoint in detection-image coordinates.
    end: Point,
    /// Distance between the observed endpoints.
    length: f32,
}

/// An inferred outline and the retained cards that explain its hidden sections.
struct InferredOutline {
    /// Card geometry supported by the observed contour.
    detection: CardDetection,
    /// Indices into the occluder slice supplied for this inference pass.
    occluder_indices: Vec<usize>,
}

impl InferredOutline {
    /// Requires one associated card to explain both the gaps and clipped side.
    fn can_replace(&self, original: &CardDetection, occluders: &[CardDetection]) -> bool {
        self.occluder_indices.iter().any(|&index| {
            occluders.get(index).is_some_and(|occluder| {
                replacement_follows_three_sides(
                    original,
                    &self.detection,
                    std::slice::from_ref(occluder),
                )
            })
        })
    }
}

/// Finds card quadrilaterals and expresses their corners in source-image pixels.
pub(crate) fn detect(image: &DynamicImage, config: &ExtractorConfig) -> Vec<CardDetection> {
    if image.width() < 3 || image.height() < 3 {
        return Vec::new();
    }
    let gray = working_image(image, config.max_detection_dimension);
    if gray.width() < 3 || gray.height() < 3 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    let edges = canny(
        &gray,
        config.canny_low_threshold,
        config.canny_high_threshold,
    );
    collect_candidates(&edges, config, &mut candidates);

    // Both foreground polarities recover complete borders that Canny can break
    // around rounded corners, and permit dark card backs on light surfaces.
    let minimum = gray.iter().copied().min().unwrap_or(0);
    let maximum = gray.iter().copied().max().unwrap_or(0);
    if minimum != maximum {
        // Split nonuniform intensities into masks that can recover borders missed by Canny.
        let level = otsu_level(&gray);
        for polarity in [ThresholdType::Binary, ThresholdType::BinaryInverted] {
            let mask = threshold(&gray, level, polarity);
            collect_candidates(&mask, config, &mut candidates);
        }
        // A dark border on a dark table can belong to the same class in the
        // global split. Splitting that class once more recovers its full outline
        // when Canny edges have small gaps at rounded or reflective corners.
        if let Some(darker_level) = darker_otsu_level(&gray, level) {
            let mask = threshold(&gray, darker_level, ThresholdType::BinaryInverted);
            collect_candidates(&mask, config, &mut candidates);
        }
    }

    // A dim border can have connected weak edges without any gradient strong
    // enough to seed the default Canny pass. Retry only when no mask found a
    // plausible outline, scaling thresholds to the available contrast and
    // limiting amplification to twofold so small intensity noise stays weak.
    let contrast = maximum - minimum;
    if candidates.is_empty() && contrast > 0 && contrast < u8::MAX {
        let scale = (f32::from(contrast) / f32::from(u8::MAX)).max(0.5);
        let edges = canny(
            &gray,
            config.canny_low_threshold * scale,
            config.canny_high_threshold * scale,
        );
        collect_candidates(&edges, config, &mut candidates);
    }

    // Prefer exterior borders over duplicate edges and contained printed panels.
    // A completely contained smaller physical card is geometrically ambiguous
    // with artwork, so the enclosing outline takes priority in that case too.
    candidates.sort_by(|left, right| {
        right
            .area
            .total_cmp(&left.area)
            .then_with(|| spatial_order(left, right))
    });
    let mut retained: Vec<CardDetection> = Vec::new();
    for mut candidate in candidates {
        if retained.iter().all(|existing| {
            let (iou, containment) = quadrilateral_overlap(existing, &candidate);
            iou < DUPLICATE_IOU && containment < INTERIOR_CONTAINMENT
        }) {
            // Polish an unsuppressed outline, then recheck its proportions before retaining it.
            let refined = refine_gradient_corners(candidate.corners, &gray);
            let Some(corners) = orient_card_corners(
                refined,
                config.aspect_ratio_tolerance,
                gray.width(),
                gray.height(),
            ) else {
                continue;
            };
            candidate.corners = corners;
            candidate.area = polygon_area(&corners);
            retained.push(candidate);
            if retained.len() == config.max_cards {
                break;
            }
        }
    }
    // A printed panel can be the only closed contour on a busy surface. Recover
    // its surrounding card frame only when all four outer edges are seen.
    // Refine only retained detections so suppressed contours cannot reappear as
    // extra cards when their expanded geometry changes overlap relationships.
    for candidate in &mut retained {
        if let Some(expanded) = expand_card_border(candidate.corners, &gray)
            && let Some(corners) = orient_card_corners(
                refine_gradient_corners(expanded, &gray),
                config.aspect_ratio_tolerance,
                gray.width(),
                gray.height(),
            )
        {
            let area = polygon_area(&corners);
            let image_area = gray.width() as f32 * gray.height() as f32;
            if area >= config.min_area_fraction * image_area
                && area <= config.max_area_fraction * image_area
            {
                candidate.corners = corners;
                candidate.area = area;
            }
        }
    }

    refine_occluded_outlines(&edges, &gray, config, &mut retained);

    // Enlarged borders can overlap another retained card; suppress duplicates
    // again using the final geometry before mapping corners into source pixels.
    retained.sort_by(|left, right| {
        right
            .area
            .total_cmp(&left.area)
            .then_with(|| spatial_order(left, right))
    });
    let mut distinct = Vec::new();
    for candidate in retained {
        if distinct.iter().all(|existing| {
            let (iou, containment) = quadrilateral_overlap(existing, &candidate);
            iou < DUPLICATE_IOU && containment < INTERIOR_CONTAINMENT
        }) {
            distinct.push(candidate);
        }
    }
    let mut retained = distinct;
    // A clear sleeve may surround a separately observed physical card frame.
    // Refine only retained outlines so nested printed panels stay suppressed.
    for candidate in &mut retained {
        if let Some(corners) = recover_inner_card_border(candidate.corners, &gray, config)
            .or_else(|| recover_inner_artwork(candidate.corners, &gray, image, config))
        {
            candidate.corners = corners;
            candidate.area = polygon_area(&corners);
        }
    }
    let scale_x = image.width() as f32 / gray.width() as f32;
    let scale_y = image.height() as f32 / gray.height() as f32;
    for candidate in &mut retained {
        for point in &mut candidate.corners {
            point.x = ((point.x + 0.5) * scale_x - 0.5).clamp(0.0, image.width() as f32 - 1.0);
            point.y = ((point.y + 0.5) * scale_y - 0.5).clamp(0.0, image.height() as f32 - 1.0);
        }
        candidate.area = polygon_area(&candidate.corners);
    }
    retained.sort_by(spatial_order);
    retained
}

/// Splits the darker global Otsu class using its own between-class variance.
fn darker_otsu_level(image: &GrayImage, upper_level: u8) -> Option<u8> {
    let mut counts = [0_u32; 256];
    for &value in image.iter().filter(|&&value| value <= upper_level) {
        counts[value as usize] += 1;
    }
    let total_weight = counts.iter().map(|&count| f64::from(count)).sum::<f64>();
    let total_sum = counts
        .iter()
        .enumerate()
        .map(|(level, &count)| level as f64 * f64::from(count))
        .sum::<f64>();
    let mut background_weight = 0.0;
    let mut background_sum = 0.0;
    let mut greatest_variance = 0.0;
    let mut best_level = 0;
    for (level, &count) in counts.iter().enumerate().take(upper_level as usize + 1) {
        // Sweep the dark class's thresholds and keep the split with greatest variance.
        background_weight += f64::from(count);
        background_sum += level as f64 * f64::from(count);
        let foreground_weight = total_weight - background_weight;
        if background_weight == 0.0 || foreground_weight == 0.0 {
            continue;
        }
        let mean_difference =
            background_sum / background_weight - (total_sum - background_sum) / foreground_weight;
        let variance = background_weight * foreground_weight * mean_difference.powi(2);
        if variance > greatest_variance {
            greatest_variance = variance;
            best_level = level as u8;
        }
    }
    (greatest_variance > 0.0).then_some(best_level)
}

/// Produces a bounded grayscale image without enlarging small inputs.
fn working_image(image: &DynamicImage, maximum_dimension: u32) -> GrayImage {
    let gray = image.to_luma8();
    let longest = gray.width().max(gray.height());
    if longest <= maximum_dimension {
        return gray;
    }
    let scale = maximum_dimension as f64 / longest as f64;
    let width = (gray.width() as f64 * scale).round().max(1.0) as u32;
    let height = (gray.height() as f64 * scale).round().max(1.0) as u32;
    imageops::resize(&gray, width, height, imageops::FilterType::Triangle)
}

/// Adds geometrically plausible quadrilaterals from one binary mask.
fn collect_candidates(
    mask: &GrayImage,
    config: &ExtractorConfig,
    candidates: &mut Vec<CardDetection>,
) {
    let image_area = mask.width() as f32 * mask.height() as f32;
    let minimum_area = config.min_area_fraction * image_area;
    let maximum_area = config.max_area_fraction * image_area;
    for contour in find_contours::<i32>(mask) {
        // Reject unsuitable contours before fitting supported card outlines to their hulls.
        if contour.points.len() < 4 {
            continue;
        }
        let points: Vec<Point> = contour
            .points
            .iter()
            .map(|point| Point {
                x: point.x as f32,
                y: point.y as f32,
            })
            .collect();
        let (minimum, maximum) = bounds(&points);
        if (maximum.x - minimum.x) * (maximum.y - minimum.y) < minimum_area {
            continue;
        }
        // Thresholding the background also yields the image's outer boundary.
        // It has no observed card border and must never become a detection.
        if minimum.x <= 0.0
            && minimum.y <= 0.0
            && maximum.x >= mask.width() as f32 - 1.0
            && maximum.y >= mask.height() as f32 - 1.0
        {
            continue;
        }
        let hull = convex_hull(points.clone());
        let hull_area = polygon_area(&hull);
        if hull.len() < 4 || hull_area < minimum_area || hull_area > maximum_area * 1.15 {
            continue;
        }
        let perimeter = polygon_perimeter(&hull);
        let mut found = false;
        if polygon_area(&points) >= hull_area * MINIMUM_CONTOUR_SOLIDITY {
            for fraction in SIMPLIFICATION_FRACTIONS {
                // Try progressively coarser hulls until four supported card corners remain.
                let simplified = simplify_closed(&hull, fraction * perimeter);
                let Ok(corners) = <[Point; 4]>::try_from(simplified) else {
                    continue;
                };
                let corners =
                    refine_corners(order_corners(corners), &points, mask.width(), mask.height());
                let area = polygon_area(&corners);
                if area >= minimum_area
                    && area <= maximum_area
                    && area >= hull_area * 0.85
                    && edges_are_supported(&corners, &points)
                    && let Some(corners) = orient_card_corners(
                        corners,
                        config.aspect_ratio_tolerance,
                        mask.width(),
                        mask.height(),
                    )
                {
                    // Keep the first supported approximation for this contour.
                    candidates.push(CardDetection { corners, area });
                    found = true;
                    break;
                }
            }
        }
        if !found {
            let mut inferred = Vec::new();
            collect_inferred_candidates(&points, perimeter, mask, config, &[], &mut inferred);
            candidates.extend(inferred.into_iter().map(|proposal| proposal.detection));
        }
    }
}

/// Infers quadrilateral corners from four well-supported observed edge lines.
///
/// A small corner bite or protruding occluder can add vertices to the hull or
/// break a Canny border into an open contour. Straight portions still locate the
/// four edge lines. Intersections restore their joins without inventing pixels.
fn collect_inferred_candidates(
    contour: &[Point],
    perimeter: f32,
    mask: &GrayImage,
    config: &ExtractorConfig,
    occluders: &[CardDetection],
    candidates: &mut Vec<InferredOutline>,
) {
    let simplified = simplify_closed(contour, (perimeter * 0.0015).clamp(1.0, 2.0));
    let mut segments: Vec<ObservedEdge> = simplified
        .iter()
        .zip(simplified.iter().cycle().skip(1))
        .filter_map(|(&start, &end)| {
            let length = squared_distance(start, end).sqrt();
            (length >= perimeter * 0.03).then_some(ObservedEdge { start, end, length })
        })
        .collect();
    segments.sort_by(|left, right| {
        right
            .length
            .total_cmp(&left.length)
            .then_with(|| point_order(left.start, right.start))
    });
    let mut edges: Vec<ObservedEdge> = Vec::new();
    for segment in segments {
        if edges
            .iter()
            .all(|existing| !same_edge_line(existing, &segment))
        {
            edges.push(segment);
            if edges.len() == MAXIMUM_OBSERVED_EDGES {
                break;
            }
        }
    }
    if edges.len() < 4 {
        return;
    }
    let (minimum, maximum) = bounds(contour);
    let center = Point {
        x: (minimum.x + maximum.x) * 0.5,
        y: (minimum.y + maximum.y) * 0.5,
    };
    let image_area = mask.width() as f32 * mask.height() as f32;
    for first in 0..edges.len() - 3 {
        for second in first + 1..edges.len() - 2 {
            for third in second + 1..edges.len() - 1 {
                for fourth in third + 1..edges.len() {
                    // Accept this four-line intersection only with in-frame, supported card edges.
                    let selected = [edges[first], edges[second], edges[third], edges[fourth]];
                    let Some(corners) = corners_from_edges(selected, center) else {
                        continue;
                    };
                    let area = polygon_area(&corners);
                    if area < config.min_area_fraction * image_area
                        || area > config.max_area_fraction * image_area
                        || corners.iter().any(|point| {
                            point.x < 0.0
                                || point.y < 0.0
                                || point.x > mask.width() as f32 - 1.0
                                || point.y > mask.height() as f32 - 1.0
                        })
                    {
                        continue;
                    }
                    let occluder_indices = if occluders.is_empty() {
                        if !edges_are_supported(&corners, contour) {
                            continue;
                        }
                        Vec::new()
                    } else {
                        let indices: Vec<_> = occluders
                            .iter()
                            .enumerate()
                            .filter_map(|(index, occluder)| {
                                occlusion_explains_edges(
                                    &corners,
                                    contour,
                                    std::slice::from_ref(occluder),
                                )
                                .then_some(index)
                            })
                            .collect();
                        if indices.is_empty() {
                            continue;
                        }
                        indices
                    };
                    if let Some(corners) = orient_card_corners(
                        corners,
                        config.aspect_ratio_tolerance,
                        mask.width(),
                        mask.height(),
                    ) {
                        candidates.push(InferredOutline {
                            detection: CardDetection { corners, area },
                            occluder_indices,
                        });
                    }
                }
            }
        }
    }
}

/// Tests whether two observed segments lie on approximately the same line.
fn same_edge_line(first: &ObservedEdge, second: &ObservedEdge) -> bool {
    let dot = (first.end.x - first.start.x) * (second.end.x - second.start.x)
        + (first.end.y - first.start.y) * (second.end.y - second.start.y);
    dot.abs() / (first.length * second.length) > 0.998
        && cross(first.start, first.end, second.start).abs() / first.length <= 2.5
        && cross(first.start, first.end, second.end).abs() / first.length <= 2.5
}

/// Intersects four observed lines in their angular order around a contour center.
fn corners_from_edges(mut edges: [ObservedEdge; 4], center: Point) -> Option<[Point; 4]> {
    edges.sort_by(|left, right| {
        // Order the lines by their midpoint angles so adjacent entries form corner joins.
        let left_angle = ((left.start.y + left.end.y) * 0.5 - center.y)
            .atan2((left.start.x + left.end.x) * 0.5 - center.x);
        let right_angle = ((right.start.y + right.end.y) * 0.5 - center.y)
            .atan2((right.start.x + right.end.x) * 0.5 - center.x);
        left_angle.total_cmp(&right_angle)
    });
    let lines = edges.map(|edge| {
        (
            edge.start,
            Point {
                x: (edge.end.x - edge.start.x) / edge.length,
                y: (edge.end.y - edge.start.y) / edge.length,
            },
        )
    });
    let mut corners = [center; 4];
    for index in 0..4 {
        corners[index] = intersect_lines(lines[(index + 3) % 4], lines[index])?;
    }
    Some(order_corners(corners))
}

/// Requires distributed contour evidence along every proposed card edge.
fn edges_are_supported(corners: &[Point; 4], contour: &[Point]) -> bool {
    (0..4).all(|index| {
        // Bin nearby contour evidence along this edge to require distributed support.
        let start = corners[index];
        let end = corners[(index + 1) % 4];
        let length_squared = squared_distance(start, end);
        let length = length_squared.sqrt();
        if length < 1.0 {
            return false;
        }
        let bin_count = (length.ceil() as usize).clamp(4, 32);
        let mut occupied = [false; 32];
        let maximum_distance = (length * 0.006).clamp(2.0, 3.0);
        for &point in contour {
            let projection = ((point.x - start.x) * (end.x - start.x)
                + (point.y - start.y) * (end.y - start.y))
                / length_squared;
            if (-0.01..=1.01).contains(&projection)
                && cross(start, end, point).abs() / length <= maximum_distance
            {
                let bin = (projection.clamp(0.0, 1.0) * bin_count as f32) as usize;
                occupied[bin.min(bin_count - 1)] = true;
            }
        }
        let covered = occupied[..bin_count].iter().filter(|&&value| value).count();
        covered as f32 / bin_count as f32 >= MINIMUM_EDGE_COVERAGE
    })
}

/// Fits the straight middle sections of contour edges to restore sharp corners.
///
/// Rounded corners and rasterized diagonal edges can make Douglas–Peucker pick
/// a point slightly before the true corner. Intersecting fitted edge lines also
/// recovers the surrounding rectangular crop for a card with rounded corners.
fn refine_corners(corners: [Point; 4], contour: &[Point], width: u32, height: u32) -> [Point; 4] {
    let perimeter = polygon_perimeter(&corners);
    let lines: [(Point, Point); 4] = std::array::from_fn(|index| {
        fit_edge(
            corners[index],
            corners[(index + 1) % 4],
            contour,
            (perimeter * 0.015).max(2.0),
        )
    });
    let mut refined = corners;
    for index in 0..4 {
        // Use each fitted intersection only when it stays near the original corner and in frame.
        let Some(point) = intersect_lines(lines[(index + 3) % 4], lines[index]) else {
            return corners;
        };
        if point.x < 0.0
            || point.y < 0.0
            || point.x > width as f32 - 1.0
            || point.y > height as f32 - 1.0
            || squared_distance(point, corners[index]) > (perimeter * 0.04).powi(2)
        {
            return corners;
        }
        refined[index] = point;
    }
    order_corners(refined)
}

/// Removes binary-mask pixel bias using nearby grayscale gradient maxima.
///
/// Only final accepted outlines are polished. The search remains local to each
/// observed edge, ignores rounded corner sections, and keeps the chosen physical
/// portrait orientation. This matters when a foreshortened image is enlarged.
fn refine_gradient_corners(corners: [Point; 4], image: &GrayImage) -> [Point; 4] {
    let lines = std::array::from_fn::<_, 4, _>(|index| {
        // Prefer the local gradient fit, falling back to the observed edge when no fit is available.
        let start = corners[index];
        let end = corners[(index + 1) % 4];
        gradient_edge_line(start, end, image, 2.5).unwrap_or((
            start,
            Point {
                x: end.x - start.x,
                y: end.y - start.y,
            },
        ))
    });
    let mut refined = corners;
    for index in 0..4 {
        // Keep gradient-based intersections within four pixels of the original corners and in frame.
        let Some(point) = intersect_lines(lines[(index + 3) % 4], lines[index]) else {
            return corners;
        };
        if point.x < 0.0
            || point.y < 0.0
            || point.x > image.width() as f32 - 1.0
            || point.y > image.height() as f32 - 1.0
            || squared_distance(point, corners[index]) > 16.0
        {
            return corners;
        }
        refined[index] = point;
    }
    if point_order(refined[0], refined[2]).is_gt() {
        refined.rotate_left(2);
    }
    refined
}

/// Recovers a continuous frame around an observed rectangular printed panel.
///
/// Each panel edge must cross into a consistent observed frame tone. All four
/// enclosing edges must then cross into a persistently different surface,
/// allowing dark, white, or metallic gray frames without assuming their color.
/// Independent searches permit unequal frame widths and a wider copyright strip.
fn expand_card_border(corners: [Point; 4], image: &GrayImage) -> Option<[Point; 4]> {
    let lengths = edge_lengths(&corners);
    let maximum_offset = lengths.iter().copied().fold(f32::INFINITY, f32::min) * 0.15;
    let mut lines = [(corners[0], corners[0]); 4];
    for index in 0..4 {
        // Estimate the frame from the observed band directly outside the panel.
        let start = corners[index];
        let end = corners[(index + 1) % 4];
        let normal = Point {
            x: (end.y - start.y) / lengths[index],
            y: (start.x - end.x) / lengths[index],
        };
        let initial = border_profile(start, end, normal, 0.0, image);
        if initial.upper_quartile > -12.0 && initial.lower_quartile < 12.0 {
            return None;
        }
        let mut best_offset = None;
        let mut best_contrast = 12.0;
        let mut signed_offset = None;
        let mut signed_contrast = 16.0;
        for offset in 3..=maximum_offset as u32 {
            // Prefer the transition supported across the edge, rather than one
            // bright spot in surrounding artwork or a mark inside the border.
            let profile = border_profile(start, end, normal, offset as f32, image);
            let coherent_contrast = profile.lower_quartile.max(-profile.upper_quartile);
            if profile.band_samples >= BORDER_PROFILE_SAMPLES * 3 / 4
                && coherent_contrast > signed_contrast
            {
                signed_offset = Some(offset as f32);
                signed_contrast = coherent_contrast;
            }
            let contrast = profile.absolute_lower_quartile;
            if profile.band_samples >= BORDER_PROFILE_SAMPLES * 3 / 4 && contrast > best_contrast {
                best_offset = Some(offset as f32);
                best_contrast = contrast;
            }
        }
        // Prefer a common polarity; textured surfaces can reverse contrast against gray.
        let offset = signed_offset.or(best_offset)?;
        let shifted_start = Point {
            x: start.x + offset * normal.x,
            y: start.y + offset * normal.y,
        };
        let shifted_end = Point {
            x: end.x + offset * normal.x,
            y: end.y + offset * normal.y,
        };
        lines[index] = gradient_edge_line(shifted_start, shifted_end, image, 1.5).unwrap_or((
            shifted_start,
            Point {
                x: end.x - start.x,
                y: end.y - start.y,
            },
        ));
    }
    let mut expanded = corners;
    for index in 0..4 {
        // Intersect independently supported borders while keeping every corner
        // in frame; clipped or occluded sides do not justify inventing pixels.
        let point = intersect_lines(lines[(index + 3) % 4], lines[index])?;
        if point.x < 0.0
            || point.y < 0.0
            || point.x > image.width() as f32 - 1.0
            || point.y > image.height() as f32 - 1.0
        {
            return None;
        }
        expanded[index] = point;
    }
    (plausible_quadrilateral(&expanded) && polygon_area(&expanded) <= polygon_area(&corners) * 1.5)
        .then_some(expanded)
}

/// Measures coherent contrast across a frame edge and its sustained outer surface.
fn border_profile(
    start: Point,
    end: Point,
    normal: Point,
    offset: f32,
    image: &GrayImage,
) -> BorderProfile {
    let mut contrasts = [0.0; BORDER_PROFILE_SAMPLES];
    let mut inner_tones = [0.0; BORDER_PROFILE_SAMPLES];
    let mut persistent = [false; BORDER_PROFILE_SAMPLES];
    let mut near_tones = [0.0; BORDER_PROFILE_SAMPLES];
    for (index, contrast) in contrasts.iter_mut().enumerate() {
        // Ignore corner sections and observe the surface beyond a possible reflection.
        let fraction = 0.12 + 0.76 * index as f32 / (BORDER_PROFILE_SAMPLES - 1) as f32;
        let x = start.x + fraction * (end.x - start.x);
        let y = start.y + fraction * (end.y - start.y);
        let at =
            |distance: f32| sample_luma(image, x + distance * normal.x, y + distance * normal.y);
        let inner = at(offset - 2.0);
        inner_tones[index] = inner;
        // Follow the observed entry polarity through blur or collector-text breaks.
        let entry_sign = (at(2.0) - at(-2.0)).signum();
        near_tones[index] = [2.0_f32, 4.0, 6.0]
            .into_iter()
            .map(|distance| at(distance.min(offset - 0.5)))
            // Very narrow frames can end before the first near-band probe.
            .chain((offset <= 4.0).then_some(inner))
            .max_by(|left, right| (left * entry_sign).total_cmp(&(right * entry_sign)))
            .unwrap_or(inner);
        *contrast = at(offset + 2.0) - inner;
        persistent[index] = [6.0, 8.0]
            .into_iter()
            .map(|distance| (at(offset + distance) - inner).abs())
            .sum::<f32>()
            * 0.5
            > 16.0;
    }
    let mut sorted_tones = inner_tones;
    sorted_tones.sort_unstable_by(f32::total_cmp);
    let frame_tone = sorted_tones[BORDER_PROFILE_SAMPLES / 2];
    let coherent_inner = inner_tones
        .into_iter()
        .filter(|tone| (*tone - frame_tone).abs() <= 24.0)
        .count();
    let coherent_band = near_tones
        .into_iter()
        .filter(|tone| (*tone - frame_tone).abs() <= 24.0)
        .count();
    let persistent_outer = persistent.into_iter().filter(|persists| *persists).count();
    let band_samples = coherent_inner.min(coherent_band).min(persistent_outer);
    let mut absolute_contrasts = contrasts.map(f32::abs);
    absolute_contrasts.sort_unstable_by(f32::total_cmp);
    contrasts.sort_unstable_by(f32::total_cmp);
    BorderProfile {
        absolute_lower_quartile: absolute_contrasts[BORDER_PROFILE_SAMPLES / 4],
        lower_quartile: contrasts[BORDER_PROFILE_SAMPLES / 4],
        upper_quartile: contrasts[BORDER_PROFILE_SAMPLES * 3 / 4],
        band_samples,
    }
}

/// Fits one boundary line to local subpixel gradient peaks away from its corners.
fn gradient_edge_line(
    start: Point,
    end: Point,
    image: &GrayImage,
    search_radius: f32,
) -> Option<(Point, Point)> {
    let length = squared_distance(start, end).sqrt();
    if length < 12.0 {
        return None;
    }
    let normal = Point {
        x: (start.y - end.y) / length,
        y: (end.x - start.x) / length,
    };
    let mut samples = Vec::new();
    for index in 0..48 {
        // Sample normal profiles along the edge's middle section to avoid rounded corners.
        let fraction = 0.12 + 0.76 * index as f32 / 47.0;
        let point = Point {
            x: start.x + fraction * (end.x - start.x),
            y: start.y + fraction * (end.y - start.y),
        };
        let mut greatest_gradient = 8.0;
        let mut best_offset = None;
        let maximum_step = (search_radius * 4.0).round() as i32;
        for step in -maximum_step..=maximum_step {
            // Locate the strongest nearby luminance change across this edge sample.
            let offset = step as f32 * 0.25;
            let outside = sample_luma(
                image,
                point.x + (offset - 0.5) * normal.x,
                point.y + (offset - 0.5) * normal.y,
            );
            let inside = sample_luma(
                image,
                point.x + (offset + 0.5) * normal.x,
                point.y + (offset + 0.5) * normal.y,
            );
            let gradient = (inside - outside).abs();
            if gradient > greatest_gradient {
                greatest_gradient = gradient;
                best_offset = Some(offset);
            }
        }
        if let Some(offset) = best_offset {
            samples.push((
                offset,
                Point {
                    x: point.x + offset * normal.x,
                    y: point.y + offset * normal.y,
                },
            ));
        }
    }
    if samples.len() < 12 {
        return None;
    }
    samples.sort_by(|left, right| left.0.total_cmp(&right.0));
    let median = samples[samples.len() / 2].0;
    let points: Vec<Point> = samples
        .into_iter()
        .filter_map(|(offset, point)| ((offset - median).abs() <= 0.9).then_some(point))
        .collect();
    principal_line(&points)
}

/// Bilinearly samples a nonempty grayscale image at a clamped position.
fn sample_luma(image: &GrayImage, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, image.width() as f32 - 1.0);
    let y = y.clamp(0.0, image.height() as f32 - 1.0);
    let left = x.floor() as u32;
    let top = y.floor() as u32;
    let right = (left + 1).min(image.width() - 1);
    let bottom = (top + 1).min(image.height() - 1);
    let horizontal = x - left as f32;
    let vertical = y - top as f32;
    let upper = f32::from(image.get_pixel(left, top).0[0]) * (1.0 - horizontal)
        + f32::from(image.get_pixel(right, top).0[0]) * horizontal;
    let lower = f32::from(image.get_pixel(left, bottom).0[0]) * (1.0 - horizontal)
        + f32::from(image.get_pixel(right, bottom).0[0]) * horizontal;
    upper * (1.0 - vertical) + lower * vertical
}

/// Represents an edge by its contour centroid and principal direction.
fn fit_edge(start: Point, end: Point, contour: &[Point], maximum_distance: f32) -> (Point, Point) {
    let direction = Point {
        x: end.x - start.x,
        y: end.y - start.y,
    };
    let length_squared = squared_distance(start, end);
    let length = length_squared.sqrt();
    if length < 1.0 {
        return (start, direction);
    }
    let mut selected = Vec::new();
    for &point in contour {
        let projection = ((point.x - start.x) * direction.x + (point.y - start.y) * direction.y)
            / length_squared;
        if (0.15..=0.85).contains(&projection)
            && cross(start, end, point).abs() / length <= maximum_distance
        {
            selected.push(point);
        }
    }
    principal_line(&selected).unwrap_or((start, direction))
}

/// Returns a point-cloud centroid and its principal direction when sufficiently sampled.
fn principal_line(points: &[Point]) -> Option<(Point, Point)> {
    if points.len() < 4 {
        return None;
    }
    let count = points.len() as f32;
    let mean = Point {
        x: points.iter().map(|point| point.x).sum::<f32>() / count,
        y: points.iter().map(|point| point.y).sum::<f32>() / count,
    };
    let (xx, xy, yy) = points.iter().fold((0.0, 0.0, 0.0), |(xx, xy, yy), point| {
        // Accumulate centered covariance terms for the point cloud's principal direction.
        let x = point.x - mean.x;
        let y = point.y - mean.y;
        (xx + x * x, xy + x * y, yy + y * y)
    });
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    Some((
        mean,
        Point {
            x: angle.cos(),
            y: angle.sin(),
        },
    ))
}

/// Intersects two infinite lines, returning no point for near-parallel lines.
fn intersect_lines(first: (Point, Point), second: (Point, Point)) -> Option<Point> {
    let determinant = first.1.x * second.1.y - first.1.y * second.1.x;
    if determinant.abs() < 1e-4 {
        return None;
    }
    let offset = Point {
        x: second.0.x - first.0.x,
        y: second.0.y - first.0.y,
    };
    let fraction = (offset.x * second.1.y - offset.y * second.1.x) / determinant;
    Some(Point {
        x: first.0.x + fraction * first.1.x,
        y: first.0.y + fraction * first.1.y,
    })
}

/// Returns the axis-aligned bounds of a nonempty point sequence.
fn bounds(points: &[Point]) -> (Point, Point) {
    points.iter().fold(
        (
            Point {
                x: f32::INFINITY,
                y: f32::INFINITY,
            },
            Point {
                x: f32::NEG_INFINITY,
                y: f32::NEG_INFINITY,
            },
        ),
        |(mut minimum, mut maximum), point| {
            // Expand both bounds to include this point before processing the next one.
            minimum.x = minimum.x.min(point.x);
            minimum.y = minimum.y.min(point.y);
            maximum.x = maximum.x.max(point.x);
            maximum.y = maximum.y.max(point.y);
            (minimum, maximum)
        },
    )
}

/// Builds a clockwise convex hull in image coordinates using monotone chains.
fn convex_hull(mut points: Vec<Point>) -> Vec<Point> {
    points.sort_by(|left, right| {
        left.x
            .total_cmp(&right.x)
            .then_with(|| left.y.total_cmp(&right.y))
    });
    points.dedup_by(|left, right| left.x == right.x && left.y == right.y);
    if points.len() < 3 {
        return points;
    }
    let mut lower = Vec::new();
    for &point in &points {
        append_hull_point(&mut lower, point);
    }
    let mut upper = Vec::new();
    for &point in points.iter().rev() {
        append_hull_point(&mut upper, point);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Removes collinear or inward-facing points before extending a hull chain.
fn append_hull_point(chain: &mut Vec<Point>, point: Point) {
    while chain.len() >= 2 && cross(chain[chain.len() - 2], chain[chain.len() - 1], point) <= 0.0 {
        chain.pop();
    }
    chain.push(point);
}

/// Simplifies both arcs of a closed hull with Douglas–Peucker distances.
fn simplify_closed(points: &[Point], epsilon: f32) -> Vec<Point> {
    if points.len() <= 4 {
        return points.to_vec();
    }
    let farthest = (1..points.len())
        .max_by(|&left, &right| {
            squared_distance(points[0], points[left])
                .total_cmp(&squared_distance(points[0], points[right]))
        })
        .unwrap_or(1);
    let mut first = simplify_open(&points[..=farthest], epsilon);
    let mut second_arc = points[farthest..].to_vec();
    second_arc.push(points[0]);
    let second = simplify_open(&second_arc, epsilon);
    first.pop();
    first.extend(
        second
            .into_iter()
            .take_while(|point| point.x != points[0].x || point.y != points[0].y),
    );
    first
}

/// Simplifies an open polyline without recursion or unbounded call-stack depth.
fn simplify_open(points: &[Point], epsilon: f32) -> Vec<Point> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut retained = vec![false; points.len()];
    retained[0] = true;
    retained[points.len() - 1] = true;
    let mut segments = vec![(0, points.len() - 1)];
    while let Some((first, last)) = segments.pop() {
        // Split a segment only when its farthest interior point exceeds the tolerance.
        if last <= first + 1 {
            continue;
        }
        let length = squared_distance(points[first], points[last]).sqrt();
        let (index, distance) = ((first + 1)..last)
            .map(|index| {
                let distance = if length > 0.0 {
                    cross(points[first], points[last], points[index]).abs() / length
                } else {
                    squared_distance(points[first], points[index]).sqrt()
                };
                (index, distance)
            })
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .unwrap_or((first, 0.0));
        if distance > epsilon {
            // Preserve the deviation point and queue both halves for further simplification.
            retained[index] = true;
            segments.push((first, index));
            segments.push((index, last));
        }
    }
    points
        .iter()
        .zip(retained)
        .filter_map(|(&point, keep)| keep.then_some(point))
        .collect()
}

/// Orders a quadrilateral clockwise, starting at the upper short-side endpoint.
fn order_corners(mut corners: [Point; 4]) -> [Point; 4] {
    if signed_area(&corners) < 0.0 {
        corners.reverse();
    }
    let side_lengths = edge_lengths(&corners);
    let first = usize::from(side_lengths[0] + side_lengths[2] > side_lengths[1] + side_lengths[3]);
    let opposite = (first + 2) % 4;
    let start = if point_order(corners[first], corners[opposite]).is_gt() {
        opposite
    } else {
        first
    };
    std::array::from_fn(|offset| corners[(start + offset) % 4])
}

/// Checks convexity and conservative limits on corner angles and opposite sides.
fn plausible_quadrilateral(corners: &[Point; 4]) -> bool {
    let lengths = edge_lengths(corners);
    if lengths.iter().any(|&length| length < 4.0) {
        return false;
    }
    for index in 0..4 {
        // Require clockwise convex turns and reject excessively acute or obtuse corners.
        let previous = corners[(index + 3) % 4];
        let current = corners[index];
        let next = corners[(index + 1) % 4];
        if cross(previous, current, next) <= 0.0 {
            return false;
        }
        let dot = (previous.x - current.x) * (next.x - current.x)
            + (previous.y - current.y) * (next.y - current.y);
        if (dot / (lengths[(index + 3) % 4] * lengths[index])).abs() > 0.8 {
            return false;
        }
    }
    if lengths[0].max(lengths[2]) > lengths[0].min(lengths[2]) * 2.0
        || lengths[1].max(lengths[3]) > lengths[1].min(lengths[3]) * 2.0
    {
        return false;
    }
    true
}

/// Accepts card proportions and places the physical short side first when observable.
fn orient_card_corners(
    corners: [Point; 4],
    tolerance: f32,
    width: u32,
    height: u32,
) -> Option<[Point; 4]> {
    if !plausible_quadrilateral(&corners) {
        return None;
    }
    let corners = order_corners(corners);
    let assignments = projective_card_assignments(&corners, tolerance, width, height);
    let rotate = match assignments {
        [None, Some(_)] => true,
        [Some(first), Some(second)] => second + 0.02 < first,
        _ => false,
    };
    if rotate {
        // The physically short edge can appear longer after foreshortening.
        let first = if point_order(corners[1], corners[3]).is_gt() {
            3
        } else {
            1
        };
        return Some(std::array::from_fn(|index| corners[(first + index) % 4]));
    }
    if assignments[0].is_some() {
        return Some(corners);
    }
    let lengths = edge_lengths(&corners);
    let ratio = (lengths[0] + lengths[2]) / (lengths[1] + lengths[3]);
    ((ratio / TRADING_CARD_ASPECT_RATIO - 1.0).abs() <= tolerance).then_some(corners)
}

/// Scores both physical edge assignments under a bounded ordinary-pinhole model.
///
/// The homography's calibrated columns must be orthogonal and have the known
/// physical length ratio. This is Zhang's planar calibration constraint, with
/// square pixels, zero skew, and the principal point at the image center.
/// Focal lengths from half to twice the image diagonal avoid accepting an
/// arbitrary projective quad by allowing unbounded camera parameters. A single
/// nearly affine view cannot determine focal length, so it uses apparent ratio.
///
/// See <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf>,
/// section 2.3, equations 3 and 4. Unknown intrinsics can still leave a one-axis
/// view ambiguous; close scores retain apparent orientation rather than guessing.
fn projective_card_assignments(
    corners: &[Point; 4],
    tolerance: f32,
    width: u32,
    height: u32,
) -> [Option<f64>; 2] {
    let Some([first, second]) = centered_homography_columns(corners, width, height) else {
        return [None; 2];
    };
    if first[2].abs().max(second[2].abs()) < 0.02 {
        return [None; 2];
    }
    let ratio_tolerance = f64::from(tolerance).min(0.08);
    let expected_ratio = f64::from(TRADING_CARD_ASPECT_RATIO);
    let mut assignments: [Option<f64>; 2] = [None; 2];
    for step in 0..=64 {
        // Test both physical edge assignments against calibrated columns at this focal length.
        let focal = 0.5 * 4.0_f64.powf(f64::from(step) / 64.0);
        let first_norm =
            ((first[0].powi(2) + first[1].powi(2)) / focal.powi(2) + first[2].powi(2)).sqrt();
        let second_norm =
            ((second[0].powi(2) + second[1].powi(2)) / focal.powi(2) + second[2].powi(2)).sqrt();
        let dot =
            (first[0] * second[0] + first[1] * second[1]) / focal.powi(2) + first[2] * second[2];
        let cosine = dot.abs() / (first_norm * second_norm);
        if cosine > 0.10 {
            continue;
        }
        let ratio = first_norm / second_norm;
        for (index, ratio) in [ratio, 1.0 / ratio].into_iter().enumerate() {
            let ratio_error = (ratio / expected_ratio - 1.0).abs();
            if ratio_error <= ratio_tolerance {
                let score = ratio_error + cosine;
                assignments[index] = Some(assignments[index].map_or(score, |old| old.min(score)));
            }
        }
    }
    assignments
}

/// Derives the first two unit-square homography columns in centered image units.
fn centered_homography_columns(
    corners: &[Point; 4],
    width: u32,
    height: u32,
) -> Option<[[f64; 3]; 2]> {
    let points = corners.map(|point| [f64::from(point.x), f64::from(point.y)]);
    let dx_first = points[1][0] - points[2][0];
    let dx_second = points[3][0] - points[2][0];
    let dy_first = points[1][1] - points[2][1];
    let dy_second = points[3][1] - points[2][1];
    let dx_diagonal = points[0][0] - points[1][0] + points[2][0] - points[3][0];
    let dy_diagonal = points[0][1] - points[1][1] + points[2][1] - points[3][1];
    let determinant = dx_first * dy_second - dx_second * dy_first;
    if determinant.abs() < 1e-8 {
        return None;
    }
    let projective_first = (dx_diagonal * dy_second - dx_second * dy_diagonal) / determinant;
    let projective_second = (dx_first * dy_diagonal - dx_diagonal * dy_first) / determinant;
    let center_x = (f64::from(width) - 1.0) * 0.5;
    let center_y = (f64::from(height) - 1.0) * 0.5;
    let diagonal = f64::from(width).hypot(f64::from(height));
    Some([
        [
            (points[1][0] - points[0][0] + projective_first * (points[1][0] - center_x)) / diagonal,
            (points[1][1] - points[0][1] + projective_first * (points[1][1] - center_y)) / diagonal,
            projective_first,
        ],
        [
            (points[3][0] - points[0][0] + projective_second * (points[3][0] - center_x))
                / diagonal,
            (points[3][1] - points[0][1] + projective_second * (points[3][1] - center_y))
                / diagonal,
            projective_second,
        ],
    ])
}

/// Returns the four side lengths in corner order.
fn edge_lengths(corners: &[Point; 4]) -> [f32; 4] {
    std::array::from_fn(|index| squared_distance(corners[index], corners[(index + 1) % 4]).sqrt())
}

/// Returns intersection-over-union and overlap as a fraction of the smaller quad.
fn quadrilateral_overlap(left: &CardDetection, right: &CardDetection) -> (f32, f32) {
    let mut polygon = left.corners.to_vec();
    for index in 0..4 {
        // Intersect the left outline with each interior half-plane of the right outline.
        let start = right.corners[index];
        let end = right.corners[(index + 1) % 4];
        polygon = clip_polygon(&polygon, start, end);
        if polygon.len() < 3 {
            return (0.0, 0.0);
        }
    }
    let intersection = polygon_area(&polygon);
    let union = left.area + right.area - intersection;
    let iou = if union > 0.0 {
        (intersection / union).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let smaller_area = left.area.min(right.area);
    let containment = if smaller_area > 0.0 {
        (intersection / smaller_area).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (iou, containment)
}

/// Clips a clockwise polygon to the interior half-plane of a directed edge.
fn clip_polygon(polygon: &[Point], start: Point, end: Point) -> Vec<Point> {
    let Some(&last) = polygon.last() else {
        return Vec::new();
    };
    let mut output = Vec::new();
    let mut previous = last;
    let mut previous_distance = cross(start, end, previous);
    for &current in polygon {
        // Emit edge crossings before retaining vertices inside the clipping half-plane.
        let current_distance = cross(start, end, current);
        if (previous_distance >= 0.0) != (current_distance >= 0.0) {
            let fraction = previous_distance / (previous_distance - current_distance);
            output.push(Point {
                x: previous.x + fraction * (current.x - previous.x),
                y: previous.y + fraction * (current.y - previous.y),
            });
        }
        if current_distance >= 0.0 {
            output.push(current);
        }
        previous = current;
        previous_distance = current_distance;
    }
    output
}

/// Orders detections by their topmost extent, then leftmost extent.
fn spatial_order(left: &CardDetection, right: &CardDetection) -> std::cmp::Ordering {
    let (left_minimum, _) = bounds(&left.corners);
    let (right_minimum, _) = bounds(&right.corners);
    point_order(left_minimum, right_minimum)
        .then_with(|| point_order(left.corners[0], right.corners[0]))
}

/// Orders image points from top to bottom and then left to right.
fn point_order(left: Point, right: Point) -> std::cmp::Ordering {
    left.y
        .total_cmp(&right.y)
        .then_with(|| left.x.total_cmp(&right.x))
}

/// Computes the signed cross product of two vectors with a shared origin.
fn cross(origin: Point, first: Point, second: Point) -> f32 {
    (first.x - origin.x) * (second.y - origin.y) - (first.y - origin.y) * (second.x - origin.x)
}

/// Computes squared Euclidean distance between image points.
fn squared_distance(first: Point, second: Point) -> f32 {
    (first.x - second.x).powi(2) + (first.y - second.y).powi(2)
}

/// Computes a polygon's signed area, positive for clockwise image coordinates.
fn signed_area(points: &[Point]) -> f32 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(first, second)| first.x * second.y - second.x * first.y)
        .sum::<f32>()
        * 0.5
}

/// Computes a polygon's unsigned area, including an empty polygon's zero area.
fn polygon_area(points: &[Point]) -> f32 {
    signed_area(points).abs()
}

/// Sums the distances around a closed polygon.
fn polygon_perimeter(points: &[Point]) -> f32 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(&first, &second)| squared_distance(first, second).sqrt())
        .sum()
}

/// Bilinearly samples RGB channels in the working detection image.
fn sample_colour(image: &image::RgbImage, x: f32, y: f32) -> [f32; 3] {
    let x = x.clamp(0.0, image.width() as f32 - 1.0);
    let y = y.clamp(0.0, image.height() as f32 - 1.0);
    let left = x.floor() as u32;
    let top = y.floor() as u32;
    let right = (left + 1).min(image.width() - 1);
    let bottom = (top + 1).min(image.height() - 1);
    let horizontal = x - left as f32;
    let vertical = y - top as f32;
    std::array::from_fn(|channel| {
        let upper = f32::from(image.get_pixel(left, top)[channel]) * (1.0 - horizontal)
            + f32::from(image.get_pixel(right, top)[channel]) * horizontal;
        let lower = f32::from(image.get_pixel(left, bottom)[channel]) * (1.0 - horizontal)
            + f32::from(image.get_pixel(right, bottom)[channel]) * horizontal;
        upper * (1.0 - vertical) + lower * vertical
    })
}

/// Measures squared color contrast without assuming a shared luminance polarity.
fn colour_distance_squared(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

/// Recovers a changing-color physical perimeter inside an observed reflective sleeve.
///
/// This fallback requires the same narrow reflective rim as frame recovery and
/// matching clear-band colors on both sides. Every physical edge must be the
/// first sustained transition away from that clear band; artwork need not share
/// a luminance, color, or gradient polarity along the card perimeter.
fn recover_inner_artwork(
    corners: [Point; 4],
    gray: &GrayImage,
    source: &DynamicImage,
    config: &ExtractorConfig,
) -> Option<[Point; 4]> {
    let lengths = edge_lengths(&corners);
    let short_side = lengths.into_iter().fold(f32::INFINITY, f32::min);
    let search_limit = short_side * 0.15;
    // Complete the inexpensive reflective-rim proof before converting source colors.
    for index in 0..4 {
        let start = corners[index];
        let end = corners[(index + 1) % 4];
        let normal = Point {
            x: (start.y - end.y) / lengths[index],
            y: (end.x - start.x) / lengths[index],
        };
        let at = |fraction: f32, offset: f32| {
            sample_luma(
                gray,
                start.x + (end.x - start.x) * fraction + normal.x * offset,
                start.y + (end.y - start.y) * fraction + normal.y * offset,
            )
        };
        let support = (0..40)
            .filter(|&sample| {
                let fraction = 0.15 + 0.7 * sample as f32 / 39.0;
                let outside = (at(fraction, -4.0) + at(fraction, -6.0)) * 0.5;
                let inside = (at(fraction, 4.0) + at(fraction, 6.0)) * 0.5;
                let peak = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
                    .into_iter()
                    .map(|offset| at(fraction, offset))
                    .fold(0.0, f32::max);
                inside > 55.0
                    && (outside - inside).abs() < 30.0
                    && peak > inside + 30.0
                    && peak > outside + 30.0
            })
            .count();
        if support < 30 {
            return None;
        }
    }
    let colour_image = source.to_rgb8();
    let colour_image = if colour_image.dimensions() == gray.dimensions() {
        colour_image
    } else {
        imageops::resize(
            &colour_image,
            gray.width(),
            gray.height(),
            imageops::FilterType::Triangle,
        )
    };
    let mut lines = Vec::with_capacity(4);
    for index in 0..4 {
        let start = corners[index];
        let end = corners[(index + 1) % 4];
        let normal = Point {
            x: (start.y - end.y) / lengths[index],
            y: (end.x - start.x) / lengths[index],
        };
        let at = |fraction: f32, offset: f32| {
            sample_colour(
                &colour_image,
                start.x + (end.x - start.x) * fraction + normal.x * offset,
                start.y + (end.y - start.y) * fraction + normal.y * offset,
            )
        };
        let mut clear_band = [[0.0; 3]; 40];
        let mut clear_support = 0;
        for (sample, clear) in clear_band.iter_mut().enumerate() {
            let fraction = 0.15 + 0.7 * sample as f32 / 39.0;
            let inside =
                std::array::from_fn(|c| (at(fraction, 4.0)[c] + at(fraction, 6.0)[c]) * 0.5);
            let outside =
                std::array::from_fn(|c| (at(fraction, -4.0)[c] + at(fraction, -6.0)[c]) * 0.5);
            *clear = inside;
            clear_support +=
                usize::from(colour_distance_squared(inside, outside) < 30.0_f32.powi(2));
        }
        if clear_support < 30 {
            return None;
        }
        let mut selected = None;
        for step in 8..=(search_limit * 2.0) as i32 {
            let offset = step as f32 * 0.5;
            let support = (0..40)
                .filter(|&sample| {
                    let fraction = 0.15 + 0.7 * sample as f32 / 39.0;
                    let before = at(fraction, offset - 1.0);
                    let after = at(fraction, offset + 1.0);
                    let clear = clear_band[sample];
                    let persistent = [2.0, 3.0, 4.0]
                        .into_iter()
                        .map(|distance| {
                            colour_distance_squared(at(fraction, offset + distance), clear)
                        })
                        .sum::<f32>()
                        / 3.0;
                    colour_distance_squared(before, clear) < 25.0_f32.powi(2)
                        && colour_distance_squared(after, before) > 25.0_f32.powi(2)
                        && persistent > 30.0_f32.powi(2)
                        && [2.0, 3.0, 4.0].into_iter().all(|distance| {
                            colour_distance_squared(at(fraction, offset + distance), clear)
                                > 20.0_f32.powi(2)
                        })
                })
                .count();
            if support >= 30 {
                selected = Some(offset);
                break;
            }
        }
        let offset = selected?;
        let mut points = Vec::new();
        for sample in 0..48 {
            let fraction = 0.15 + 0.7 * sample as f32 / 47.0;
            let clear =
                std::array::from_fn(|c| (at(fraction, 4.0)[c] + at(fraction, 6.0)[c]) * 0.5);
            let mut greatest = 20.0_f32.powi(2);
            let mut found = None;
            for step in -6..=6 {
                let trial = offset + step as f32 * 0.25;
                let gradient =
                    colour_distance_squared(at(fraction, trial - 0.5), at(fraction, trial + 0.5));
                if gradient > greatest
                    && colour_distance_squared(at(fraction, trial - 1.0), clear) < 25.0_f32.powi(2)
                    && colour_distance_squared(at(fraction, trial + 1.0), clear) > 25.0_f32.powi(2)
                {
                    greatest = gradient;
                    found = Some(trial);
                }
            }
            if let Some(offset) = found {
                points.push(Point {
                    x: start.x + (end.x - start.x) * fraction + normal.x * offset,
                    y: start.y + (end.y - start.y) * fraction + normal.y * offset,
                });
            }
        }
        if points.len() < 32 {
            return None;
        }
        lines.push(principal_line(&points)?);
    }
    let mut refined = corners;
    for index in 0..4 {
        let point = intersect_lines(lines[(index + 3) % 4], lines[index])?;
        if squared_distance(point, corners[index]) > (short_side * 0.20).powi(2)
            || point.x < 0.0
            || point.y < 0.0
            || point.x > gray.width() as f32 - 1.0
            || point.y > gray.height() as f32 - 1.0
            || (0..4).any(|side| cross(corners[side], corners[(side + 1) % 4], point) < 0.0)
        {
            return None;
        }
        refined[index] = point;
    }
    let refined = orient_card_corners(
        refined,
        config.aspect_ratio_tolerance,
        gray.width(),
        gray.height(),
    )?;
    let area = polygon_area(&refined);
    if area < polygon_area(&corners) * 0.60
        || area >= polygon_area(&corners) * 0.95
        || area < config.min_area_fraction * gray.width() as f32 * gray.height() as f32
    {
        return None;
    }
    // Empty seams cannot justify replacing a rim with an unprinted uniform interior.
    let mut interior = [0.0; 63];
    for (index, tone) in interior.iter_mut().enumerate() {
        let x = 0.15 + 0.7 * (index % 7) as f32 / 6.0;
        let y = 0.15 + 0.7 * (index / 7) as f32 / 8.0;
        let point = Point {
            x: (1.0 - y) * ((1.0 - x) * refined[0].x + x * refined[1].x)
                + y * ((1.0 - x) * refined[3].x + x * refined[2].x),
            y: (1.0 - y) * ((1.0 - x) * refined[0].y + x * refined[1].y)
                + y * ((1.0 - x) * refined[3].y + x * refined[2].y),
        };
        *tone = sample_luma(gray, point.x, point.y);
    }
    interior.sort_by(f32::total_cmp);
    (interior[47] - interior[15] >= 15.0).then_some(refined)
}

/// Replaces a reflective sleeve rim with a separately observed physical card frame.
///
/// The frame must sustain a coherent tone distinct from the sleeve's clear band.
/// Either contrast polarity is allowed, so black, white, and gray borders share
/// the same evidence requirements. Printed details must remain inside the frame.
fn recover_inner_card_border(
    corners: [Point; 4],
    image: &GrayImage,
    config: &ExtractorConfig,
) -> Option<[Point; 4]> {
    let short_side = edge_lengths(&corners)
        .into_iter()
        .fold(f32::INFINITY, f32::min);
    let search_limit = short_side * 0.15;
    let mut lines = Vec::with_capacity(4);
    let mut frame_tones = Vec::with_capacity(4);
    for index in 0..4 {
        let start = corners[index];
        let end = corners[(index + 1) % 4];
        let length = squared_distance(start, end).sqrt();
        let normal = Point {
            x: (start.y - end.y) / length,
            y: (end.x - start.x) / length,
        };
        let at = |fraction: f32, offset: f32| {
            sample_luma(
                image,
                start.x + (end.x - start.x) * fraction + normal.x * offset,
                start.y + (end.y - start.y) * fraction + normal.y * offset,
            )
        };
        // A narrow bright line must return to similar background tones on both sides.
        let mut rim_support = 0;
        let mut clear_band = [0.0; 40];
        for (sample, tone) in clear_band.iter_mut().enumerate() {
            let fraction = 0.15 + 0.7 * sample as f32 / 39.0;
            let outside = (at(fraction, -4.0) + at(fraction, -6.0)) * 0.5;
            let inside = (at(fraction, 4.0) + at(fraction, 6.0)) * 0.5;
            *tone = inside;
            let peak = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
                .into_iter()
                .map(|offset| at(fraction, offset))
                .fold(0.0, f32::max);
            if inside > 55.0
                && (outside - inside).abs() < 30.0
                && peak > inside + 30.0
                && peak > outside + 30.0
            {
                rim_support += 1;
            }
        }
        if rim_support < 30 {
            return None;
        }
        // Select the first sustained frame transition, rather than stronger nested artwork.
        let mut selected = None;
        for step in 8..=(search_limit * 2.0) as i32 {
            let offset = step as f32 * 0.5;
            let mut tones = [0.0; 40];
            for (sample, tone) in tones.iter_mut().enumerate() {
                let fraction = 0.15 + 0.7 * sample as f32 / 39.0;
                *tone = (at(fraction, offset + 2.0) + at(fraction, offset + 4.0)) * 0.5;
            }
            let mut ordered = tones;
            ordered.sort_by(f32::total_cmp);
            if ordered[30] - ordered[10] > 20.0 {
                continue;
            }
            let frame_tone = ordered[20];
            let mut support = [0; 2];
            for sample in 0..40 {
                let fraction = 0.15 + 0.7 * sample as f32 / 39.0;
                let before = at(fraction, offset - 1.0);
                let after = at(fraction, offset + 1.0);
                let near = at(fraction, offset + 2.0);
                let middle = at(fraction, offset + 3.0);
                let far = at(fraction, offset + 4.0);
                for (polarity, sign) in [-1.0, 1.0].into_iter().enumerate() {
                    if (before - clear_band[sample]).abs() < 20.0
                        && sign * (after - before) > 25.0
                        && sign * (tones[sample] - clear_band[sample]) > 25.0
                        && (near - middle).abs() < 20.0
                        && (middle - far).abs() < 20.0
                        && (near - far).abs() < 20.0
                        && (tones[sample] - frame_tone).abs() < 20.0
                    {
                        support[polarity] += 1;
                    }
                }
            }
            if let Some(polarity) = support.into_iter().position(|count| count >= 30) {
                selected = Some((offset, [-1.0, 1.0][polarity], frame_tone));
                break;
            }
        }
        let (offset, sign, frame_tone) = selected?;
        frame_tones.push(frame_tone);
        if frame_tones
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            - frame_tones.iter().copied().fold(f32::INFINITY, f32::min)
            > 20.0
        {
            return None;
        }
        // Fit observed subpixel gradients near that first coherent physical boundary.
        let mut points = Vec::new();
        for sample in 0..48 {
            let fraction = 0.15 + 0.7 * sample as f32 / 47.0;
            let mut best_gradient = 25.0;
            let mut best_offset = None;
            for step in -6..=6 {
                let trial = offset + step as f32 * 0.25;
                let gradient = sign * (at(fraction, trial + 0.5) - at(fraction, trial - 0.5));
                if gradient > best_gradient
                    && (at(fraction, trial + 1.0) - frame_tone).abs() < 20.0
                    && (at(fraction, trial - 1.0) - (at(fraction, 4.0) + at(fraction, 6.0)) * 0.5)
                        .abs()
                        < 20.0
                {
                    best_gradient = gradient;
                    best_offset = Some(trial);
                }
            }
            if let Some(found) = best_offset {
                points.push(Point {
                    x: start.x + (end.x - start.x) * fraction + normal.x * found,
                    y: start.y + (end.y - start.y) * fraction + normal.y * found,
                });
            }
        }
        if points.len() < 32 {
            return None;
        }
        lines.push(principal_line(&points)?);
    }
    let mut refined = corners;
    for index in 0..4 {
        let point = intersect_lines(lines[(index + 3) % 4], lines[index])?;
        if squared_distance(point, corners[index]) > (short_side * 0.20).powi(2)
            || point.x < 0.0
            || point.y < 0.0
            || point.x > image.width() as f32 - 1.0
            || point.y > image.height() as f32 - 1.0
            || (0..4).any(|side| cross(corners[side], corners[(side + 1) % 4], point) < 0.0)
        {
            return None;
        }
        refined[index] = point;
    }
    let refined = orient_card_corners(
        refined,
        config.aspect_ratio_tolerance,
        image.width(),
        image.height(),
    )?;
    let area = polygon_area(&refined);
    if area < polygon_area(&corners) * 0.60
        || area >= polygon_area(&corners) * 0.95
        || area < config.min_area_fraction * image.width() as f32 * image.height() as f32
    {
        return None;
    }
    // Interior variation distinguishes printed card content from another empty plastic seam.
    let mut interior = [0.0; 63];
    for (index, tone) in interior.iter_mut().enumerate() {
        let x = 0.15 + 0.7 * (index % 7) as f32 / 6.0;
        let y = 0.15 + 0.7 * (index / 7) as f32 / 8.0;
        let point = Point {
            x: (1.0 - y) * ((1.0 - x) * refined[0].x + x * refined[1].x)
                + y * ((1.0 - x) * refined[3].x + x * refined[2].x),
            y: (1.0 - y) * ((1.0 - x) * refined[0].y + x * refined[1].y)
                + y * ((1.0 - x) * refined[3].y + x * refined[2].y),
        };
        *tone = sample_luma(image, point.x, point.y);
    }
    interior.sort_by(f32::total_cmp);
    (interior[47] - interior[15] >= 15.0).then_some(refined)
}

#[cfg(test)]
/// Regression tests for contour recovery and duplicate suppression.
mod tests {
    use image::Luma;

    use super::*;

    /// Renders complete quadrilaterals against a uniform contrasting surface.
    fn render(width: u32, height: u32, cards: &[[Point; 4]], dark: bool) -> DynamicImage {
        let background = if dark { 240 } else { 20 };
        let foreground = if dark { 20 } else { 240 };
        let image = GrayImage::from_fn(width, height, |x, y| {
            let point = Point {
                x: x as f32,
                y: y as f32,
            };
            let inside = cards.iter().any(|corners| {
                (0..4).all(|index| cross(corners[index], corners[(index + 1) % 4], point) >= 0.0)
            });
            Luma([if inside { foreground } else { background }])
        });
        DynamicImage::ImageLuma8(image)
    }

    /// Produces a trading card rectangle rotated 45 degrees around its center.
    fn rotated_card(center_x: f32, center_y: f32) -> [Point; 4] {
        [(-50.0, -70.0), (50.0, -70.0), (50.0, 70.0), (-50.0, 70.0)].map(|(x, y)| Point {
            x: center_x + (x - y) * std::f32::consts::FRAC_1_SQRT_2,
            y: center_y + (x + y) * std::f32::consts::FRAC_1_SQRT_2,
        })
    }

    #[test]
    /// A uniform image with card proportions still has no observable card.
    fn uniform_card_proportions_produce_no_detection() {
        let image = DynamicImage::ImageLuma8(GrayImage::from_pixel(250, 350, Luma([180])));
        assert!(detect(&image, &ExtractorConfig::default()).is_empty());
    }

    #[test]
    /// An observed inner panel recovers independently supported unequal borders.
    fn recovers_dark_border_with_wider_bottom_strip() {
        let image = GrayImage::from_fn(360, 400, |x, y| {
            Luma([if (88..=252).contains(&x) && (68..=292).contains(&y) {
                220
            } else if (80..=260).contains(&x) && (60..=312).contains(&y) {
                14
            } else {
                (70 + (x * 17 + y * 31) % 140) as u8
            }])
        });
        let panel = [
            Point { x: 88.0, y: 68.0 },
            Point { x: 252.0, y: 68.0 },
            Point { x: 252.0, y: 292.0 },
            Point { x: 88.0, y: 292.0 },
        ];
        let expanded = expand_card_border(panel, &image).expect("four observed outer edges");
        for (actual, expected) in expanded.into_iter().zip([
            Point { x: 79.5, y: 59.5 },
            Point { x: 260.5, y: 59.5 },
            Point { x: 260.5, y: 312.5 },
            Point { x: 79.5, y: 312.5 },
        ]) {
            assert!(squared_distance(actual, expected).sqrt() < 1.0);
        }
    }

    #[test]
    /// A dark band without an observed enclosing edge cannot justify padding.
    fn border_recovery_requires_all_four_outer_edges() {
        let panel = [
            Point { x: 88.0, y: 68.0 },
            Point { x: 252.0, y: 68.0 },
            Point { x: 252.0, y: 292.0 },
            Point { x: 88.0, y: 292.0 },
        ];
        for missing_edge in 0..4 {
            let image = GrayImage::from_fn(360, 400, |x, y| {
                let extended_band = match missing_edge {
                    0 => y < 60 && (80..=260).contains(&x),
                    1 => x > 260 && (60..=312).contains(&y),
                    2 => y > 312 && (80..=260).contains(&x),
                    _ => x < 80 && (60..=312).contains(&y),
                };
                Luma([if (88..=252).contains(&x) && (68..=292).contains(&y) {
                    220
                } else if extended_band || ((80..=260).contains(&x) && (60..=312).contains(&y)) {
                    14
                } else {
                    160
                }])
            });
            assert!(
                expand_card_border(panel, &image).is_none(),
                "edge {missing_edge} has no enclosing boundary"
            );
        }
    }

    #[test]
    /// Rotated neighbors survive even when their axis-aligned bounds overlap.
    fn rotated_neighbors_are_not_suppressed() {
        let cards = [rotated_card(140.0, 140.0), rotated_card(220.0, 220.0)];
        let image = render(370, 370, &cards, false);
        let detections = detect(&image, &ExtractorConfig::default());
        assert_eq!(detections.len(), 2);
        let (iou, containment) = quadrilateral_overlap(&detections[0], &detections[1]);
        assert!(iou < 0.01);
        assert!(containment < 0.01);
    }

    #[test]
    /// Landscape dark cards retain portrait corner ordering after downsampling.
    fn dark_landscape_coordinates_reference_source_pixels() {
        let corners = [
            Point { x: 120.0, y: 90.0 },
            Point { x: 680.0, y: 90.0 },
            Point { x: 680.0, y: 490.0 },
            Point { x: 120.0, y: 490.0 },
        ];
        let image = render(900, 650, &[corners], true);
        let config = ExtractorConfig {
            max_detection_dimension: 300,
            ..ExtractorConfig::default()
        };
        let detections = detect(&image, &config);
        assert_eq!(detections.len(), 1);
        let detection = &detections[0];
        let (minimum, maximum) = bounds(&detection.corners);
        assert!((minimum.x - 120.0).abs() < 6.0);
        assert!((minimum.y - 90.0).abs() < 6.0);
        assert!((maximum.x - 680.0).abs() < 6.0);
        assert!((maximum.y - 490.0).abs() < 6.0);
        assert!(signed_area(&detection.corners) > 0.0);
        let lengths = edge_lengths(&detection.corners);
        assert!(lengths[0] < lengths[1]);
    }

    #[test]
    /// A moderately foreshortened quadrilateral remains a plausible card.
    fn accepts_moderate_perspective() {
        let corners = [
            Point { x: 100.0, y: 70.0 },
            Point { x: 255.0, y: 95.0 },
            Point { x: 290.0, y: 330.0 },
            Point { x: 85.0, y: 350.0 },
        ];
        let image = render(420, 430, &[corners], false);
        assert_eq!(detect(&image, &ExtractorConfig::default()).len(), 1);
    }

    #[test]
    /// Missing most of one side cannot be repaired into a card by its hull.
    fn rejects_deeply_notched_silhouette() {
        let image = GrayImage::from_fn(340, 400, |x, y| {
            let in_rectangle = (60..=240).contains(&x) && (60..=312).contains(&y);
            let in_notch = x >= 110 && (125..=245).contains(&y);
            Luma([if in_rectangle && !in_notch { 240 } else { 20 }])
        });
        let detections = detect(
            &DynamicImage::ImageLuma8(image),
            &ExtractorConfig::default(),
        );
        assert!(
            detections.is_empty(),
            "unexpected detections: {detections:?}"
        );
    }

    #[test]
    /// A triangular object with card-shaped bounds has no four-sided outline.
    fn rejects_triangular_silhouette() {
        let corners = [
            Point { x: 60.0, y: 60.0 },
            Point { x: 240.0, y: 60.0 },
            Point { x: 150.0, y: 312.0 },
        ];
        let image = GrayImage::from_fn(340, 400, |x, y| {
            let point = Point {
                x: x as f32,
                y: y as f32,
            };
            let inside =
                (0..3).all(|index| cross(corners[index], corners[(index + 1) % 3], point) >= 0.0);
            Luma([if inside { 240 } else { 20 }])
        });
        let detections = detect(
            &DynamicImage::ImageLuma8(image),
            &ExtractorConfig::default(),
        );
        assert!(
            detections.is_empty(),
            "unexpected detections: {detections:?}"
        );
    }

    #[test]
    /// Polygon clipping agrees with an analytically known partial overlap.
    fn quadrilateral_iou_matches_known_overlap() {
        let first = CardDetection {
            corners: [
                Point { x: 0.0, y: 0.0 },
                Point { x: 10.0, y: 0.0 },
                Point { x: 10.0, y: 10.0 },
                Point { x: 0.0, y: 10.0 },
            ],
            area: 100.0,
        };
        let mut second = first.clone();
        for point in &mut second.corners {
            point.x += 5.0;
        }
        let (iou, containment) = quadrilateral_overlap(&first, &second);
        let (reverse_iou, reverse_containment) = quadrilateral_overlap(&second, &first);
        assert!((iou - 1.0 / 3.0).abs() < 1e-6);
        assert!((containment - 0.5).abs() < 1e-6);
        assert!((reverse_iou - iou).abs() < 1e-6);
        assert!((reverse_containment - containment).abs() < 1e-6);
        assert!((quadrilateral_overlap(&first, &first).0 - 1.0).abs() < 1e-6);
        assert!(iou < DUPLICATE_IOU && containment < INTERIOR_CONTAINMENT);
    }

    #[test]
    /// A landscape art window inside a portrait card is an interior panel.
    fn suppresses_contained_artwork_window() {
        let image = GrayImage::from_fn(340, 400, |x, y| {
            let card = (60..=260).contains(&x) && (50..=330).contains(&y);
            let art = (75..=245).contains(&x) && (90..=210).contains(&y);
            Luma([if art {
                60
            } else if card {
                240
            } else {
                20
            }])
        });
        let detections = detect(
            &DynamicImage::ImageLuma8(image),
            &ExtractorConfig::default(),
        );
        assert_eq!(detections.len(), 1);
        assert!(detections[0].area > 55_000.0);
    }

    #[test]
    /// The dark-class split recovers a border merged with the table globally.
    fn detects_dark_rim_on_mid_dark_background() {
        let image = GrayImage::from_fn(340, 400, |x, y| {
            let card = (60..=260).contains(&x) && (50..=330).contains(&y);
            let face = (70..=250).contains(&x) && (60..=320).contains(&y);
            Luma([if face {
                200
            } else if card {
                20
            } else {
                60
            }])
        });
        let config = ExtractorConfig {
            // Isolate the threshold-mask path by exceeding possible gradients.
            canny_low_threshold: 1200.0,
            canny_high_threshold: 2400.0,
            ..ExtractorConfig::default()
        };
        let detections = detect(&DynamicImage::ImageLuma8(image), &config);
        assert_eq!(detections.len(), 1);
        assert!(detections[0].area > 55_000.0);
        let (minimum, _) = bounds(&detections[0].corners);
        assert!((minimum.x - 60.0).abs() < 1.0);
        assert!((minimum.y - 50.0).abs() < 1.0);
    }

    #[test]
    /// Visible straight edges recover a small hidden corner without filling pixels.
    fn infers_outline_behind_small_corner_occluders() {
        for circular in [false, true] {
            let image = GrayImage::from_fn(360, 400, |x, y| {
                let card = (80..=260).contains(&x) && (60..=312).contains(&y);
                let occluded = if circular {
                    (x as i32 - 80).pow(2) + (y as i32 - 60).pow(2) <= 40_i32.pow(2)
                } else {
                    (60..=128).contains(&x) && (40..=104).contains(&y)
                };
                Luma([if occluded && circular {
                    130
                } else if card && !occluded {
                    240
                } else {
                    20
                }])
            });
            let detections = detect(
                &DynamicImage::ImageLuma8(image),
                &ExtractorConfig::default(),
            );
            assert_eq!(detections.len(), 1, "circular={circular}: {detections:?}");
            let (minimum, maximum) = bounds(&detections[0].corners);
            for (actual, expected) in [
                (minimum.x, 80.0),
                (minimum.y, 60.0),
                (maximum.x, 260.0),
                (maximum.y, 312.0),
            ] {
                assert!(
                    (actual - expected).abs() < 3.0,
                    "circular={circular}: {detections:?}"
                );
            }
        }
    }

    #[test]
    /// Edge intersections retain perspective when one corner is hidden.
    fn infers_perspective_corner_from_visible_edges() {
        let corners = [
            Point { x: 100.0, y: 70.0 },
            Point { x: 255.0, y: 95.0 },
            Point { x: 290.0, y: 330.0 },
            Point { x: 85.0, y: 350.0 },
        ];
        let mut image = render(420, 430, &[corners], false).into_luma8();
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            if (x as i32 - 100).pow(2) + (y as i32 - 70).pow(2) <= 32_i32.pow(2) {
                *pixel = Luma([130]);
            }
        }
        let detections = detect(
            &DynamicImage::ImageLuma8(image),
            &ExtractorConfig::default(),
        );
        assert_eq!(detections.len(), 1, "{detections:?}");
        for corner in corners {
            let distance = detections[0]
                .corners
                .iter()
                .map(|&detected| squared_distance(detected, corner).sqrt())
                .fold(f32::INFINITY, f32::min);
            assert!(distance < 4.0, "{distance}: {detections:?}");
        }
    }

    #[test]
    /// A pinhole view preserves the physical short edge even when its image is longer.
    fn detects_oblique_cards_with_physical_portrait_orientation() {
        for (tilt, focal, distance) in [
            (20.0_f32, 700.0, 850.0),
            (35.0, 1050.0, 1400.0),
            (50.0, 800.0, 1000.0),
            (65.0, 1150.0, 1450.0),
        ] {
            for horizontal_axis in [false, true] {
                let (sin, cos) = tilt.to_radians().sin_cos();
                let corners = [
                    (-90.0, -126.0),
                    (90.0, -126.0),
                    (90.0, 126.0),
                    (-90.0, 126.0),
                ]
                .map(|(x, y)| {
                    let (x, y, depth) = if horizontal_axis {
                        (x, y * cos, distance + y * sin)
                    } else {
                        (x * cos, y, distance - x * sin)
                    };
                    Point {
                        x: 319.5 + focal * x / depth,
                        y: 259.5 + focal * y / depth,
                    }
                });
                let image = render(640, 520, &[corners], false);
                let detections = detect(&image, &ExtractorConfig::default());
                assert_eq!(
                    detections.len(),
                    1,
                    "tilt={tilt}, horizontal={horizontal_axis}"
                );
                let canonical_error = [0, 2]
                    .into_iter()
                    .map(|offset| {
                        (0..4)
                            .map(|index| {
                                squared_distance(
                                    corners[(index + offset) % 4],
                                    detections[0].corners[index],
                                )
                                .sqrt()
                            })
                            .fold(0.0, f32::max)
                    })
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    canonical_error < 4.0,
                    "tilt={tilt}, horizontal={horizontal_axis}: {canonical_error}, {:?}",
                    detections[0]
                );
            }
        }
    }
}

#[cfg(test)]
/// Checks that inward refinement follows a physical frame inside a transparent rim.
mod sleeve_tests {
    use super::*;
    use image::Luma;

    /// Renders independent sleeve, physical frame, and printed interior boundaries.
    fn sleeve_control(card: bool, missing_frame: bool) -> GrayImage {
        GrayImage::from_fn(280, 340, |x, y| {
            let in_sleeve = (40..=240).contains(&x) && (30..=310).contains(&y);
            let sleeve_edge = in_sleeve && (x == 40 || x == 240 || y == 30 || y == 310);
            let in_card = card && (50..=230).contains(&x) && (47..=299).contains(&y);
            let in_panel = (58..=222).contains(&x) && (55..=279).contains(&y);
            let pixel = if sleeve_edge {
                220
            } else if in_card {
                if missing_frame && x > 222 {
                    110
                } else if in_panel {
                    (70 + (x * 17 + y * 31) % 150) as u8
                } else {
                    14
                }
            } else if in_sleeve {
                110
            } else {
                100
            };
            Luma([pixel])
        })
    }

    /// Locates the reflective sleeve perimeter around the physical card.
    fn sleeve_corners() -> [Point; 4] {
        [
            Point { x: 39.5, y: 29.5 },
            Point { x: 240.5, y: 29.5 },
            Point { x: 240.5, y: 310.5 },
            Point { x: 39.5, y: 310.5 },
        ]
    }

    #[test]
    /// Independent observed frame lines recover unequal sleeve margins.
    fn selects_card_frame_inside_reflective_rim() {
        let image = sleeve_control(true, false);
        let refined =
            recover_inner_card_border(sleeve_corners(), &image, &ExtractorConfig::default())
                .expect("four physical frame edges");
        let expected = [
            Point { x: 49.5, y: 46.5 },
            Point { x: 230.5, y: 46.5 },
            Point { x: 230.5, y: 299.5 },
            Point { x: 49.5, y: 299.5 },
        ];
        for (actual, expected) in refined.into_iter().zip(expected) {
            assert!(squared_distance(actual, expected).sqrt() < 1.0);
        }
    }

    #[test]
    /// An empty rim or three-sided inner frame cannot justify shrinking.
    fn requires_four_inner_frame_edges() {
        for (card, missing_frame) in [(false, false), (true, true)] {
            assert!(
                recover_inner_card_border(
                    sleeve_corners(),
                    &sleeve_control(card, missing_frame),
                    &ExtractorConfig::default()
                )
                .is_none()
            );
        }
    }

    #[test]
    /// Already selected physical borders and opaque card margins remain unchanged.
    fn preserves_physical_card_border_and_printed_panel() {
        let image = sleeve_control(true, false);
        let physical = [
            Point { x: 49.5, y: 46.5 },
            Point { x: 230.5, y: 46.5 },
            Point { x: 230.5, y: 299.5 },
            Point { x: 49.5, y: 299.5 },
        ];
        assert!(recover_inner_card_border(physical, &image, &ExtractorConfig::default()).is_none());
        let opaque = GrayImage::from_fn(280, 340, |x, y| {
            let in_card = (40..=240).contains(&x) && (30..=310).contains(&y);
            let in_printed_panel = (50..=230).contains(&x) && (47..=299).contains(&y);
            Luma([if in_printed_panel {
                14
            } else if in_card {
                220
            } else {
                100
            }])
        });
        assert!(
            recover_inner_card_border(sleeve_corners(), &opaque, &ExtractorConfig::default())
                .is_none()
        );
    }
}

#[cfg(test)]
/// Independent black, white, and silver-gray card controls around reflective sleeve rims.
mod sleeve_colour_tests {
    use super::*;
    use crate::CardExtractor;
    use image::Luma;

    /// Draws an unequal sleeve margin, sustained physical frame, and printed interior.
    fn scene(frame: u8, background: u8, sleeve: bool, inner_seam: bool) -> GrayImage {
        GrayImage::from_fn(280, 340, |x, y| {
            let in_sleeve = sleeve && (40..=240).contains(&x) && (30..=310).contains(&y);
            let rim = in_sleeve && (x == 40 || x == 240 || y == 30 || y == 310);
            let seam = inner_seam
                && in_sleeve
                && ((x == 47 || x == 233) && (39..=302).contains(&y)
                    || (y == 39 || y == 302) && (47..=233).contains(&x));
            let in_card = (50..=230).contains(&x) && (47..=299).contains(&y);
            let in_panel = (58..=222).contains(&x) && (55..=279).contains(&y);
            let value = if rim || seam {
                250
            } else if in_card {
                if in_panel && (!(62..=218).contains(&x) || !(59..=275).contains(&y)) {
                    (60 + (x * 17 + y * 31) % 160) as u8
                } else if in_panel {
                    let base = if x < 108 {
                        70
                    } else if x < 171 {
                        140
                    } else {
                        210
                    };
                    base + (if (y / 30) % 2 == 0 { 10 } else { 0 })
                } else {
                    frame
                }
            } else if in_sleeve {
                background + 10
            } else {
                background
            };
            Luma([value])
        })
    }

    /// The sleeve perimeter is separate from the independently rendered physical frame.
    fn outer() -> [Point; 4] {
        [
            Point { x: 39.5, y: 29.5 },
            Point { x: 240.5, y: 29.5 },
            Point { x: 240.5, y: 310.5 },
            Point { x: 39.5, y: 310.5 },
        ]
    }

    /// Physical pixel transitions give these corners without relying on detector output.
    fn physical() -> [Point; 4] {
        [
            Point { x: 49.5, y: 46.5 },
            Point { x: 230.5, y: 46.5 },
            Point { x: 230.5, y: 299.5 },
            Point { x: 49.5, y: 299.5 },
        ]
    }

    #[test]
    /// All frame colors use observed edges and preserve the complete card in public extraction.
    fn recovers_black_white_and_silver_frames_with_crop_fidelity() {
        for frame in [14, 150, 235] {
            for background in [100, 180] {
                let gray = scene(frame, background, true, false);
                let refined =
                    recover_inner_card_border(outer(), &gray, &ExtractorConfig::default())
                        .expect("four colored physical edges");
                for (actual, expected) in refined.into_iter().zip(physical()) {
                    assert!(
                        squared_distance(actual, expected).sqrt() < 1.0,
                        "frame {frame}, background {background}, actual {actual:?}"
                    );
                }
                let image = DynamicImage::ImageLuma8(gray.clone());
                let cards = CardExtractor::default()
                    .extract(&image)
                    .expect("valid image");
                assert_eq!(cards.len(), 1, "frame {frame}, background {background}");
                for (actual, expected) in cards[0].detection.corners.into_iter().zip(physical()) {
                    assert!(
                        squared_distance(actual, expected).sqrt() < 1.0,
                        "public frame {frame}, background {background}, actual {actual:?}"
                    );
                }
                for (x, y) in [(50, 70), (125, 120), (200, 230), (2, 175)] {
                    let source_x = 49.5 + x as f32 / 249.0 * 181.0;
                    let source_y = 46.5 + y as f32 / 349.0 * 253.0;
                    let expected = sample_luma(&gray, source_x, source_y);
                    let actual = f32::from(cards[0].image.get_pixel(x, y)[0]);
                    assert!(
                        (actual - expected).abs() < 3.0,
                        "frame {frame}, sample {x}/{y}: {actual} vs {expected}"
                    );
                }
            }
        }
    }

    #[test]
    /// A thin bright seam cannot replace the first sustained physical frame.
    fn ignores_inner_sleeve_seam_and_stronger_printed_panels() {
        for frame in [14, 150, 235] {
            let gray = scene(frame, 100, true, true);
            let refined = recover_inner_card_border(outer(), &gray, &ExtractorConfig::default())
                .expect("physical frame after thin seam");
            for (actual, expected) in refined.into_iter().zip(physical()) {
                assert!(
                    squared_distance(actual, expected).sqrt() < 1.0,
                    "frame {frame}, {actual:?}"
                );
            }
        }
    }

    #[test]
    /// Opaque white and silver frames with printed panels remain physical boundaries.
    fn preserves_unsleeved_white_and_silver_cards() {
        for frame in [14, 150, 235] {
            for background in [100, 180] {
                let gray = scene(frame, background, false, false);
                assert!(
                    recover_inner_card_border(physical(), &gray, &ExtractorConfig::default())
                        .is_none()
                );
                let cards = CardExtractor::default()
                    .detect(&DynamicImage::ImageLuma8(gray))
                    .expect("valid image");
                assert_eq!(cards.len(), 1);
                for (actual, expected) in cards[0].corners.into_iter().zip(physical()) {
                    assert!(
                        squared_distance(actual, expected).sqrt() < 1.0,
                        "unsleeved frame {frame}, {actual:?}"
                    );
                }
            }
        }
    }
    #[test]
    /// Every color requires four physical frame edges and printed interior content.
    fn rejects_missing_edges_and_empty_colored_rims() {
        for frame in [14, 150, 235] {
            for missing in 0..4 {
                let mut image = scene(frame, 100, true, false);
                for y in 47..=299 {
                    for x in 50..=230 {
                        let absent = match missing {
                            0 => y < 55,
                            1 => x > 222,
                            2 => y > 279,
                            _ => x < 58,
                        };
                        if absent {
                            image.put_pixel(x, y, Luma([110]));
                        }
                    }
                }
                assert!(
                    recover_inner_card_border(outer(), &image, &ExtractorConfig::default())
                        .is_none(),
                    "frame {frame}, missing {missing}"
                );
            }
            let mut empty = scene(frame, 100, true, false);
            for y in 55..=279 {
                for x in 58..=222 {
                    empty.put_pixel(x, y, Luma([110]));
                }
            }
            assert!(
                recover_inner_card_border(outer(), &empty, &ExtractorConfig::default()).is_none(),
                "empty colored rim {frame}"
            );
        }
    }
}

#[cfg(test)]
/// Verifies that outward frame recovery depends on observed bands, not their color.
mod frame_colour_tests {
    use super::*;
    use image::Luma;

    /// The printed panel deliberately has a wider frame beneath its lower edge.
    fn panel_corners() -> [Point; 4] {
        [
            Point { x: 88.0, y: 68.0 },
            Point { x: 252.0, y: 68.0 },
            Point { x: 252.0, y: 292.0 },
            Point { x: 88.0, y: 292.0 },
        ]
    }

    /// Physical half-pixel boundaries of the independent observed card frame.
    fn physical_corners() -> [Point; 4] {
        [
            Point { x: 79.5, y: 59.5 },
            Point { x: 260.5, y: 59.5 },
            Point { x: 260.5, y: 312.5 },
            Point { x: 79.5, y: 312.5 },
        ]
    }

    /// Draws a frame, optionally continuing one side without an observable edge.
    fn framed_scene(tone: u8, missing_edge: Option<usize>) -> GrayImage {
        GrayImage::from_fn(360, 400, |x, y| {
            let extended = match missing_edge {
                Some(0) => y < 60 && (80..=260).contains(&x),
                Some(1) => x > 260 && (60..=312).contains(&y),
                Some(2) => y > 312 && (80..=260).contains(&x),
                Some(3) => x < 80 && (60..=312).contains(&y),
                _ => false,
            };
            Luma([if (88..=252).contains(&x) && (68..=292).contains(&y) {
                if tone > 200 { 60 } else { 235 }
            } else if extended || ((80..=260).contains(&x) && (60..=312).contains(&y)) {
                tone
            } else if tone > 100 {
                70
            } else {
                160
            }])
        })
    }

    #[test]
    /// Black, white, and silver frames recover the same independently observed corners.
    fn recovers_black_white_and_silver_frames() {
        for tone in [14, 235, 150] {
            let image = framed_scene(tone, None);
            let actual = expand_card_border(panel_corners(), &image)
                .unwrap_or_else(|| panic!("frame {tone} has four observed edges"));
            for (actual, expected) in actual.into_iter().zip(physical_corners()) {
                assert!(
                    squared_distance(actual, expected).sqrt() < 1.0,
                    "frame {tone}: {actual:?}"
                );
            }
        }
    }

    #[test]
    /// Each frame color needs a separately observed outer transition on every side.
    fn requires_each_frame_edge_for_every_colour() {
        for tone in [14, 235, 150] {
            for missing_edge in 0..4 {
                assert!(
                    expand_card_border(panel_corners(), &framed_scene(tone, Some(missing_edge)))
                        .is_none(),
                    "frame {tone}, missing edge {missing_edge}"
                );
            }
        }
    }

    #[test]
    /// Complete physical outlines are left alone rather than padded with surface pixels.
    fn preserves_selected_physical_edges_for_every_colour() {
        for tone in [14, 235, 150] {
            assert!(
                expand_card_border(physical_corners(), &framed_scene(tone, None)).is_none(),
                "frame {tone}"
            );
        }
    }

    #[test]
    /// A reflective sleeve line returning to the surrounding tone cannot enclose a frame.
    fn does_not_expand_card_into_reflective_sleeve() {
        for tone in [14, 235, 150] {
            let card = framed_scene(tone, None);
            let image = GrayImage::from_fn(360, 400, |x, y| {
                let physical = (80..=260).contains(&x) && (60..=312).contains(&y);
                let sleeve = (70..=270).contains(&x) && (50..=322).contains(&y);
                let rim = sleeve && (x <= 71 || x >= 269 || y <= 51 || y >= 321);
                if physical {
                    *card.get_pixel(x, y)
                } else {
                    Luma([if rim {
                        240
                    } else if sleeve {
                        110
                    } else {
                        100
                    }])
                }
            });
            assert!(
                expand_card_border(physical_corners(), &image).is_none(),
                "frame {tone}"
            );
        }
    }

    #[test]
    /// Narrow physical borders remain recoverable when the outer transition is nearby.
    fn recovers_narrow_frames_for_every_colour() {
        let panel = [
            Point { x: 83.0, y: 63.0 },
            Point { x: 257.0, y: 63.0 },
            Point { x: 257.0, y: 292.0 },
            Point { x: 83.0, y: 292.0 },
        ];
        for tone in [14, 235, 150] {
            let image = GrayImage::from_fn(360, 400, |x, y| {
                Luma([if (83..=257).contains(&x) && (63..=292).contains(&y) {
                    if tone > 200 { 60 } else { 235 }
                } else if (80..=260).contains(&x) && (60..=312).contains(&y) {
                    tone
                } else if tone > 100 {
                    70
                } else {
                    160
                }])
            });
            let recovered = expand_card_border(panel, &image).expect("narrow observed frame");
            for (actual, expected) in recovered.into_iter().zip(physical_corners()) {
                assert!(
                    squared_distance(actual, expected).sqrt() < 1.0,
                    "frame {tone}: {actual:?}"
                );
            }
        }
    }

    #[test]
    /// A soft printed-panel edge must not beat a weaker physical border farther out.
    fn follows_physical_frame_after_blurred_panel_edge() {
        let image = GrayImage::from_fn(360, 400, |x, y| {
            Luma([if (88..=252).contains(&x) && (68..=292).contains(&y) {
                235
            } else if (80..=260).contains(&x) && (60..=312).contains(&y) {
                14
            } else {
                50
            }])
        });
        let blurred = imageops::blur(&image, 1.3);
        let recovered =
            expand_card_border(panel_corners(), &blurred).expect("observed outer frame");
        for (actual, expected) in recovered.into_iter().zip(physical_corners()) {
            assert!(
                squared_distance(actual, expected).sqrt() < 1.5,
                "{actual:?}"
            );
        }
    }
}

/// Reconsiders partially covered outlines only against independently found cards.
fn refine_occluded_outlines(
    mask: &GrayImage,
    image: &GrayImage,
    config: &ExtractorConfig,
    candidates: &mut [CardDetection],
) {
    if !outlines_may_overlap(candidates) {
        return;
    }
    let occluders: Vec<_> = candidates
        .iter()
        .filter(|candidate| {
            candidate.corners.iter().all(|p| {
                p.x > 0.0
                    && p.y > 0.0
                    && p.x < mask.width() as f32 - 1.0
                    && p.y < mask.height() as f32 - 1.0
            })
        })
        .cloned()
        .collect();
    if occluders.len() < 2 {
        return;
    }
    let image_area = mask.width() as f32 * mask.height() as f32;
    let mut recovered = Vec::new();
    for contour in find_contours::<i32>(mask) {
        if contour.points.len() < 4 {
            continue;
        }
        let points: Vec<_> = contour
            .points
            .iter()
            .map(|p| Point {
                x: p.x as f32,
                y: p.y as f32,
            })
            .collect();
        let (minimum, maximum) = bounds(&points);
        if (maximum.x - minimum.x) * (maximum.y - minimum.y) < config.min_area_fraction * image_area
        {
            continue;
        }
        if minimum.x <= 0.0
            && minimum.y <= 0.0
            && maximum.x >= mask.width() as f32 - 1.0
            && maximum.y >= mask.height() as f32 - 1.0
        {
            continue;
        }
        let hull = convex_hull(points.clone());
        let area = polygon_area(&hull);
        if hull.len() < 4
            || area < config.min_area_fraction * image_area
            || area > config.max_area_fraction * image_area * 1.15
        {
            continue;
        }
        collect_inferred_candidates(
            &points,
            polygon_perimeter(&hull),
            mask,
            config,
            &occluders,
            &mut recovered,
        );
    }
    recovered.sort_by(|a, b| b.detection.area.total_cmp(&a.detection.area));
    for mut proposal in recovered {
        let recovered = &mut proposal.detection;
        let Some(corners) = orient_card_corners(
            refine_gradient_corners(recovered.corners, image),
            config.aspect_ratio_tolerance,
            image.width(),
            image.height(),
        ) else {
            continue;
        };
        recovered.corners = corners;
        recovered.area = polygon_area(&corners);
        if recovered.area < config.min_area_fraction * image_area
            || recovered.area > config.max_area_fraction * image_area
        {
            continue;
        }
        if let Some(original) = candidates
            .iter_mut()
            .find(|original| proposal.can_replace(original, &occluders))
        {
            *original = proposal.detection;
        }
    }
}

/// Allows recovery only when two retained outlines have touching or overlapping bounds.
fn outlines_may_overlap(candidates: &[CardDetection]) -> bool {
    candidates.iter().enumerate().any(|(index, left)| {
        let (left_minimum, left_maximum) = bounds(&left.corners);
        candidates[index + 1..].iter().any(|right| {
            let (right_minimum, right_maximum) = bounds(&right.corners);
            left_minimum.x <= right_maximum.x
                && right_minimum.x <= left_maximum.x
                && left_minimum.y <= right_maximum.y
                && right_minimum.y <= left_maximum.y
        })
    })
}

/// Requires visible sections on all four sides and a detected occluder over gaps.
fn occlusion_explains_edges(
    corners: &[Point; 4],
    contour: &[Point],
    occluders: &[CardDetection],
) -> bool {
    if occluders.is_empty() {
        return false;
    }
    let mut bins = [[false; 32]; 4];
    for i in 0..4 {
        let start = corners[i];
        let end = corners[(i + 1) % 4];
        let ls = squared_distance(start, end);
        let length = ls.sqrt();
        if length < 12.0 {
            return false;
        }
        let distance = (length * 0.006).clamp(2.0, 3.0);
        for &p in contour {
            let v =
                ((p.x - start.x) * (end.x - start.x) + (p.y - start.y) * (end.y - start.y)) / ls;
            if (-0.01..=1.01).contains(&v) && cross(start, end, p).abs() / length <= distance {
                bins[i][((v.clamp(0.0, 1.0) * 32.0) as usize).min(31)] = true;
            }
        }
    }
    let coverage = bins.map(|edge| edge.iter().filter(|&&v| v).count());
    if coverage.iter().any(|&v| v < 8)
        || coverage.iter().filter(|&&v| v >= 28).count() < 2
        || coverage.iter().all(|&v| v >= 20)
    {
        return false;
    }
    let target = CardDetection {
        corners: *corners,
        area: polygon_area(corners),
    };
    occluders.iter().any(|occluder| {
        let (iou, containment) = quadrilateral_overlap(&target, occluder);
        if !(0.02..=0.45).contains(&iou) || containment > 0.9 {
            return false;
        }
        (0..4).all(|i| {
            let start = corners[i];
            let end = corners[(i + 1) % 4];
            let unexplained = (0..32)
                .filter(|&bin| {
                    if bins[i][bin] {
                        return false;
                    }
                    let fraction = (bin as f32 + 0.5) / 32.0;
                    let p = Point {
                        x: start.x + (end.x - start.x) * fraction,
                        y: start.y + (end.y - start.y) * fraction,
                    };
                    !(0..4).all(|j| {
                        cross(occluder.corners[j], occluder.corners[(j + 1) % 4], p)
                            / squared_distance(occluder.corners[j], occluder.corners[(j + 1) % 4])
                                .sqrt()
                            >= -3.0
                    })
                })
                .count();
            unexplained <= 2
        })
    })
}

/// Replaces a clipped crop only when its fourth side follows an occluder's edge.
fn replacement_follows_three_sides(
    original: &CardDetection,
    recovered: &CardDetection,
    occluders: &[CardDetection],
) -> bool {
    let lengths = edge_lengths(&recovered.corners);
    if (0..4).any(|i| {
        let p = recovered.corners[(i + 3) % 4];
        let c = recovered.corners[i];
        let n = recovered.corners[(i + 1) % 4];
        let dot = (p.x - c.x) * (n.x - c.x) + (p.y - c.y) * (n.y - c.y);
        dot.abs() > lengths[(i + 3) % 4] * lengths[i] * 0.05
    }) {
        return false;
    }
    if recovered.area < original.area * 1.05 || recovered.area > original.area * 1.6 {
        return false;
    }
    let (iou, containment) = quadrilateral_overlap(original, recovered);
    if iou < 0.55 || containment < 0.85 {
        return false;
    }
    let distance_limit = edge_lengths(&recovered.corners)
        .into_iter()
        .fold(f32::INFINITY, f32::min)
        * 0.12;
    (0..4).any(|shift| {
        let shared: Vec<_> = (0..4)
            .map(|i| {
                let a = original.corners[i];
                let b = original.corners[(i + 1) % 4];
                let c = recovered.corners[(i + shift) % 4];
                let d = recovered.corners[(i + shift + 1) % 4];
                let dot = (b.x - a.x) * (d.x - c.x) + (b.y - a.y) * (d.y - c.y);
                let parallel = dot.abs() / (squared_distance(a, b) * squared_distance(c, d)).sqrt();
                let midpoint = Point {
                    x: (a.x + b.x) * 0.5,
                    y: (a.y + b.y) * 0.5,
                };
                parallel > 0.995
                    && cross(c, d, midpoint).abs() / squared_distance(c, d).sqrt() <= distance_limit
            })
            .collect();
        if shared.iter().filter(|&&v| v).count() != 3 {
            return false;
        }
        let Some(i) = shared.iter().position(|&v| !v) else {
            return false;
        };
        let a = original.corners[i];
        let b = original.corners[(i + 1) % 4];
        occluders.iter().any(|occluder| {
            let (iou, containment) = quadrilateral_overlap(recovered, occluder);
            if !(0.02..=0.45).contains(&iou) || containment > 0.9 {
                return false;
            }
            (0..4).any(|j| {
                let c = occluder.corners[j];
                let d = occluder.corners[(j + 1) % 4];
                let length = squared_distance(c, d).sqrt();
                let dot = (b.x - a.x) * (d.x - c.x) + (b.y - a.y) * (d.y - c.y);
                dot.abs() / (squared_distance(a, b).sqrt() * length) > 0.995
                    && cross(c, d, a).abs() / length <= 4.0
                    && cross(c, d, b).abs() / length <= 4.0
                    && observed_segments_overlap(a, b, c, d)
            })
        })
    })
}

/// Requires a quarter of the shorter observed segment to lie along the other.
///
/// Collinearity alone does not establish that the front card occupies the
/// clipped boundary: a distant segment can share the same infinite line.
fn observed_segments_overlap(first: Point, second: Point, start: Point, end: Point) -> bool {
    let first_length = squared_distance(first, second).sqrt();
    let second_length = squared_distance(start, end).sqrt();
    if first_length < 1.0 || second_length < 1.0 {
        return false;
    }
    let project = |point: Point| {
        ((point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y))
            / second_length
    };
    let first = project(first);
    let second = project(second);
    let overlap = first.max(second).min(second_length) - first.min(second).max(0.0);
    overlap >= first_length.min(second_length) * 0.25
}

#[cfg(test)]
/// Safety guards for extending a clipped outline behind an observed card.
mod occlusion_tests {
    use super::*;

    /// Makes a measured outline from four supplied visible corners.
    fn outline(corners: [Point; 4]) -> CardDetection {
        CardDetection {
            area: polygon_area(&corners),
            corners,
        }
    }

    /// Places an axis-aligned outline at the supplied scene coordinates.
    fn rectangle(left: f32, top: f32, right: f32, bottom: f32) -> CardDetection {
        outline([
            Point { x: left, y: top },
            Point { x: right, y: top },
            Point {
                x: right,
                y: bottom,
            },
            Point { x: left, y: bottom },
        ])
    }

    /// Supplies two complete rear sides and the visible starts of its other sides.
    fn visible_rear_edges() -> Vec<Point> {
        let mut points = Vec::new();
        points.extend((100..=200).map(|x| Point {
            x: x as f32,
            y: 80.0,
        }));
        points.extend((80..=220).map(|y| Point {
            x: 100.0,
            y: y as f32,
        }));
        points.extend((80..=125).map(|y| Point {
            x: 200.0,
            y: y as f32,
        }));
        points.extend((100..=165).map(|x| Point {
            x: x as f32,
            y: 220.0,
        }));
        points
    }

    #[test]
    /// A front card explains the rear border gaps while every rear side remains visible.
    fn visible_side_sections_with_one_covering_card_allow_recovery() {
        let rear = rectangle(100.0, 80.0, 200.0, 220.0);
        let front = rectangle(165.0, 120.0, 245.0, 250.0);
        assert!(occlusion_explains_edges(
            &rear.corners,
            &visible_rear_edges(),
            &[front],
        ));
    }

    #[test]
    /// Missing pixels need an overlapping card rather than absent or unrelated evidence.
    fn missing_side_sections_require_an_independent_occluder() {
        let rear = rectangle(100.0, 80.0, 200.0, 220.0);
        let evidence = visible_rear_edges();
        for occluders in [
            Vec::new(),
            vec![rectangle(260.0, 80.0, 360.0, 220.0)],
            vec![rear.clone()],
            vec![rectangle(0.0, 0.0, 640.0, 480.0)],
        ] {
            assert!(!occlusion_explains_edges(
                &rear.corners,
                &evidence,
                &occluders
            ));
        }
    }

    #[test]
    /// Two unrelated objects cannot combine to explain one card's missing border.
    fn a_single_occluder_must_explain_all_missing_sections() {
        let rear = rectangle(100.0, 80.0, 200.0, 220.0);
        let occluders = [
            rectangle(190.0, 120.0, 250.0, 190.0),
            rectangle(155.0, 190.0, 220.0, 250.0),
        ];
        assert!(!occlusion_explains_edges(
            &rear.corners,
            &visible_rear_edges(),
            &occluders,
        ));
    }

    #[test]
    /// An occluder cannot justify inventing an entirely unobserved rear side.
    fn an_entirely_hidden_side_still_prevents_recovery() {
        let rear = rectangle(100.0, 80.0, 200.0, 220.0);
        let front = rectangle(165.0, 120.0, 245.0, 250.0);
        let evidence: Vec<_> = visible_rear_edges()
            .into_iter()
            .filter(|point| point.x != 200.0 || point.y == 80.0)
            .collect();
        assert!(!occlusion_explains_edges(
            &rear.corners,
            &evidence,
            &[front]
        ));
    }

    /// Gives a clipped rear crop whose diagonal fourth side belongs to the front card.
    fn clipped_rear_and_front() -> (CardDetection, CardDetection, CardDetection) {
        let clipped = outline([
            Point { x: 100.0, y: 80.0 },
            Point { x: 185.0, y: 80.0 },
            Point { x: 150.0, y: 220.0 },
            Point { x: 100.0, y: 220.0 },
        ]);
        let rear = rectangle(100.0, 80.0, 200.0, 220.0);
        let front = outline([
            Point { x: 185.0, y: 80.0 },
            Point { x: 245.0, y: 95.0 },
            Point { x: 210.0, y: 235.0 },
            Point { x: 150.0, y: 220.0 },
        ]);
        (clipped, rear, front)
    }

    #[test]
    /// A mistaken front-card boundary can be replaced while the rear's three sides agree.
    fn replacement_requires_the_occluders_observed_boundary() {
        let (clipped, rear, front) = clipped_rear_and_front();
        assert!(replacement_follows_three_sides(&clipped, &rear, &[front]));
        assert!(!replacement_follows_three_sides(&clipped, &rear, &[]));
        let unrelated_boundary = rectangle(165.0, 100.0, 240.0, 245.0);
        assert!(!replacement_follows_three_sides(
            &clipped,
            &rear,
            &[unrelated_boundary],
        ));
    }

    #[test]
    /// Separate cards cannot contribute the gap explanation and clipped boundary.
    fn one_occluder_must_explain_both_gaps_and_clipped_boundary() {
        let rear = rectangle(100.0, 80.0, 200.0, 220.0);
        let clipped = outline([
            Point { x: 100.0, y: 80.0 },
            Point { x: 200.0, y: 80.0 },
            Point { x: 170.0, y: 220.0 },
            Point { x: 100.0, y: 220.0 },
        ]);
        let covering = rectangle(165.0, 120.0, 245.0, 250.0);
        let parallel = outline([
            Point { x: 185.0, y: 150.0 },
            Point {
                x: 285.0,
                y: 171.428_57,
            },
            Point {
                x: 255.0,
                y: 311.428_57,
            },
            Point { x: 155.0, y: 290.0 },
        ]);
        let occluders = [covering, parallel];
        for detection in [&rear, &clipped, &occluders[0], &occluders[1]] {
            assert!(orient_card_corners(detection.corners, 0.2, 400, 400).is_some());
        }
        let evidence = visible_rear_edges();
        let associated: Vec<_> = occluders
            .iter()
            .enumerate()
            .filter_map(|(index, occluder)| {
                occlusion_explains_edges(&rear.corners, &evidence, std::slice::from_ref(occluder))
                    .then_some(index)
            })
            .collect();
        assert_eq!(associated, vec![0]);
        assert!(!replacement_follows_three_sides(
            &clipped,
            &rear,
            std::slice::from_ref(&occluders[0]),
        ));
        assert!(replacement_follows_three_sides(
            &clipped,
            &rear,
            std::slice::from_ref(&occluders[1]),
        ));
        let proposal = InferredOutline {
            detection: rear,
            occluder_indices: associated,
        };
        assert!(!proposal.can_replace(&clipped, &occluders));
    }

    #[test]
    /// A real shared section is required even when infinite edge lines coincide.
    fn clipped_boundary_requires_finite_observed_segment_overlap() {
        let point = |y| Point { x: 100.0, y };
        assert!(observed_segments_overlap(
            point(0.0),
            point(100.0),
            point(60.0),
            point(160.0),
        ));
        for start in [95.0, 101.0] {
            assert!(!observed_segments_overlap(
                point(0.0),
                point(100.0),
                point(start),
                point(start + 100.0),
            ));
        }
    }

    #[test]
    /// A parallel front edge at another position cannot justify a replacement.
    fn an_occluder_boundary_must_match_the_clipped_side_position() {
        let (clipped, rear, mut front) = clipped_rear_and_front();
        for point in &mut front.corners {
            point.x += 10.0;
        }
        assert!(!replacement_follows_three_sides(&clipped, &rear, &[front]));
    }

    #[test]
    /// Border recovery must retain three rear side directions rather than changing two sides.
    fn replacement_preserves_three_existing_side_lines() {
        let (mut clipped, rear, front) = clipped_rear_and_front();
        clipped.corners[1].y += 20.0;
        clipped.area = polygon_area(&clipped.corners);
        assert!(!replacement_follows_three_sides(&clipped, &rear, &[front]));
    }

    #[test]
    /// Even matching side and occluder evidence does not remove ambiguity in an oblique pose.
    fn ambiguous_oblique_recovery_is_rejected() {
        let (clipped, rear, front) = clipped_rear_and_front();
        let shear = |detection: CardDetection| {
            outline(detection.corners.map(|point| Point {
                x: point.x + 0.15 * (point.y - 80.0),
                y: point.y,
            }))
        };
        assert!(!replacement_follows_three_sides(
            &shear(clipped),
            &shear(rear),
            &[shear(front)],
        ));
    }

    #[test]
    /// An already rectangular crop cannot be enlarged merely because another card overlaps.
    fn a_complete_rectangular_crop_is_not_arbitrarily_enlarged() {
        let (_, recovered, front) = clipped_rear_and_front();
        let original = rectangle(110.0, 90.0, 190.0, 210.0);
        assert!(!replacement_follows_three_sides(
            &original,
            &recovered,
            &[front]
        ));
    }

    #[test]
    /// Empty scenes and separated cards cannot benefit from occlusion recovery.
    fn separated_outline_bounds_skip_occlusion_recovery() {
        let cards = [
            rectangle(20.0, 20.0, 120.0, 160.0),
            rectangle(140.0, 20.0, 240.0, 160.0),
        ];
        assert!(!outlines_may_overlap(&[]));
        assert!(!outlines_may_overlap(std::slice::from_ref(&cards[0])));
        assert!(!outlines_may_overlap(&cards));
    }

    #[test]
    /// Bounds that merely touch remain eligible because a clipped rear crop can abut its occluder.
    fn touching_outline_bounds_remain_eligible() {
        let cards = [
            rectangle(20.0, 20.0, 120.0, 160.0),
            rectangle(120.0, 20.0, 220.0, 160.0),
        ];
        assert!(outlines_may_overlap(&cards));
    }

    #[test]
    /// The inexpensive gate remains permissive when only rotated bounding boxes overlap.
    fn overlapping_bounds_do_not_require_polygon_overlap() {
        let cards = [
            outline([
                Point { x: 50.0, y: 0.0 },
                Point { x: 100.0, y: 50.0 },
                Point { x: 50.0, y: 100.0 },
                Point { x: 0.0, y: 50.0 },
            ]),
            outline([
                Point { x: 130.0, y: 80.0 },
                Point { x: 180.0, y: 130.0 },
                Point { x: 130.0, y: 180.0 },
                Point { x: 80.0, y: 130.0 },
            ]),
        ];
        assert_eq!(quadrilateral_overlap(&cards[0], &cards[1]).0, 0.0);
        assert!(outlines_may_overlap(&cards));
    }
}
#[cfg(test)]
/// Checks physical-color recovery without assuming a printed frame or luma contrast.
mod artwork_sleeve_tests {
    use super::*;
    use crate::CardExtractor;
    use image::{Rgb, RgbImage};

    /// Renders an independently measured perimeter with changing, low-luma-contrast artwork.
    fn scene(card: bool, seam: bool, mismatch: bool) -> RgbImage {
        RgbImage::from_fn(280, 340, |x, y| {
            let sleeve = (40..=240).contains(&x) && (30..=310).contains(&y);
            let rim = sleeve && (x == 40 || x == 240 || y == 30 || y == 310);
            let inner_seam = seam
                && ((x == 47 || x == 233) && (39..=302).contains(&y)
                    || (y == 39 || y == 302) && (47..=233).contains(&x));
            let physical = card && (50..=230).contains(&x) && (47..=299).contains(&y);
            let panel = (62..=218).contains(&x) && (59..=287).contains(&y);
            let printed_line = (x == 60 || x == 220) && (57..=289).contains(&y)
                || (y == 57 || y == 289) && (60..=220).contains(&x);
            Rgb(if rim || inner_seam {
                [250; 3]
            } else if physical {
                if printed_line {
                    [5; 3]
                } else if panel {
                    if x < 140 { [40; 3] } else { [190; 3] }
                } else if (x / 40 + y / 40) % 3 == 0 {
                    [10, 120, 170]
                } else if (x / 40 + y / 40) % 3 == 1 {
                    [170, 80, 60]
                } else {
                    [80, 100, 220]
                }
            } else if sleeve {
                if mismatch {
                    [20, 120, 170]
                } else {
                    [90, 110, 80]
                }
            } else {
                [80, 100, 70]
            })
        })
    }

    /// The observed outer reflective sleeve is separate from the physical image.
    fn outer() -> [Point; 4] {
        [
            Point { x: 39.5, y: 29.5 },
            Point { x: 240.5, y: 29.5 },
            Point { x: 240.5, y: 310.5 },
            Point { x: 39.5, y: 310.5 },
        ]
    }

    /// Ground truth comes from the physical color transition, independently of detection.
    fn physical() -> [Point; 4] {
        [
            Point { x: 49.5, y: 46.5 },
            Point { x: 230.5, y: 46.5 },
            Point { x: 230.5, y: 299.5 },
            Point { x: 49.5, y: 299.5 },
        ]
    }

    #[test]
    /// Similar luma cannot hide color edges or let a stronger printed line win.
    fn follows_first_physical_colour_edges_and_preserves_crop() {
        let rgb = scene(true, true, false);
        let image = DynamicImage::ImageRgb8(rgb.clone());
        let gray = image.to_luma8();
        let recovered = recover_inner_artwork(outer(), &gray, &image, &ExtractorConfig::default())
            .expect("four physical color boundaries");
        for (actual, expected) in recovered.into_iter().zip(physical()) {
            assert!(
                squared_distance(actual, expected).sqrt() < 1.0,
                "{actual:?}"
            );
        }
        let cards = CardExtractor::default()
            .extract(&image)
            .expect("valid colors");
        assert_eq!(cards.len(), 1);
        for (actual, expected) in cards[0].detection.corners.into_iter().zip(physical()) {
            assert!(
                squared_distance(actual, expected).sqrt() < 1.0,
                "{actual:?}"
            );
        }
        for (x, y) in [(2, 175), (50, 70), (125, 120), (200, 230)] {
            let expected = sample_colour(
                &rgb,
                49.5 + x as f32 / 249.0 * 181.0,
                46.5 + y as f32 / 349.0 * 253.0,
            );
            let actual = cards[0].image.get_pixel(x, y);
            for channel in 0..3 {
                assert!(
                    (f32::from(actual[channel]) - expected[channel]).abs() < 3.0,
                    "sample {x}/{y}, channel {channel}: {actual:?} versus {expected:?}"
                );
            }
        }
    }

    #[test]
    /// Reflective seams and different-colored inner bands do not prove a clear sleeve.
    fn rejects_empty_seams_and_nonmatching_clear_bands() {
        for (card, seam, mismatch) in [
            (false, false, false),
            (false, true, false),
            (true, false, true),
        ] {
            let image = DynamicImage::ImageRgb8(scene(card, seam, mismatch));
            assert!(
                recover_inner_artwork(
                    outer(),
                    &image.to_luma8(),
                    &image,
                    &ExtractorConfig::default()
                )
                .is_none()
            );
        }
    }

    #[test]
    /// A clipped color side cannot borrow evidence from the surrounding rim.
    fn requires_clear_band_and_separate_physical_edge_on_every_side() {
        for side in 0..4 {
            let mut rgb = scene(true, false, false);
            for y in 31..310 {
                for x in 41..240 {
                    let missing = match side {
                        0 => y < 47 && (50..=230).contains(&x),
                        1 => x > 230 && (47..=299).contains(&y),
                        2 => y > 299 && (50..=230).contains(&x),
                        _ => x < 50 && (47..=299).contains(&y),
                    };
                    if missing {
                        rgb.put_pixel(x, y, Rgb([10, 120, 170]));
                    }
                }
            }
            let image = DynamicImage::ImageRgb8(rgb);
            assert!(
                recover_inner_artwork(
                    outer(),
                    &image.to_luma8(),
                    &image,
                    &ExtractorConfig::default()
                )
                .is_none(),
                "side {side}"
            );
        }
    }

    #[test]
    /// A selected physical color perimeter lacks the surrounding reflective-rim proof.
    fn preserves_selected_physical_colour_edges() {
        let image = DynamicImage::ImageRgb8(scene(true, false, false));
        assert!(
            recover_inner_artwork(
                physical(),
                &image.to_luma8(),
                &image,
                &ExtractorConfig::default()
            )
            .is_none()
        );
    }
}
