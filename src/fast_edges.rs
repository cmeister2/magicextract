//! Canny edges with fixed-size filters and algebraic gradient direction bins.
//!
//! The blur, Sobel kernels, suppression directions, and six-neighbor hysteresis
//! follow `imageproc::edges::canny`. Integer squared magnitudes preserve the
//! ordering of Sobel strengths; `hypot` is needed only for surviving maxima.

use std::sync::OnceLock;

use image::GrayImage;

/// Tangent of 22.5 degrees, separating horizontal, diagonal, and vertical bins.
const TAN_22_5: f32 = 0.414_213_57;

/// Produces the same Canny edge mask without per-pixel trigonometric functions.
///
/// The caller supplies finite, positive thresholds with `high > low`.
pub(super) fn canny(image: &GrayImage, low: f32, high: f32) -> GrayImage {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return GrayImage::new(width, height);
    }
    let blurred = gaussian_blur(image);
    let row_length = width as usize;
    let row_count = height as usize;
    let pixels = blurred.as_raw();
    let mut horizontal = vec![0_i16; pixels.len()];
    let mut vertical = vec![0_i16; pixels.len()];
    let mut magnitude = vec![0_u32; pixels.len()];

    for y in 0..row_count {
        // Compute both Sobel gradients from shared rows, replicating image edges.
        let top = &pixels[y.saturating_sub(1) * row_length..][..row_length];
        let middle = &pixels[y * row_length..][..row_length];
        let bottom = &pixels[(y + 1).min(row_count - 1) * row_length..][..row_length];
        let offset = y * row_length;
        for x in [0, row_length - 1] {
            // Clamp the boundary neighbors before evaluating the scalar kernels.
            let left = x.saturating_sub(1);
            let right = (x + 1).min(row_length - 1);
            let dx = i32::from(top[right]) - i32::from(top[left])
                + 2 * (i32::from(middle[right]) - i32::from(middle[left]))
                + i32::from(bottom[right])
                - i32::from(bottom[left]);
            let dy = i32::from(bottom[left]) - i32::from(top[left])
                + 2 * (i32::from(bottom[x]) - i32::from(top[x]))
                + i32::from(bottom[right])
                - i32::from(top[right]);
            horizontal[offset + x] = dx as i16;
            vertical[offset + x] = dy as i16;
            magnitude[offset + x] = (dx * dx + dy * dy) as u32;
        }
        // Uniform interior slices let LLVM vectorize the two Sobel kernels
        // together, while the replicated border columns stay scalar.
        let count = row_length - 2;
        let top_left = &top[..count];
        let top_center = &top[1..count + 1];
        let top_right = &top[2..];
        let middle_left = &middle[..count];
        let middle_right = &middle[2..];
        let bottom_left = &bottom[..count];
        let bottom_center = &bottom[1..count + 1];
        let bottom_right = &bottom[2..];
        let horizontal_row = &mut horizontal[offset + 1..offset + row_length - 1];
        let vertical_row = &mut vertical[offset + 1..offset + row_length - 1];
        let magnitude_row = &mut magnitude[offset + 1..offset + row_length - 1];
        for x in 0..count {
            // Store signed gradients and squared strength in one vectorizable pass.
            let dx = i32::from(top_right[x]) - i32::from(top_left[x])
                + 2 * (i32::from(middle_right[x]) - i32::from(middle_left[x]))
                + i32::from(bottom_right[x])
                - i32::from(bottom_left[x]);
            let dy = i32::from(bottom_left[x]) - i32::from(top_left[x])
                + 2 * (i32::from(bottom_center[x]) - i32::from(top_center[x]))
                + i32::from(bottom_right[x])
                - i32::from(top_right[x]);
            horizontal_row[x] = dx as i16;
            vertical_row[x] = dy as i16;
            magnitude_row[x] = (dx * dx + dy * dy) as u32;
        }
    }

    // A full integer below the threshold makes this early rejection independent
    // of the final `hypot` rounding. Sobel magnitudes cannot reach 2048.
    let low_floor = (low as u32).saturating_sub(1).min(2048);
    let minimum_squared = low_floor * low_floor;
    let mut output = vec![0_u8; pixels.len()];
    for y in 1..row_count - 1 {
        for x in 1..row_length - 1 {
            // Reject weak or nonmaximal gradients before computing exact strength.
            let index = y * row_length + x;
            let value = magnitude[index];
            if value == 0 || value < minimum_squared {
                continue;
            }
            let dx = horizontal[index];
            let dy = vertical[index];
            let (first, second) = match direction_bin(dx, dy) {
                0 => (index - 1, index + 1),
                1 => (index - row_length - 1, index + row_length + 1),
                2 => (index - row_length, index + row_length),
                _ => (index - row_length + 1, index + row_length - 1),
            };
            if value < magnitude[first] || value < magnitude[second] {
                continue;
            }
            // Keep reference threshold semantics, including strengths that lie
            // exactly on an arbitrary caller-supplied floating-point threshold.
            let strength = f32::from(dx).hypot(f32::from(dy));
            output[index] = if strength >= high {
                2
            } else if strength >= low {
                1
            } else {
                0
            };
        }
    }

    follow_edges(&mut output, row_length, row_count);
    // The buffer length and dimensions come from the already allocated input.
    GrayImage::from_raw(width, height, output).expect("edge buffer matches source dimensions")
}

/// Quantizes an integer Sobel gradient to the reference Canny direction bins.
fn direction_bin(dx: i16, dy: i16) -> u8 {
    let horizontal = f32::from(dx).abs();
    let vertical = f32::from(dy).abs();
    let horizontal_limit = horizontal * TAN_22_5;
    let vertical_limit = vertical * TAN_22_5;
    // atan2 and the reference degree conversion can round a nearby slope onto
    // the boundary itself, notably (-985, -408) onto exactly 22.5 degrees.
    // Preserve those rare cases while keeping the common path algebraic.
    if (vertical - horizontal_limit).abs() < 0.001 || (horizontal - vertical_limit).abs() < 0.001 {
        return reference_direction_bin(dx, dy);
    }
    if vertical < horizontal_limit {
        0
    } else if horizontal < vertical_limit {
        2
    } else if (dx < 0) == (dy < 0) {
        1
    } else {
        3
    }
}

/// Preserves floating-point boundary rounding for nearly exact diagonal bins.
fn reference_direction_bin(dx: i16, dy: i16) -> u8 {
    let mut angle = f32::from(dy).atan2(f32::from(dx)) * (180.0 / std::f32::consts::PI);
    if angle < 0.0 {
        angle += 180.0;
    }
    if !(22.5..157.5).contains(&angle) {
        0
    } else if angle < 67.5 {
        1
    } else if angle < 112.5 {
        2
    } else {
        3
    }
}

/// Retains strong edges and the weak edges reached by the reference traversal.
fn follow_edges(output: &mut [u8], width: usize, height: usize) {
    let mut pending = Vec::new();
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            // Grow each unvisited strong edge through connected weak candidates.
            let index = y * width + x;
            if output[index] != 2 {
                continue;
            }
            output[index] = u8::MAX;
            pending.push(index);
            while let Some(index) = pending.pop() {
                // Preserve imageproc's six directed neighbors and scan order.
                // Border strengths are zero, so border pixels never enter here.
                for next in [
                    index + 1,
                    index + width + 1,
                    index + width,
                    index - width - 1,
                    index - 1,
                    index + width - 1,
                ] {
                    if output[next] == 1 || output[next] == 2 {
                        output[next] = u8::MAX;
                        pending.push(next);
                    }
                }
            }
        }
    }
    for pixel in output {
        if *pixel != u8::MAX {
            *pixel = 0;
        }
    }
}

/// Applies the reference sigma-1.4 blur with seven explicitly ordered taps.
fn gaussian_blur(image: &GrayImage) -> GrayImage {
    let kernel = gaussian_kernel();
    let (width, height) = image.dimensions();
    let row_length = width as usize;
    let row_count = height as usize;
    let mut horizontal = vec![0; image.as_raw().len()];
    let mut padded = vec![0; row_length + 6];
    for (input, output) in image
        .as_raw()
        .chunks_exact(row_length)
        .zip(horizontal.chunks_exact_mut(row_length))
    {
        // Replicated padding keeps horizontal convolution free of boundary branches.
        padded[..3].fill(input[0]);
        padded[3..row_length + 3].copy_from_slice(input);
        padded[row_length + 3..].fill(input[row_length - 1]);
        for (window, value) in padded.windows(7).zip(output) {
            *value = (f32::from(window[0]) * kernel[0]
                + f32::from(window[1]) * kernel[1]
                + f32::from(window[2]) * kernel[2]
                + f32::from(window[3]) * kernel[3]
                + f32::from(window[4]) * kernel[4]
                + f32::from(window[5]) * kernel[5]
                + f32::from(window[6]) * kernel[6]) as u8;
        }
    }
    let mut output = vec![0; horizontal.len()];
    for y in 0..row_count {
        let rows: [&[u8]; 7] = std::array::from_fn(|tap| {
            let row = (y + tap).saturating_sub(3).min(row_count - 1);
            &horizontal[row * row_length..][..row_length]
        });
        for (x, value) in output[y * row_length..][..row_length]
            .iter_mut()
            .enumerate()
        {
            *value = (f32::from(rows[0][x]) * kernel[0]
                + f32::from(rows[1][x]) * kernel[1]
                + f32::from(rows[2][x]) * kernel[2]
                + f32::from(rows[3][x]) * kernel[3]
                + f32::from(rows[4][x]) * kernel[4]
                + f32::from(rows[5][x]) * kernel[5]
                + f32::from(rows[6][x]) * kernel[6]) as u8;
        }
    }
    GrayImage::from_raw(width, height, output).expect("blur buffer matches source dimensions")
}

/// Initializes the same normalized floating-point kernel as the reference blur.
fn gaussian_kernel() -> &'static [f32; 7] {
    /// Kernel values are constant across frames and need computing only once.
    static KERNEL: OnceLock<[f32; 7]> = OnceLock::new();
    KERNEL.get_or_init(|| {
        // Retain runtime transcendental evaluation used by imageproc's filter.
        let sigma = std::hint::black_box(1.4_f32);
        let mut kernel = [0.0; 7];
        for index in 0..=3 {
            // Mirror each Gaussian weight around the center before normalization.
            let value = (sigma * (2.0 * std::f32::consts::PI).sqrt()).recip()
                * (-(index as f32).powi(2) / (2.0 * sigma.powi(2))).exp();
            kernel[3 + index] = value;
            kernel[3 - index] = value;
        }
        let sum: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|value| *value /= sum);
        kernel
    })
}

#[cfg(test)]
/// Pixel-level compatibility tests for the optimized Canny implementation.
mod tests {
    use image::{GrayImage, Luma};

    use super::{canny, direction_bin, gaussian_blur};

    #[test]
    /// Every possible integer Sobel direction receives the reference angle bin.
    fn gradient_bins_match_reference_angles() {
        for dx in -1020_i16..=1020 {
            for dy in -1020_i16..=1020 {
                let mut angle = f32::from(dy).atan2(f32::from(dx)) * (180.0 / std::f32::consts::PI);
                if angle < 0.0 {
                    angle += 180.0;
                }
                let expected = if !(22.5..157.5).contains(&angle) {
                    0
                } else if angle < 67.5 {
                    1
                } else if angle < 112.5 {
                    2
                } else {
                    3
                };
                assert_eq!(direction_bin(dx, dy), expected, "gradient {dx},{dy}");
            }
        }
    }

    #[test]
    /// Padding and byte quantization agree even on images narrower than the kernel.
    fn fixed_blur_matches_reference_pixels() {
        for (width, height) in [(3, 3), (4, 8), (8, 4), (31, 29), (128, 96)] {
            for seed in [0, 127, 255, 789] {
                let image = test_image(width, height, seed);
                assert_eq!(
                    gaussian_blur(&image),
                    imageproc::filter::gaussian_blur_f32(&image, 1.4),
                    "blur {width}x{height}, seed {seed}",
                );
            }
        }
    }

    #[test]
    /// Noise, flat images, and exact strength thresholds retain identical masks.
    fn edge_masks_match_reference_pixels() {
        let boundary = 2.0_f32.sqrt() * 10.0;
        let thresholds = [
            (0.001, 0.1),
            (7.5, 22.5),
            (15.0, 45.0),
            (boundary.next_down(), boundary),
            (boundary, boundary.next_up()),
            (254.5, 400.0),
            (2000.0, 3000.0),
        ];
        for (width, height) in [(3, 3), (4, 8), (8, 4), (31, 29), (128, 96)] {
            for seed in [0, 127, 255, 789] {
                let image = test_image(width, height, seed);
                for (low, high) in thresholds {
                    assert_eq!(
                        canny(&image, low, high),
                        imageproc::edges::canny(&image, low, high),
                        "edges {width}x{height}, seed {seed}, thresholds {low},{high}",
                    );
                }
            }
        }
    }

    /// Creates flat and deterministically textured images without a random dependency.
    fn test_image(width: u32, height: u32, seed: u32) -> GrayImage {
        GrayImage::from_fn(width, height, |x, y| {
            let value = if seed <= 255 {
                seed as u8
            } else {
                let index = y * width + x;
                let mut state = index.wrapping_add(seed).wrapping_mul(0x045d_9f3b);
                state = (state ^ (state >> 16)).wrapping_mul(0x045d_9f3b);
                (state ^ (state >> 16)) as u8
            };
            Luma([value])
        })
    }
}
