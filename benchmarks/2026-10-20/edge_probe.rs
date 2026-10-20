//! Reproduce Canny pixel compatibility and isolated frontend timings.

#[path = "../../src/fast_edges.rs"]
mod fast_edges;

use std::{env, fs, path::PathBuf, time::Instant};

use image::{GrayImage, RgbImage, imageops};

/// Selects verification or a repeated comparison with the upstream Canny filter.
fn main() {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("verify") => verify(args.map(PathBuf::from).collect()),
        Some("bench") => benchmark(args.map(PathBuf::from).collect()),
        _ => panic!("usage: edge_probe verify|bench [JPEG paths...]"),
    }
}

/// Checks four image channels at both thresholds on every supplied scene.
fn verify(mut paths: Vec<PathBuf>) {
    if paths.is_empty() {
        for folder in [
            "scenes",
            "borders",
            "camera",
            "playmats",
            "materials",
            "art-treatments",
            "adversarial",
        ] {
            let directory = PathBuf::from("tests/fixtures").join(folder).join("images");
            for entry in fs::read_dir(directory).expect("saved fixture directory") {
                let path = entry.expect("saved fixture entry").path();
                if path.extension().is_some_and(|extension| extension == "jpg") {
                    paths.push(path);
                }
            }
        }
    }
    paths.sort();
    assert!(
        !paths.is_empty(),
        "verification requires at least one image"
    );
    let mut masks = 0_u64;
    let mut pixels = 0_u64;
    for path in &paths {
        // Compare grayscale and individual color channels at the same bounded resolution.
        let rgb = bounded_rgb(path);
        let gray = image::DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
        let planes = std::iter::once(gray).chain((0..3).map(|channel| {
            GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
                image::Luma([rgb[(x, y)][channel]])
            })
        }));
        for (channel, plane) in planes.enumerate() {
            for (low, high) in [(15.0, 45.0), (7.5, 22.5)] {
                // Require exact mask equality before counting the verified pixels.
                let reference = imageproc::edges::canny(&plane, low, high);
                let actual = fast_edges::canny(&plane, low, high);
                assert_eq!(
                    reference,
                    actual,
                    "{} channel {channel}, thresholds {low},{high}",
                    path.display(),
                );
                masks += 1;
                pixels += u64::from(plane.width()) * u64::from(plane.height());
            }
        }
    }
    println!(
        "{{\"images\":{},\"identical_masks\":{masks},\"pixels\":{pixels}}}",
        paths.len(),
    );
}

/// Measures upstream and optimized edges after warming both implementations.
fn benchmark(mut paths: Vec<PathBuf>) {
    if paths.is_empty() {
        paths = [
            "baseline-black-lotus-jpeg.jpg",
            "baseline-sol-ring-perspective-1.jpg",
            "occlusion-two-cards-1.jpg",
        ]
        .map(|file| PathBuf::from("tests/fixtures/scenes/images").join(file))
        .into();
    }
    for path in paths {
        // Check equivalence and warm both filters before alternating timed calls.
        let rgb = bounded_rgb(&path);
        let gray = image::DynamicImage::ImageRgb8(rgb).to_luma8();
        let upstream = || imageproc::edges::canny(&gray, 15.0, 45.0);
        let optimized = || fast_edges::canny(&gray, 15.0, 45.0);
        assert_eq!(upstream(), optimized());
        for _ in 0..8 {
            std::hint::black_box(upstream());
            std::hint::black_box(optimized());
        }
        let mut times = [Vec::new(), Vec::new()];
        for iteration in 0..32 {
            for index in [iteration % 2, 1 - iteration % 2] {
                // Record one filter call while alternating which implementation runs first.
                let start = Instant::now();
                std::hint::black_box(if index == 0 { upstream() } else { optimized() });
                times[index].push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        for values in &mut times {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "{}: {}x{}, upstream median {:.3} ms, optimized median {:.3} ms",
            path.display(),
            gray.width(),
            gray.height(),
            times[0][16],
            times[1][16],
        );
    }
}

/// Applies the same 640-pixel longest-side bound used by the saved corpus audit.
fn bounded_rgb(path: &PathBuf) -> RgbImage {
    let rgb = image::open(path)
        .expect("read decoded JPEG fixture")
        .to_rgb8();
    let longest = rgb.width().max(rgb.height());
    if longest <= 640 {
        return rgb;
    }
    let scale = 640.0 / f64::from(longest);
    imageops::resize(
        &rgb,
        (f64::from(rgb.width()) * scale).round() as u32,
        (f64::from(rgb.height()) * scale).round() as u32,
        imageops::FilterType::Triangle,
    )
}
