# Usage

## Dependencies

For a local checkout:

```toml
[dependencies]
magicextract = { path = "../magicextract" }
image = { version = "0.25.10", default-features = false, features = ["jpeg", "png"] }
```

The library accepts decoded `image::DynamicImage` values and enables no codecs.
Choose image formats through your application's `image` dependency. Supply video
frames from your own decoder; each extraction is independent.

## Configuration

`extract_cards(&frame)` uses defaults: 250 × 350 pixel crops and a 5:7 aspect
ratio filter. Use `CardExtractor` for custom settings:

```rust
use magicextract::{CardExtractor, ExtractorConfig};

let extractor = CardExtractor::new(ExtractorConfig {
    output_width: 500,
    output_height: 700,
    max_detection_dimension: 1200,
    ..ExtractorConfig::default()
})?;

let cards = extractor.extract(&frame)?;
```

Keep output dimensions at 5:7 to preserve proportions. Adjust
`aspect_ratio_tolerance` and the image-relative area bounds for your capture
setup. Detection may reduce the input to `max_detection_dimension`; crops always
sample the original RGB pixels.

Use `extractor.detect(&frame)` for outlines without crops. Results contain
clockwise source-image corners and area in square pixels. They are sorted by
bounding box, top-to-bottom then left-to-right. `max_cards` retains the largest
distinct candidates before sorting. Reuse an extractor across frames.

Generate the full API reference with `cargo doc --open`.

## Camera frame latency

Use a release build and measure `detect` or `extract` on the target phone. A
15 FPS pipeline has 66.7 ms for the whole frame, including camera conversion,
card identification, and display. This library measures outlines and crops;
it does not identify card contents.

Three independent controls reduce work:

- `max_detection_dimension` bounds the working resolution.
- `detection_region` crops the input before grayscale conversion and resizing.
  Give source-pixel coordinates after applying camera orientation. Keep the
  complete card and some surrounding background inside the window.
- `detection_mode: DetectionMode::Fast` omits inference of broken outlines,
  overlap recovery, color perimeter recovery, and sleeve recovery. It keeps
  grayscale edges, threshold contours, gradient refinement, and frame expansion.
  Full mode remains the default for difficult scenes.

For example, a fixed guide in a 1280 × 720 camera buffer:

```rust
use magicextract::{CardExtractor, DetectionMode, DetectionRegion, ExtractorConfig};

let extractor = CardExtractor::new(ExtractorConfig {
    max_detection_dimension: 480,
    detection_mode: DetectionMode::Fast,
    detection_region: Some(DetectionRegion {
        x: 320,
        y: 60,
        width: 640,
        height: 600,
    }),
    ..ExtractorConfig::default()
})?;
```

Returned corners still refer to the complete source buffer, and extraction
samples its original pixels. Area fractions refer to the selected window.
The region must fit inside every frame; invalid bounds return
`Error::InvalidRegion`. The full source camera center is retained for perspective
checks, including windows away from the center.

`orientation: Some(OrientationConstraint { angle_degrees: 0.0,
tolerance_degrees: 15.0 })` accepts approximately upright portrait cards.
Angles describe the mean short-edge direction, clockwise from horizontal,
modulo 180 degrees. This filter rejects candidates before the card-count cap
and expensive recovery. It still computes the pixel edge map, so it need not
provide a large speed improvement.

Reducing resolution or enabling Fast mode changes accuracy. A fixed ROI can
also change threshold statistics and the apparent card size. Compare settings
on the same eligible scenes using the [performance benchmark](performance.md).
Use `detect` for live overlays and generate crops only when needed; `extract`
already includes detection, so calling both per frame repeats that work.

## Input preparation and limits

- Apply EXIF orientation before extraction when needed, and composite transparent
  images onto a background. Coordinates refer to the supplied pixel buffer;
  crops are RGB8.
- The shape filter checks trading card proportions, not physical dimensions.
  Oblique views also use a bounded pinhole-camera model to handle foreshortening
  and identify the physical short edge. It assumes square pixels, zero skew,
  an image-centered principal point, and a focal length between half and twice
  the image diagonal. Lens distortion and near edge-on views can fall outside
  these assumptions.
- Can infer hidden corner joins when enough straight edges remain visible.
  Occluding pixels remain in the crop. Overlapping cards, clipped edges, glare,
  heavy blur, and clutter can still cause misses or false matches.
- Crops are portrait-oriented geometrically and may be upside down. Card
  recognition and tracking belong to the downstream pipeline.
- Nearly contained outlines are suppressed as printed interior panels. A smaller
  physical card entirely inside another card's outline may also be suppressed.

The [saved scene tests](testing.md) cover controlled scenes. Decoded-frame
[performance measurements](performance.md) accompany their accuracy scores;
real-camera video throughput still needs validation on the target device.
