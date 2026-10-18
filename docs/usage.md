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

The [saved scene tests](testing.md) cover controlled scenes; real-camera accuracy
and video throughput have not been benchmarked.
