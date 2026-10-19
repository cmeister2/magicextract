# magicextract

A Rust library that extracts trading card sized outlines from pictures and
decoded video frames. Returns perspective-corrected RGB crops and their corners
in the original image. Pure Rust, with no OpenCV or model weights.

Requires **Rust 1.90.0+**.

[Input and output examples](https://cmeister2.github.io/magicextract/) include
successful matches, missed cards, wrong crops, and false matches.

```rust,no_run
use magicextract::{extract_cards, image};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame = image::open("table.jpg")?;
    for (index, card) in extract_cards(&frame)?.iter().enumerate() {
        card.image.save(format!("card-{index}.png"))?;
    }
    Ok(())
}
```

Enable JPEG/PNG support on your application's `image` dependency; see
[setup and configuration](docs/usage.md).

Try the included example:

```sh
cargo run --release --example extract -- table.jpg crops/
```

[Usage and limitations](docs/usage.md) · [Testing](docs/testing.md) ·
[Releases](docs/releasing.md) · [MIT license](LICENSE)
