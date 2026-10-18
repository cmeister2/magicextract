#!/usr/bin/env python3
"""Explicitly author persistent JPEG scenes; tests never call this renderer.

Authoring dependency: Pillow==10.2.0. Cached PNG sources are downloaded separately
with fetch-scryfall-fixtures.py. Run with --write; --force permits replacement.
Saved JPEG/reference bytes and their hashes are the authoritative test inputs.

Reusable API for adversarial searches:
  sources = load_sources()
  image, cards = render_scene(recipe, sources, seed=20261018)
  entry, hashes = write_scene(output, recipe, sources, seed=20261018)
Recipes declare cards, background, optional occluders and image-wide effects.

Image backgrounds use {"kind": "image", "file": "repository/relative.png",
"sha256": "<64 hexadecimal characters>"}; the pinned bitmap is converted to RGB
and center-fitted to SIZE with LANCZOS. No network access is involved.
Any background may add "overlays": [{"kind": "rectangle", "bounds": [x0, y0,
x1, y1], "color": [r, g, b, alpha], "width": 2}] for printed zone outlines.
"rounded_rectangle" also accepts "radius"; "line" uses "points": [[x, y], ...].
Overlay alpha is optional, coordinates are scene pixels, and overlays precede cards.
"""

import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import sys

import PIL
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageMath, ImageOps, features


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRECTORY = ROOT / "tests/fixtures/scryfall/cache"
SOURCE_MANIFEST = ROOT / "tests/fixtures/scryfall/manifest.json"
DEFAULT_OUTPUT = ROOT / "tests/fixtures/scenes"
DEFAULT_SEED = 20261018
SIZE = (640, 520)
REFERENCE_SIZE = (250, 350)
CARD_NAMES = ("lightning-bolt", "counterspell", "black-lotus", "sol-ring")
AUTHORING_PILLOW_VERSION = "10.2.0"


def digest(path):
    """Hash saved bytes rather than decoded pixels."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sources(source_directory=SOURCE_DIRECTORY):
    """Load RGBA PNG faces without removing their transparent rounded corners."""
    sources = {}
    for name in CARD_NAMES:
        path = Path(source_directory) / f"{name}.png"
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected an original PNG source: {path}")
            sources[name] = image.convert("RGBA")
    return sources


def rotated_corners(center, width, degrees):
    """Return exact clockwise trading-card corners, starting on a short edge."""
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return [
        [center[0] + x * cosine - y * sine, center[1] + x * sine + y * cosine]
        for x, y in [(-width / 2, -width * 0.7), (width / 2, -width * 0.7),
                     (width / 2, width * 0.7), (-width / 2, width * 0.7)]
    ]


def solve_linear(matrix, values):
    """Solve the small homography system with partial-pivot Gauss-Jordan."""
    augmented = [list(row) + [value] for row, value in zip(matrix, values)]
    size = len(augmented)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("Cannot map a degenerate quadrilateral")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[column])]
    return tuple(row[-1] for row in augmented)


def homography(destination, source):
    """Compute Pillow's inverse projective map from destination to source points."""
    matrix, values = [], []
    for (x, y), (u, v) in zip(destination, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        values.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.append(v)
    return solve_linear(matrix, values)


def project(source, corners, size=SIZE):
    """Project source RGBA pixels and alpha into scene coordinates."""
    # Pillow's projective bicubic sampler does not integrate a large minification
    # footprint. Band-limit source detail first so small photographed faces do
    # not acquire aliases absent from the clean reference image.
    edge_lengths = [math.dist(corners[index], corners[(index + 1) % 4]) for index in range(4)]
    filtered_size = (min(source.width, max(2, math.ceil(max(edge_lengths[0], edge_lengths[2])))),
                     min(source.height, max(2, math.ceil(max(edge_lengths[1], edge_lengths[3])))))
    if filtered_size != source.size:
        source = source.resize(filtered_size, Image.Resampling.LANCZOS)
    width, height = source.size
    source_corners = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    return source.transform(
        size, Image.Transform.PERSPECTIVE, homography(corners, source_corners),
        resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0),
    )


def rectify_reference(scene, corners):
    """Rectify the final decoded JPEG with independent known-truth geometry."""
    width, height = REFERENCE_SIZE
    destination = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    return scene.transform(
        REFERENCE_SIZE, Image.Transform.PERSPECTIVE, homography(destination, corners),
        resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
    )


def background_overlays(image, overlays):
    """Print optional translucent zone outlines or lines before placing cards."""
    if not overlays:
        return image
    layer = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(layer)
    for overlay in overlays:
        color = tuple(overlay.get("color", (235, 235, 235, 160)))
        if len(color) == 3:
            color += (255,)
        width = overlay.get("width", 2)
        if not isinstance(width, int) or width < 1:
            raise ValueError("Printed background overlay widths must be positive integers")
        kind = overlay["kind"]
        if kind == "rectangle":
            draw.rectangle(overlay["bounds"], outline=color, width=width)
        elif kind == "rounded_rectangle":
            draw.rounded_rectangle(overlay["bounds"], radius=overlay.get("radius", 12),
                                   outline=color, width=width)
        elif kind == "line":
            draw.line([tuple(point) for point in overlay["points"]], fill=color, width=width)
        else:
            raise ValueError(f"Unknown printed background overlay: {kind}")
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def background_image(parameters, seed):
    """Create a procedural table or a hash-pinned local bitmap with optional print."""
    kind = parameters.get("kind", "solid")
    color = tuple(parameters.get("color", (75, 115, 90)))
    overlays = parameters.get("overlays", [])
    if kind == "image":
        filename, expected_hash = parameters.get("file"), parameters.get("sha256")
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
            raise ValueError("Image backgrounds require a repository-relative file path")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash.lower()
        ):
            raise ValueError("Image backgrounds require a 64-character hexadecimal sha256 pin")
        path = (ROOT / filename).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Image background files must stay inside the repository")
        if digest(path) != expected_hash.lower():
            raise ValueError(f"Image background checksum mismatch: {filename}")
        with Image.open(path) as bitmap:
            image = ImageOps.fit(bitmap.convert("RGB"), SIZE,
                                 method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        return background_overlays(image, overlays)
    if kind == "solid":
        return background_overlays(Image.new("RGB", SIZE, color), overlays)
    image = Image.new("RGB", SIZE)
    pixels = image.load()
    rng = random.Random(seed)
    for y in range(SIZE[1]):
        for x in range(SIZE[0]):
            if kind == "gradient":
                offset = (x + y) // 40
            elif kind == "checker":
                offset = (18 if (x // 27 + y // 27) % 2 else -18)
            elif kind == "wood":
                offset = int(12 * math.sin(y / 5 + 0.9 * math.sin(x / 48))) + rng.randrange(-4, 5)
            else:
                raise ValueError(f"Unknown background: {kind}")
            pixels[x, y] = tuple(max(0, min(255, channel + offset)) for channel in color)
    return background_overlays(image, overlays)


def occluder_layer(parameters):
    """Draw a specified object onto an otherwise transparent scene layer."""
    layer = Image.new("RGBA", SIZE)
    draw = ImageDraw.Draw(layer)
    color = tuple(parameters.get("color", (120, 83, 58)))
    if len(color) == 3:
        color += (255,)
    kind = parameters["kind"]
    if kind == "rectangle":
        draw.rectangle(parameters["bounds"], fill=color)
    elif kind == "ellipse":
        draw.ellipse(parameters["bounds"], fill=color)
    elif kind == "rounded_rectangle":
        draw.rounded_rectangle(parameters["bounds"], radius=parameters.get("radius", 20), fill=color)
    elif kind == "polygon":
        draw.polygon([tuple(point) for point in parameters["points"]], fill=color)
    elif kind == "line":
        draw.line([tuple(point) for point in parameters["points"]], fill=color,
                  width=parameters.get("width", 5))
    else:
        raise ValueError(f"Unknown occluder: {kind}")
    return layer


def apply_effect(image, effect, seed):
    """Apply declared camera/lighting effects without changing geometric truth."""
    kind = effect["kind"]
    if kind == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(effect["radius"]))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(effect["factor"])
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(effect["factor"])
    if kind == "motion_blur":
        length = int(effect["length"])
        if length <= 0 or length % 2 == 0:
            raise ValueError("Motion-blur lengths must be positive odd integers")
        horizontal = effect.get("direction", "horizontal") == "horizontal"
        padding = length // 2
        width, height = image.size
        if horizontal:
            extended = Image.new("RGB", (width + 2 * padding, height))
            extended.paste(image, (padding, 0))
            if padding:
                extended.paste(image.crop((0, 0, 1, height)).resize((padding, height)), (0, 0))
                extended.paste(image.crop((width - 1, 0, width, height)).resize((padding, height)),
                               (padding + width, 0))
        else:
            extended = Image.new("RGB", (width, height + 2 * padding))
            extended.paste(image, (0, padding))
            if padding:
                extended.paste(image.crop((0, 0, width, 1)).resize((width, padding)), (0, 0))
                extended.paste(image.crop((0, height - 1, width, height)).resize((width, padding)),
                               (0, padding + height))
        totals = [Image.new("F", image.size) for _ in range(3)]
        for displacement in range(-padding, padding + 1):
            left = padding + displacement if horizontal else 0
            top = 0 if horizontal else padding + displacement
            shifted = extended.crop((left, top, left + width, top + height))
            totals = [ImageMath.eval("a + b", a=total, b=channel.convert("F"))
                      for total, channel in zip(totals, shifted.split())]
        return Image.merge("RGB", tuple(ImageMath.eval("a / n", a=total, n=length).convert("L")
                                        for total in totals))
    if kind == "jpeg_roundtrip":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=effect["quality"], subsampling=0)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    if kind == "noise":
        amplitude = effect.get("amplitude", 4)
        pixels = bytearray(image.tobytes())
        for y in range(SIZE[1]):
            for x in range(SIZE[0]):
                offset = ((x * 73 ^ y * 151 ^ effect.get("seed_offset", 0)) % (2 * amplitude + 1)) - amplitude
                index = (y * SIZE[0] + x) * 3
                for channel in range(3):
                    pixels[index + channel] = max(0, min(255, pixels[index + channel] + offset))
        return Image.frombytes("RGB", SIZE, bytes(pixels))
    if kind == "lighting":
        pixels = bytearray(image.tobytes())
        tint = effect.get("tint", (1.0, 1.0, 1.0))
        for y in range(SIZE[1]):
            for x in range(SIZE[0]):
                if effect.get("gradient"):
                    fraction = (x + y * 0.35) / (SIZE[0] + SIZE[1] * 0.35)
                    factor = effect.get("minimum", 0.35) + fraction * (effect.get("maximum", 1.0) - effect.get("minimum", 0.35))
                else:
                    factor = effect.get("factor", 1.0)
                if effect.get("glare"):
                    cx, cy, radius = effect["glare"]
                    glare = max(0, 1 - math.hypot(x - cx, y - cy) / radius) ** 2
                else:
                    glare = 0
                index = (y * SIZE[0] + x) * 3
                for channel in range(3):
                    value = pixels[index + channel] * factor * tint[channel]
                    pixels[index + channel] = max(0, min(255, round(value + glare * (255 - value))))
        return Image.frombytes("RGB", SIZE, bytes(pixels))
    raise ValueError(f"Unknown scene effect: {kind}")


def render_scene(recipe, sources, seed=DEFAULT_SEED):
    """Render a declared scene and record each card's full outline and visibility.

    Returns (RGB_image, cards). Occluders are composited after cards. Card metadata
    always describes the complete pre-occlusion rectangle, including hidden corners.
    """
    scene = background_image(recipe.get("background", {}), seed).convert("RGBA")
    for shadow in recipe.get("shadows", []):
        shadow_layer = Image.new("RGBA", SIZE)
        draw = ImageDraw.Draw(shadow_layer)
        draw.polygon([tuple(point) for point in shadow["corners"]],
                     fill=tuple(shadow.get("color", (0, 0, 0, 70))))
        scene = Image.alpha_composite(scene, shadow_layer.filter(ImageFilter.GaussianBlur(shadow.get("radius", 3))))
    visible_masks, original_totals, metadata = [], [], []
    for card in recipe.get("cards", []):
        layer = project(sources[card["source"]], card["corners"])
        alpha = layer.getchannel("A")
        inverse_alpha = ImageChops.invert(alpha)
        visible_masks = [ImageChops.multiply(mask, inverse_alpha) for mask in visible_masks]
        visible_masks.append(alpha.copy())
        if all(0 <= x < SIZE[0] and 0 <= y < SIZE[1] for x, y in card["corners"]):
            complete_alpha = alpha
        else:
            # Include the original source pixels projected beyond the frame when
            # measuring visibility; the expected corners still retain full truth.
            left = math.floor(min(x for x, _ in card["corners"])) - 1
            top = math.floor(min(y for _, y in card["corners"])) - 1
            right = math.ceil(max(x for x, _ in card["corners"])) + 2
            bottom = math.ceil(max(y for _, y in card["corners"])) + 2
            complete_alpha = project(sources[card["source"]],
                                     [[x - left, y - top] for x, y in card["corners"]],
                                     (right - left, bottom - top)).getchannel("A")
        original_totals.append(sum(value * count for value, count in enumerate(complete_alpha.histogram())))
        metadata.append({"source": card["source"], "corners": card["corners"]})
        scene = Image.alpha_composite(scene, layer)
    for occluder in recipe.get("occluders", []):
        layer = occluder_layer(occluder)
        inverse_alpha = ImageChops.invert(layer.getchannel("A"))
        visible_masks = [ImageChops.multiply(mask, inverse_alpha) for mask in visible_masks]
        scene = Image.alpha_composite(scene, layer)
    for card, denominator, visible in zip(metadata, original_totals, visible_masks):
        numerator = sum(value * count for value, count in enumerate(visible.histogram()))
        card["visible_fraction"] = round(numerator / denominator, 6) if denominator else 0.0
    scene = scene.convert("RGB")
    for effect in recipe.get("effects", []):
        scene = apply_effect(scene, effect, seed)
    return scene, metadata


def write_scene(output_directory, recipe, sources, seed=DEFAULT_SEED, overwrite=False):
    """Persist one JPEG and optional per-card appearance references; return metadata."""
    directory = Path(output_directory)
    identifier = recipe["id"]
    if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in identifier):
        raise ValueError("Scene IDs must be lowercase filename-safe identifiers")
    relative = f"images/{identifier}.jpg"
    destination = directory / relative
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace {destination}; pass overwrite=True explicitly")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene, cards = render_scene(recipe, sources, seed)
    scene.save(destination, format="JPEG", quality=recipe.get("jpeg_quality", 94),
               subsampling=0, optimize=False, progressive=False)
    hashes = {relative: digest(destination)}
    if recipe.get("appearance_reference", False):
        with Image.open(destination) as decoded:
            decoded = decoded.convert("RGB")
            for index, card in enumerate(cards):
                reference = f"references/{identifier}-{index}-{card['source']}.png"
                reference_path = directory / reference
                if reference_path.exists() and not overwrite:
                    raise FileExistsError(f"Refusing to replace {reference_path}")
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                rectify_reference(decoded, card["corners"]).save(reference_path, format="PNG")
                card["reference"] = reference
                hashes[reference] = digest(reference_path)
    entry = {
        "id": identifier, "file": relative, "tier": recipe.get("tier", "challenge"),
        "description": recipe["description"], "tags": recipe.get("tags", []),
        "cards": cards, "corner_tolerance": 8.0,
        "pixel_mae_limit": recipe.get("pixel_mae_limit", 26.0 if len(cards) > 1 else 22.0),
        "params": deepcopy(recipe),
    }
    return entry, hashes


def recipe(identifier, name="lightning-bolt", corners=None, **kwargs):
    """Construct a complete declarative single-card recipe."""
    result = {
        "id": identifier, "tier": "regression", "description": identifier.replace("-", " "),
        "tags": [], "background": {"kind": "solid", "color": [75, 115, 90]},
        "cards": [{"source": name, "corners": corners or rotated_corners((320, 260), 180, 0)}],
    }
    result.update(kwargs)
    return result


def baseline_recipes():
    """Preserve the earlier 57 scene families without a runtime renderer."""
    scenes = []
    backgrounds = [("green", [75, 115, 90]), ("dark", [24, 60, 39]), ("light", [210, 210, 205])]
    perspectives = [
        [[230, 120], [405, 140], [420, 390], [210, 370]],
        [[240, 120], [410, 100], [390, 380], [220, 360]],
    ]
    for name in CARD_NAMES:
        for label, color in backgrounds:
            scenes.append(recipe(f"baseline-{name}-background-{label}", name,
                                 background={"kind": "solid", "color": color}, tags=["baseline", "background"]))
        for angle in [-42, -13, 27, 73, 155]:
            label = f"minus{abs(angle)}" if angle < 0 else str(angle)
            scenes.append(recipe(f"baseline-{name}-rotation-{label}", name,
                                 rotated_corners((320, 260), 180, angle), tags=["baseline", "rotation"]))
        for index, corners in enumerate(perspectives):
            scenes.append(recipe(f"baseline-{name}-perspective-{index + 1}", name, corners,
                                 tags=["baseline", "perspective"]))
        corners = rotated_corners((320, 260), 180, 17)
        scenes.append(recipe(f"baseline-{name}-soft-shadow", name, corners,
                             background={"kind": "gradient", "color": [75, 115, 90]},
                             shadows=[{"corners": [[x + 6, y + 8] for x, y in corners], "radius": 2,
                                       "color": [30, 45, 35, 75]}], tags=["baseline", "shadow", "lighting"]))
        for label, effect in [("blur", {"kind": "gaussian_blur", "radius": 0.65}),
                              ("jpeg", {"kind": "jpeg_roundtrip", "quality": 78})]:
            scenes.append(recipe(f"baseline-{name}-{label}", name,
                                 rotated_corners((320, 260), 180, -19),
                                 effects=[effect], tags=["baseline", label]))
        scenes.append(recipe(f"baseline-{name}-sensor-noise", name,
                             rotated_corners((320, 260), 180, 8),
                             effects=[{"kind": "noise", "amplitude": 4}], tags=["baseline", "noise"]))
    placements = [((160, 135), -8), ((470, 135), 12), ((160, 385), 9), ((470, 385), -11)]
    scenes.append(recipe("baseline-four-cards", cards=[
        {"source": name, "corners": rotated_corners(center, 130, angle)}
        for name, (center, angle) in zip(CARD_NAMES, placements)
    ], background={"kind": "gradient", "color": [75, 115, 90]},
        tags=["baseline", "multiple-cards"], pixel_mae_limit=26.0))
    assert len(scenes) == 57
    return scenes


def challenge_recipes():
    """Describe feasible partial outlines and explicit harder camera failure cases."""
    scenes = []
    corners = rotated_corners((320, 260), 180, 0)
    bites = {
        "top-left": [220, 124, 278, 178], "top-right": [362, 124, 420, 178],
        "bottom-right": [362, 342, 420, 396], "bottom-left": [220, 342, 278, 396],
    }
    for index, (label, bounds) in enumerate(bites.items()):
        scenes.append(recipe(f"occlusion-corner-{label}", CARD_NAMES[index], corners,
                             occluders=[{"kind": "rectangle", "bounds": bounds, "color": [120, 83, 58]}],
                             tags=["occlusion", "hidden-corner", "full-outline-inference"],
                             description=f"A small object hides the {label} corner; the other three corners and long edges determine the full outline."))
    tabs = {"top": [295, 120, 345, 170], "right": [375, 228, 430, 292],
            "bottom": [295, 350, 345, 400], "left": [210, 228, 265, 292]}
    for index, (label, bounds) in enumerate(tabs.items()):
        scenes.append(recipe(f"occlusion-edge-{label}", CARD_NAMES[index], corners,
                             occluders=[{"kind": "rectangle", "bounds": bounds, "color": [154, 105, 74]}],
                             tags=["occlusion", "edge-tab", "full-outline-inference"],
                             description=f"A foreground tab interrupts the {label} edge while all original card corners remain visible."))
    scenes.append(recipe("occlusion-finger-corner", "lightning-bolt", corners,
                         occluders=[{"kind": "ellipse", "bounds": [193, 94, 270, 174], "color": [188, 139, 107]}],
                         tags=["occlusion", "finger", "hidden-corner", "full-outline-inference"],
                         description="A rounded fingertip covers one corner with three corners and the remaining sides visible."))
    scenes.append(recipe("occlusion-pen-across-middle", "counterspell", corners,
                         occluders=[{"kind": "line", "points": [[175, 245], [468, 283]], "width": 13,
                                     "color": [25, 31, 35]}],
                         tags=["occlusion", "disconnected-outline", "full-outline-inference"],
                         description="A narrow diagonal pen divides the face; visible edge segments still determine the original rectangle."))
    for index, center in enumerate([(420, 325), (435, 240)]):
        scenes.append(recipe(f"occlusion-two-cards-{index + 1}", cards=[
            {"source": "black-lotus", "corners": rotated_corners((285, 250), 180, -8)},
            {"source": "sol-ring", "corners": rotated_corners(center, 170, 14)},
        ], tags=["occlusion", "multiple-cards", "full-outline-inference"], pixel_mae_limit=22.0,
            description="Two partially overlapping real cards require retaining both complete source outlines."))
    scenes.append(recipe("lighting-directional-shadow", "black-lotus",
                         rotated_corners((320, 260), 180, 17),
                         effects=[{"kind": "lighting", "gradient": True, "minimum": 0.20, "maximum": 1.15}],
                         tags=["lighting", "directional-shadow"],
                         description="A strong illumination gradient darkens part of the scene and card."))
    scenes.append(recipe("lighting-warm-dim", "sol-ring", rotated_corners((320, 260), 180, -12),
                         effects=[{"kind": "lighting", "factor": 0.60, "tint": [1.35, 0.93, 0.65]}],
                         tags=["lighting", "white-balance", "dim"],
                         description="Warm dim illumination changes the observed artwork and border contrast."))
    scenes.append(recipe("lighting-local-glare", "lightning-bolt", rotated_corners((320, 260), 180, 9),
                         effects=[{"kind": "lighting", "glare": [389, 192, 110]}],
                         tags=["lighting", "glare"],
                         description="A local clipped highlight fades part of a card edge and artwork."))
    for radius in [1.3, 2.1]:
        scenes.append(recipe(f"blur-gaussian-{str(radius).replace('.', '-')}", "counterspell",
                             rotated_corners((320, 260), 180, -19),
                             effects=[{"kind": "gaussian_blur", "radius": radius}],
                             tags=["blur", "defocus"], description=f"Gaussian defocus with radius {radius} pixels."))
    scenes.append(recipe("blur-motion-horizontal", "sol-ring", rotated_corners((320, 260), 180, 21),
                         effects=[{"kind": "motion_blur", "length": 9}], tags=["blur", "motion"],
                         description="A nine-pixel horizontal camera-motion exposure smears the full scene."))
    scenes.append(recipe("low-contrast-dark-surface", "black-lotus", rotated_corners((320, 260), 180, 5),
                         background={"kind": "solid", "color": [35, 37, 35]},
                         effects=[{"kind": "brightness", "factor": 0.38}], tags=["low-contrast", "lighting"],
                         description="A dark-bordered card lies on a dim surface of similar brightness."))
    for kind, color in [("checker", [86, 108, 93]), ("wood", [123, 87, 54])]:
        scenes.append(recipe(f"clutter-{kind}-surface", "lightning-bolt",
                             rotated_corners((320, 260), 180, -23),
                             background={"kind": kind, "color": color},
                             tags=["clutter", "texture"], description=f"Card edges compete with a {kind} tabletop texture."))
    for index, shape in enumerate([
        [[274, 150], [382, 122], [435, 395], [195, 365]],
        [[250, 180], [411, 151], [415, 290], [229, 350]],
    ]):
        scenes.append(recipe(f"perspective-steep-{index + 1}", "counterspell", shape,
                             tags=["perspective", "foreshortening"],
                             description="Stronger projective foreshortening challenges apparent aspect-ratio filtering."))
    scenes.append(recipe("frame-clipped-left", "sol-ring", rotated_corners((70, 260), 180, -8),
                         tags=["occlusion", "out-of-frame", "full-outline-inference"],
                         description="The frame hides part of the left edge; the expected outline records its complete source rectangle."))
    supported_occlusions = {"occlusion-corner-top-left", "occlusion-edge-top", "occlusion-finger-corner"}
    for scene in scenes:
        scene["tier"] = "regression" if scene["id"] in supported_occlusions else "challenge"
        scene["appearance_reference"] = True
    assert len(scenes) == 24
    return scenes


def negative_recipes():
    """Keep zero-card ground truth for plausible distractor objects."""
    definitions = [
        ("empty-table", [], "An empty softly illuminated tabletop.", []),
        ("poker-chip", [{"kind": "ellipse", "bounds": [235, 175, 405, 345], "color": [210, 47, 47]},
                        {"kind": "ellipse", "bounds": [255, 195, 385, 325], "color": [236, 224, 208]}],
         "A round poker chip has no four-corner card outline.", ["round-object"]),
        ("finger", [{"kind": "rounded_rectangle", "bounds": [285, 180, 355, 519], "radius": 35,
                     "color": [186, 139, 106]}], "A rounded finger shape enters from the edge.", ["finger"]),
        ("phone", [{"kind": "rounded_rectangle", "bounds": [263, 130, 377, 390], "radius": 20,
                    "color": [22, 26, 28]},
                   {"kind": "rounded_rectangle", "bounds": [271, 145, 369, 373], "radius": 12,
                    "color": [93, 119, 139]}], "A tall phone has a non-card aspect ratio.", ["wrong-aspect"]),
        ("square-paper", [{"kind": "rectangle", "bounds": [230, 170, 410, 350], "color": [235, 231, 220]}],
         "A square paper sheet is outside the trading-card proportion filter.", ["wrong-aspect"]),
        ("triangle-paper", [{"kind": "polygon", "points": [[230, 134], [410, 134], [320, 386]],
                             "color": [235, 231, 220]}],
         "A triangular sheet shares card-shaped bounds but lacks four corners.", ["non-quadrilateral"]),
        ("deep-notched-paper", [{"kind": "polygon", "points": [[230, 134], [410, 134], [410, 200],
                                  [280, 200], [280, 320], [410, 320], [410, 386], [230, 386]],
                                 "color": [235, 231, 220]}],
         "A deeply notched object must not become a complete card merely through its convex hull.", ["concave-object"]),
        ("receipt", [{"kind": "rectangle", "bounds": [278, 120, 362, 400], "color": [235, 231, 220]}],
         "A narrow receipt has a rectangular outline with the wrong proportions.", ["wrong-aspect"]),
    ]
    return [recipe(f"negative-{name}", cards=[], occluders=occluders, tags=["negative"] + tags,
                   description=description, pixel_mae_limit=None)
            for name, occluders, description, tags in definitions]


def contact_sheet(output_directory, entries, destination):
    """Write an inspection sheet outside the versioned fixture directory."""
    width, height, columns = 192, 185, 6
    sheet = Image.new("RGB", (width * columns, height * math.ceil(len(entries) / columns)), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    for index, entry in enumerate(entries):
        x, y = index % columns * width, index // columns * height
        with Image.open(Path(output_directory) / entry["file"]) as image:
            image.thumbnail((188, 153))
            sheet.paste(image, (x + 2, y + 2))
        label = entry["id"]
        draw.text((x + 3, y + 155), label[:29], fill=(0, 0, 0))
        if len(label) > 29:
            draw.text((x + 3, y + 168), label[29:58], fill=(0, 0, 0))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=90)


def build(output_directory, seed, force=False, only=None):
    """Persist the corpus from validated PNG sources with provenance and hashes."""
    if PIL.__version__ != AUTHORING_PILLOW_VERSION:
        raise ValueError(f"Authoring requires Pillow=={AUTHORING_PILLOW_VERSION}; found {PIL.__version__}")
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()) and not force:
        raise FileExistsError(f"{output_directory} already contains files; use --force for an intentional rebuild")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    provenance = {}
    for record in source_manifest["cards"]:
        path = SOURCE_DIRECTORY / record["file"]
        if path.suffix != ".png" or digest(path) != record["sha256"]:
            raise ValueError(f"Expected the pinned original PNG source: {path}")
        provenance[path.stem] = record
    sources = load_sources()
    (output_directory / "references").mkdir(parents=True, exist_ok=True)
    hashes = {}
    references = {}
    for name, image in sources.items():
        relative = f"references/{name}.png"
        reference = Image.new("RGB", image.size, (0, 0, 0))
        reference.paste(image, mask=image.getchannel("A"))
        reference.resize(REFERENCE_SIZE, Image.Resampling.LANCZOS).save(output_directory / relative, format="PNG")
        hashes[relative] = digest(output_directory / relative)
        references[name] = relative
    recipes = baseline_recipes() + challenge_recipes() + negative_recipes()
    if only:
        recipes = [item for item in recipes if item["id"] in only]
        absent = set(only) - {item["id"] for item in recipes}
        if absent:
            raise ValueError(f"Unknown scene IDs: {sorted(absent)}")
    entries = []
    for scene_recipe in recipes:
        entry, scene_hashes = write_scene(output_directory, scene_recipe, sources, seed=seed, overwrite=force)
        hashes.update(scene_hashes)
        entries.append(entry)
        print(f"saved {entry['id']}", flush=True)
    manifest = {
        "format_version": 1,
        "generator": {"script": "scripts/build-scene-fixtures.py", "script_sha256": digest(__file__),
                      "pillow": PIL.__version__,
                      "libjpeg": features.version("jpg"), "seed": seed, "size": list(SIZE),
                      "jpeg_quality": 94, "jpeg_subsampling": 0,
                      "projection_antialias": "RGBA LANCZOS prefilter to maximum projected opposite-edge lengths",
                      "source_alpha": "preserved", "authoritative_inputs": "saved JPEG and reference PNG bytes"},
        "sources": provenance, "references": references, "sha256": hashes, "scenes": entries,
    }
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entries


def main():
    """Require an explicit authoring action and report where saved inputs reside."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="explicitly authorize fixture creation")
    parser.add_argument("--force", action="store_true", help="explicitly replace an existing corpus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--only", nargs="+", help="author selected scene IDs into a separate output directory")
    parser.add_argument("--contact-sheet", type=Path, default=ROOT / "target/scene-preview/contact-sheet.jpg")
    options = parser.parse_args()
    if not options.write:
        parser.error("This is an offline authoring tool; pass --write to create saved fixture files")
    entries = build(options.output, options.seed, force=options.force, only=options.only)
    contact_sheet(options.output, entries, options.contact_sheet)
    print(f"Saved {len(entries)} scenes; manifest: {options.output / 'manifest.json'}")
    print(f"Contact sheet: {options.contact_sheet}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"Scene authoring failed: {error}", file=sys.stderr)
        sys.exit(1)
