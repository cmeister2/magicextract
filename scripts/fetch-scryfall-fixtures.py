#!/usr/bin/env python3
"""Download a small pinned Scryfall image set once; tests never use the network."""

import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests/fixtures/scryfall/manifest.json"
CACHE = MANIFEST.parent / "cache"
USER_AGENT = "magicextract/0.0.0 (https://github.com/cmeister2/magicextract; test fixtures)"
REQUEST_INTERVAL = 0.25
MAX_IMAGE_BYTES = 2 * 1024 * 1024
IMAGE_TYPES = {
    ".jpg": ("image/jpeg", b"\xff\xd8"),
    ".png": ("image/png", b"\x89PNG\r\n\x1a\n"),
}


def fetch_fixtures(manifest_path=MANIFEST, cache_directory=CACHE):
    """Verify cached images and fetch only missing ones, stopping on HTTP errors.

    Pinned static URLs need no metadata API calls. Requests are sequential, at
    most four per second, and carry explicit User-Agent and Accept headers.
    There are no automatic retries, including on HTTP 429; Retry-After is shown
    to the caller. SHA-256 pins protect deterministic synthetic test inputs.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    cache_directory = Path(cache_directory)
    cache_directory.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    last_request = None
    for card in manifest["cards"]:
        filename = card["file"]
        if Path(filename).name != filename or Path(filename).suffix not in IMAGE_TYPES:
            raise ValueError("Fixture filenames must be plain PNG or JPEG basenames")
        content_type, signature = IMAGE_TYPES[Path(filename).suffix]
        expected_hash = card["sha256"]
        destination = cache_directory / filename
        if destination.exists():
            actual_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Cached fixture checksum mismatch: {destination}; remove it to download again")
            print(f"cached: {filename}")
            continue
        url = card["image_url"]
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc != "cards.scryfall.io":
            raise ValueError("Fixtures must use HTTPS static images from cards.scryfall.io")
        if last_request is not None:
            time.sleep(max(0.0, REQUEST_INTERVAL - (time.monotonic() - last_request)))
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": content_type}
        )
        last_request = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(MAX_IMAGE_BYTES + 1)
        except urllib.error.HTTPError as error:
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "unspecified")
                raise RuntimeError(
                    f"Scryfall rate limit reached; stopped without retrying (Retry-After: {retry_after})"
                ) from error
            raise RuntimeError(f"Fixture request failed with HTTP {error.code}; stopped without retrying") from error
        if len(data) > MAX_IMAGE_BYTES or not data.startswith(signature):
            raise ValueError(f"Expected {content_type} no larger than 2 MiB: {filename}")
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Downloaded fixture checksum mismatch: {filename}; review the source before updating the pin")
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
        downloaded += 1
        print(f"downloaded: {filename} ({len(data):,} bytes)")
    return downloaded


if __name__ == "__main__":
    try:
        count = fetch_fixtures()
        print(f"Ready; downloaded {count} image(s). Test runs make no network requests.")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Fixture fetch failed: {error}", file=sys.stderr)
        sys.exit(1)
