"""Offline regressions for fixture caching, checksums, and polite HTTP behavior."""

import hashlib
from http.client import HTTPMessage
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error


SPEC = importlib.util.spec_from_file_location(
    "fetch_scryfall", Path(__file__).with_name("fetch-scryfall-fixtures.py")
)
FETCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCHER)
JPEG = b"\xff\xd8synthetic-test-content"


class FixtureDownloaderTests(unittest.TestCase):
    """Exercise external-service behavior without making any real requests."""

    def setUp(self):
        """Create an isolated, pinned fixture manifest for every scenario."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "cache"
        self.manifest = self.root / "manifest.json"
        self.cards = [
            {
                "file": f"card-{index}.jpg",
                "image_url": f"https://cards.scryfall.io/normal/front/a/b/{index}.jpg",
                "sha256": hashlib.sha256(JPEG).hexdigest(),
            }
            for index in range(2)
        ]
        self.write_manifest()

    def write_manifest(self):
        """Write the current cases for validation by the production entry point."""
        self.manifest.write_text(json.dumps({"cards": self.cards}), encoding="utf-8")

    def test_downloads_have_headers_spacing_and_verified_cache(self):
        """Missing images are spaced apart; a rerun makes zero HTTP calls."""
        with (
            patch.object(FETCHER.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(JPEG)) as request,
            patch.object(FETCHER.time, "monotonic", return_value=10.0),
            patch.object(FETCHER.time, "sleep") as sleep,
        ):
            self.assertEqual(FETCHER.fetch_fixtures(self.manifest, self.cache), 2)
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once_with(0.25)
            for call in request.call_args_list:
                headers = dict((key.lower(), value) for key, value in call.args[0].header_items())
                self.assertIn("magicextract", headers["user-agent"])
                self.assertEqual(headers["accept"], "image/jpeg")
                self.assertEqual(call.kwargs["timeout"], 30)
            request.reset_mock()
            self.assertEqual(FETCHER.fetch_fixtures(self.manifest, self.cache), 0)
            request.assert_not_called()

    def test_rate_limit_stops_without_retrying_or_fetching_next_card(self):
        """HTTP 429 stops the entire batch and surfaces Retry-After."""
        headers = HTTPMessage()
        headers["Retry-After"] = "60"
        error = urllib.error.HTTPError(self.cards[0]["image_url"], 429, "limit", headers, None)
        with patch.object(FETCHER.urllib.request, "urlopen", side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, r"without retrying.*60"):
                FETCHER.fetch_fixtures(self.manifest, self.cache)
            request.assert_called_once()
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_png_sources_use_png_headers_and_signature(self):
        """Lossless source PNGs retain their own format rather than a JPEG label."""
        data = b"\x89PNG\r\n\x1a\nsynthetic-test-content"
        self.cards = [{
            "file": "card.png",
            "image_url": "https://cards.scryfall.io/png/front/a/b/card.png",
            "sha256": hashlib.sha256(data).hexdigest(),
        }]
        self.write_manifest()
        with patch.object(FETCHER.urllib.request, "urlopen", return_value=io.BytesIO(data)) as request:
            self.assertEqual(FETCHER.fetch_fixtures(self.manifest, self.cache), 1)
            headers = {key.lower(): value for key, value in request.call_args.args[0].header_items()}
            self.assertEqual(headers["accept"], "image/png")
        self.assertEqual((self.cache / "card.png").read_bytes(), data)

    def test_wrong_download_checksum_is_never_saved(self):
        """Changed upstream bytes cannot silently change the synthetic inputs."""
        with patch.object(FETCHER.urllib.request, "urlopen", return_value=io.BytesIO(JPEG + b"changed")):
            with self.assertRaisesRegex(ValueError, "Downloaded fixture checksum mismatch"):
                FETCHER.fetch_fixtures(self.manifest, self.cache)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_wrong_cached_checksum_is_not_requested_again(self):
        """Corruption is visible and cannot cause repeated download attempts."""
        self.cache.mkdir()
        (self.cache / self.cards[0]["file"]).write_bytes(b"corrupted")
        with patch.object(FETCHER.urllib.request, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, "Cached fixture checksum mismatch"):
                FETCHER.fetch_fixtures(self.manifest, self.cache)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
