"""Check that genuine findings are distinguishable from a broken search tool."""

from contextlib import redirect_stderr
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "adversarial_search", Path(__file__).with_name("adversarial-search.py")
)
SEARCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEARCH)


class AdversarialSearchTests(unittest.TestCase):
    """Exercise search status and report handling without rendering or networking."""

    def test_missing_pillow_is_operational_error_and_writes_no_fixtures(self):
        """A missing authoring dependency must not masquerade as a found defect."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            with (
                patch.object(SEARCH.sys, "argv", ["search", "--cases", "1", "--output", str(output)]),
                patch.object(SEARCH, "authoring_module", side_effect=ImportError("Pillow is missing")),
                redirect_stderr(io.StringIO()) as errors,
            ):
                self.assertEqual(SEARCH.main(), 2)
                self.assertIn("Pillow", errors.getvalue())
            self.assertFalse(output.exists())

    def test_scored_failure_is_returned_for_preservation(self):
        """Exit 1 with a complete matching report is a counterexample, not a crash."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, output = root / "manifest.json", root / "evaluation.json"
            manifest.write_text(json.dumps({"scenes": [{"id": "blur"}]}))
            report = {
                "summary": {"scenes": 1, "failed": 1},
                "scenes": [{"id": "blur", "passed": False, "failures": ["missed card"]}],
            }
            output.write_text(json.dumps(report))
            process = subprocess.CompletedProcess([], 1, stdout="", stderr="")
            with patch.object(SEARCH.subprocess, "run", return_value=process):
                self.assertEqual(SEARCH.evaluate(Path("evaluator"), manifest, output), report)

    def test_wrong_scene_report_is_not_accepted_as_success(self):
        """An old or malformed evaluator cannot provide a green result for another input."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, output = root / "manifest.json", root / "evaluation.json"
            manifest.write_text(json.dumps({"scenes": [{"id": "requested"}]}))
            output.write_text(json.dumps({
                "summary": {"scenes": 1, "failed": 0},
                "scenes": [{"id": "different", "passed": True, "failures": []}],
            }))
            process = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with patch.object(SEARCH.subprocess, "run", return_value=process):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    SEARCH.evaluate(Path("evaluator"), manifest, output)

    def test_camera_reduction_preserves_view_and_world_geometry(self):
        """Reducing blur must not silently change the camera or physical card."""
        author = SimpleNamespace(
            CARD_NAMES=("card",),
            rotated_corners=lambda center, width, degrees: [[0, 0], [10, 0], [10, 14], [0, 14]],
        )
        high = SEARCH.candidate_recipe(author, 12, 7, "blur", 0.9, camera_angles=True)
        low = SEARCH.candidate_recipe(author, 12, 7, "blur", 0.0, camera_angles=True)
        self.assertEqual(high["camera"], low["camera"])
        self.assertEqual(high["cards"], low["cards"])
        self.assertGreater(high["effects"][0]["radius"], low["effects"][0]["radius"])
        tilt = SEARCH.candidate_recipe(author, 12, 8, "camera-tilt", 1.0)
        control = SEARCH.candidate_recipe(author, 12, 8, "camera-tilt", 0.0)
        self.assertEqual(tilt["camera"]["pitch_degrees"], 65.0)
        self.assertEqual(control["camera"]["pitch_degrees"], 0.0)
        self.assertEqual(tilt["camera"]["azimuth_degrees"], control["camera"]["azimuth_degrees"])
        self.assertEqual(tilt["camera"]["roll_degrees"], control["camera"]["roll_degrees"])

    def test_archived_fixtures_do_not_receive_checkout_diagnostics(self):
        """A complete search archives image annotations while reports stay in scratch."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_manifest = root / "sources.json"
            source_manifest.write_text(json.dumps({"cards": [{"file": "card.png", "sha256": "image-pin"}]}))
            binary = root / "evaluator"
            binary.write_bytes(b"test evaluator")

            def save_scene(directory, recipe, sources, seed):
                directory = Path(directory)
                filename = f"images/{recipe['id']}.jpg"
                path = directory / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test image bytes")
                return {"id": recipe["id"], "file": filename, "params": recipe}, {
                    filename: hashlib.sha256(path.read_bytes()).hexdigest(),
                }

            author = SimpleNamespace(
                CARD_NAMES=("card",),
                rotated_corners=lambda center, width, degrees: [[0, 0], [10, 0], [10, 14], [0, 14]],
                load_sources=lambda: {}, SOURCE_MANIFEST=source_manifest,
                __file__=__file__, PIL=SimpleNamespace(__version__="test"), write_scene=save_scene,
            )

            def score(binary, manifest, output):
                scenes = json.loads(Path(manifest).read_text())["scenes"]
                return {"summary": {"scenes": len(scenes), "failed": len(scenes)},
                        "scenes": [{"id": scene["id"], "passed": False, "failures": ["missed card"]}
                                   for scene in scenes]}

            options = SimpleNamespace(
                seed=12, cases=1, max_save=1, reduce_steps=0, camera_angles=False,
                output=root / "scratch", save_failures=root / "fixtures", evaluator=binary,
            )
            with (
                patch.object(SEARCH, "authoring_module", return_value=author),
                patch.object(SEARCH, "evaluate", side_effect=score),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(SEARCH.run(options), 1)
            self.assertTrue((options.output / "search-report.json").exists())
            self.assertTrue((options.output / "discovery.json").exists())
            self.assertFalse((options.save_failures / "discovery.json").exists())
            saved = (options.save_failures / "manifest.json").read_text()
            self.assertNotIn(str(SEARCH.ROOT), saved)
            self.assertNotIn("source_sha256", saved)


if __name__ == "__main__":
    unittest.main()
