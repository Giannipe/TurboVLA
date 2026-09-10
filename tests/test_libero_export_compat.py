"""CPU-only tests for narrowly allowlisted public release containers."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from turbovla.evaluation import policy


class ExportCompatibilityTests(unittest.TestCase):
    def test_prefers_ema_even_when_raw_is_present(self):
        ema, raw = {"ema": 1}, {"raw": 2}
        self.assertIs(policy._checkpoint_state_dict(
            {"ema_model_state_dict": ema, "model_state_dict": raw}), ema)

    def test_rejects_non_mapping(self):
        with self.assertRaises(TypeError):
            policy._checkpoint_state_dict(None)

    def test_rejects_raw_without_artifact_verification(self):
        with self.assertRaises(KeyError):
            policy._checkpoint_state_dict({"model_state_dict": {}})

    def test_rejects_unknown_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.pth"
            path.write_bytes(b"not the official checkpoint")
            with self.assertRaises(KeyError):
                policy._checkpoint_state_dict({"model_state_dict": {}}, checkpoint_path=str(path))

    def test_accepts_allowlisted_export_without_modifying_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export.pth"
            contents = b"test fixture"
            path.write_bytes(contents)
            state = {"tensor": object()}
            with patch.object(policy, "_RELEASED_LIBERO_EXPORT_SHA256", hashlib.sha256(contents).hexdigest()):
                self.assertIs(policy._checkpoint_state_dict(
                    {"model_state_dict": state}, checkpoint_path=str(path)), state)

    def test_accepts_verified_legacy_export_without_converting_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            contents = b"legacy test fixture"
            path.write_bytes(contents)
            state = {"tensor": object()}
            with patch.object(policy, "_LEGACY_LIBERO_EXPORTS_SHA256",
                              {hashlib.sha256(contents).hexdigest(): "spatial"}):
                self.assertIs(policy._checkpoint_state_dict(
                    {"model_state_dict": state}, checkpoint_path=str(path)), state)

    def test_legacy_filename_and_metadata_are_not_sufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spatial.pth"
            path.write_bytes(b"unverified weights")
            with self.assertRaises(KeyError):
                policy._checkpoint_state_dict(
                    {"model_state_dict": {}, "suite": "spatial", "model_name": "TurboVLA"},
                    checkpoint_path=str(path))


if __name__ == "__main__":
    unittest.main()
