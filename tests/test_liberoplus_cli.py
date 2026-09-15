"""CPU-only LIBERO+ isolation, default protocol and archive validation."""
import io
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cluster/_internal"))
import assets
import evaluate
import liberoplus


class LiberoPlusTests(unittest.TestCase):
    def test_extraction_manifest_checks_count_bytes_and_paths(self):
        checksum = assets.HASH_LOCKS["simulator_assets/liberoplus/assets.zip"]
        valid = {"zip_sha256": checksum, "files": {"assets/scene.xml": 3}}
        bad_files = ({}, {"assets/scene.xml": 2}, {"assets/scene.xml": -1},
                     {"assets/scene.xml": True}, {"": 3}, {"assets": 3},
                     {"assets/../escape": 3}, {"assets\\escape": 3}, {"assets//scene.xml": 3})
        with patch.object(liberoplus, "EXTRACTED_FILE_COUNT", 1), \
             patch.object(liberoplus, "EXTRACTED_BYTES", 3):
            self.assertEqual(liberoplus.extraction_files(valid), valid["files"])
            for files in bad_files:
                with self.subTest(files=files), self.assertRaises(ValueError):
                    liberoplus.extraction_files(dict(valid, files=files))
            with self.assertRaises(ValueError):
                liberoplus.extraction_files(dict(valid, zip_sha256="incorrect"))

    def test_prepare_lock_only_for_writes(self):
        with patch.object(assets, "locked") as lock, patch.object(liberoplus, "_prepare", return_value={}) as prepare:
            liberoplus.prepare(Path("/store"), download=False)
            lock.assert_not_called()
            prepare.assert_called_once_with(Path("/store"), download=False)
            prepare.reset_mock()
            liberoplus.prepare(Path("/store"), download=True)
            lock.assert_called_once_with(Path("/store/.prepare_liberoplus.lock"))
            lock.return_value.__enter__.assert_called_once()
            prepare.assert_called_once_with(Path("/store"), download=True)

    def test_env_alias_guard_does_not_block_unrelated_benchmarks(self):
        import os
        env = dict(os.environ, TURBOVLA_LIBEROPLUS_ENV="turbovla-libero", TURBOVLA_LIBERO_ENV="turbovla-libero")
        script = str(ROOT / "scripts/cluster/envs.sh")
        normal = subprocess.run(["bash", script, "--benchmark", "libero", "--dry-run"], env=env,
                                capture_output=True, text=True)
        plus = subprocess.run(["bash", script, "--benchmark", "liberoplus", "--dry-run"], env=env,
                              capture_output=True, text=True)
        self.assertEqual(normal.returncode, 0, normal.stderr)
        self.assertNotEqual(plus.returncode, 0)
        self.assertIn("distinct environment", plus.stderr)

    def test_plus_dry_run_command_matches_public_evaluator_parser(self):
        from dataclasses import fields
        # Load the real parser module without importing the policy or simulator.
        import importlib.util
        spec = importlib.util.spec_from_file_location("liberoplus_rollout_test", ROOT / "third_party/vla_adapter/vla_adapter/rollout.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)
            args = evaluate.parser().parse_args(["--benchmark", "liberoplus", "--checkpoint", "/tmp/model.pth"])
            cfg = evaluate.libero_config(args, "libero_spatial", Path("/results"))
            command = evaluate.command_for(cfg, ROOT)
            with patch.object(sys, "argv", [command[2], *command[3:]]):
                parsed = module.parse_args()
            self.assertEqual({f.name: getattr(parsed, f.name) for f in fields(parsed) if f.name in cfg}, cfg)

    def test_perturbation_probe_is_non_grayscale_rgb_at_camera_resolution(self):
        import numpy as np
        picture = liberoplus.perturbation_probe_image()
        pixels = np.asarray(picture)
        self.assertEqual(picture.mode, "RGB")
        self.assertEqual(pixels.shape, (256, 256, 3))
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertFalse(np.array_equal(pixels[..., 0], pixels[..., 1]))
        self.assertFalse(np.array_equal(pixels[..., 1], pixels[..., 2]))
        self.assertGreater(np.unique(pixels[..., 0]).size, 1)

    def pip_fixture(self, directory):
        roots = [Path(directory) / name for name in ("source", "clone")]
        for root in roots:
            (root / "conda-meta").mkdir(parents=True)
        states = [{"prefix": str(root), "versions": ["26.2.1"], "import_version": "26.2.1"}
                  for root in roots]
        return [root / "bin/python" for root in roots], states

    def test_clone_pip_refuses_source_target_alias(self):
        with patch.object(liberoplus.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "source environment"):
                liberoplus.ensure_clone_pip("/tmp/source/bin/python", "/tmp/source/bin/python")
            run.assert_not_called()

    def test_clone_pip_healthy_environment_is_not_reinstalled(self):
        with tempfile.TemporaryDirectory() as directory:
            interpreters, states = self.pip_fixture(directory)
            with patch.object(liberoplus, "pip_inventory", side_effect=states), \
                 patch.object(liberoplus.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, \
                 redirect_stdout(io.StringIO()):
                liberoplus.ensure_clone_pip(*interpreters)
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(call.args[0][-1] == "--version" for call in run.call_args_list))

    def test_clone_pip_repairs_both_records_only_in_target(self):
        with tempfile.TemporaryDirectory() as directory:
            interpreters, (source, healthy) = self.pip_fixture(directory)
            mixed = dict(healthy, versions=["26.0.1", "26.2.1"], import_version="26.0.1")
            remaining = dict(healthy, versions=["26.0.1"], import_version=None)
            empty = dict(healthy, versions=[], import_version=None)
            def execute(command, **kwargs):
                if "download" in command:
                    dest = Path(command[command.index("--dest") + 1])
                    (dest / "pip-26.2.1-py3-none-any.whl").write_bytes(b"test fixture")
                return subprocess.CompletedProcess(command, 0)
            with patch.object(liberoplus, "pip_inventory", side_effect=[source, mixed, remaining, empty, healthy, source]), \
                 patch.object(liberoplus.subprocess, "run", side_effect=execute) as run, \
                 redirect_stdout(io.StringIO()):
                liberoplus.ensure_clone_pip(*interpreters)
            commands = [call.args[0] for call in run.call_args_list]
            mutations = [cmd for cmd in commands if "uninstall" in cmd or "install" in cmd]
            self.assertEqual(len(mutations), 3)
            for cmd in mutations:
                self.assertEqual(cmd[0], str(interpreters[0]))
                self.assertEqual(cmd[cmd.index("--python") + 1], str(interpreters[1]))
            self.assertIn("download", commands[1])
            self.assertIn("--no-deps", mutations[-1])

    def test_clone_pip_failed_download_does_not_uninstall(self):
        with tempfile.TemporaryDirectory() as directory:
            interpreters, (source, healthy) = self.pip_fixture(directory)
            broken = dict(healthy, versions=["26.0.1", "26.2.1"])
            def execute(command, **kwargs):
                if "download" in command:
                    raise subprocess.CalledProcessError(1, command)
                return subprocess.CompletedProcess(command, 0)
            with patch.object(liberoplus, "pip_inventory", side_effect=[source, broken]), \
                 patch.object(liberoplus.subprocess, "run", side_effect=execute) as run, \
                 redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):
                    liberoplus.ensure_clone_pip(*interpreters)
            self.assertFalse(any("uninstall" in call.args[0] for call in run.call_args_list))

    def test_only_evaluation_assets_selected(self):
        chosen = assets.selected("liberoplus", ["models", "datasets", "checkpoints", "simulators"])
        self.assertEqual(set(chosen), {"dino_b", "bert", "libero_checkpoint", "liberoplus_sim_assets"})
        self.assertEqual(assets.selected("liberoplus", ["datasets"]), {})
        self.assertNotIn("liberoplus_sim_assets", assets.selected("all", ["simulators"]))

    def test_protocol_and_paths_do_not_replace_libero(self):
        shared = ["--checkpoint", "/tmp/model.pth", "--store", "/assets"]
        plus = evaluate.parser().parse_args(["--benchmark", "liberoplus", *shared])
        normal = evaluate.parser().parse_args(["--benchmark", "libero", *shared])
        plus_cfg = evaluate.libero_config(plus, "libero_object", Path("/results"))
        normal_cfg = evaluate.libero_config(normal, "libero_object", Path("/results"))
        self.assertEqual(plus_cfg["num_trials_per_task"], 1)
        self.assertEqual(normal_cfg["num_trials_per_task"], 50)
        self.assertEqual(plus_cfg["libero_root"], "/assets/simulators/LIBERO-plus")
        self.assertEqual(normal_cfg["libero_root"], "/assets/simulators/LIBERO")
        for field in ("stats_key", "stats_path", "chunk_size", "seed", "precision", "text_padding_length"):
            self.assertEqual(plus_cfg[field], normal_cfg[field])
        self.assertEqual({key for key in plus_cfg if plus_cfg[key] != normal_cfg[key]},
                         {"num_trials_per_task", "libero_root"})

    def test_plus_dry_run_stays_offline_and_separate(self):
        output = io.StringIO()
        with patch.object(assets.subprocess, "run", side_effect=AssertionError("spawned")), redirect_stdout(output):
            evaluate.main(["--benchmark", "liberoplus", "--checkpoint", "/tmp/model.pth", "--store", "/assets", "--dry-run"])
            assets.main(["--benchmark", "liberoplus", "--dry-run"])
        self.assertEqual(output.getvalue().count("--num_trials_per_task 1 "), 4)
        self.assertIn("/results/liberoplus/", output.getvalue())
        self.assertIn("text_padding_length=21", output.getvalue())

    def test_virtual_bddl_resolves_to_real_base(self):
        self.assertEqual(liberoplus.resolved_bddl("task_language_10_view_0_0_100_0_0_initstate_0"), "task_language_10")
        self.assertEqual(liberoplus.resolved_bddl("task_light_2"), "task_light_2")
        self.assertEqual(sum(liberoplus.COUNTS.values()), 10030)

    def test_zip_rejects_escape_and_symlink(self):
        for filename, mode in [("../escape", 0), ("/assets/file", 0), ("other/file", 0),
                               ("assets/link", (stat.S_IFLNK | 0o777) << 16)]:
            with self.subTest(filename=filename), zipfile.ZipFile(io.BytesIO(), "w") as archive:
                item = zipfile.ZipInfo(filename)
                item.external_attr = mode
                archive.writestr(item, b"x")
                with self.assertRaises(ValueError):
                    liberoplus.safe_members(archive)

    def test_zip_accepts_expected_asset_root(self):
        with zipfile.ZipFile(io.BytesIO(), "w") as archive:
            archive.writestr("assets/scene.xml", b"<xml/>")
            archive.writestr("__MACOSX/._scene.xml", b"metadata")
            self.assertEqual([(x.filename, str(path)) for x, path in liberoplus.safe_members(archive)],
                             [("assets/scene.xml", "assets/scene.xml")])

    def test_official_prefix_extraction_keeps_original_header_and_normalizes_manifest(self):
        prefix = str(liberoplus.ZIP_PREFIX)
        content = b"<scene>fixture</scene>"
        for base in ("", prefix + "/"):
            with self.subTest(base=base), tempfile.TemporaryDirectory() as directory:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as archive:
                    archive.writestr(base + "assets/", b"")
                    archive.writestr(base + "assets/scenes/scene.xml", content)
                with zipfile.ZipFile(buffer) as archive, redirect_stdout(io.StringIO()):
                    names = [item.filename for item in archive.infolist()]
                    result = liberoplus.extract_assets(archive, Path(directory))
                    self.assertEqual(names, [item.filename for item in archive.infolist()])
                self.assertEqual(result, {"assets/scenes/scene.xml": len(content)})
                self.assertEqual((Path(directory) / "assets/scenes/scene.xml").read_bytes(), content)
                self.assertFalse((Path(directory) / "inspire").exists())

    def test_prefixed_archive_still_rejects_unsafe_paths_and_unexpected_prefixes(self):
        prefix = str(liberoplus.ZIP_PREFIX)
        for filename in (prefix + "/assets/../escape", "/" + prefix + "/assets/file",
                         "unknown/assets/file", prefix + "/other/file", "assets\\..\\escape"):
            with self.subTest(filename=filename), zipfile.ZipFile(io.BytesIO(), "w") as archive:
                archive.writestr(filename, b"x")
                with self.assertRaises(ValueError):
                    liberoplus.safe_members(archive)

    def test_colliding_normalized_paths_are_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(io.BytesIO(), "w") as archive:
            archive.writestr("assets/file", b"first")
            archive.writestr(str(liberoplus.ZIP_PREFIX) + "/assets/file", b"second")
            with self.assertRaisesRegex(ValueError, "Duplicate ZIP destination"):
                liberoplus.extract_assets(archive, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_extraction_refuses_existing_assets_and_dangling_symlink(self):
        with tempfile.TemporaryDirectory() as directory, zipfile.ZipFile(io.BytesIO(), "w") as archive:
            root = Path(directory)
            archive.writestr("assets/file", b"new")
            (root / "assets").mkdir()
            (root / "assets/file").write_bytes(b"keep")
            with self.assertRaises(ValueError):
                liberoplus.extract_assets(archive, root)
            self.assertEqual((root / "assets/file").read_bytes(), b"keep")
            other = root / "other"
            other.mkdir()
            (other / "assets").symlink_to(root / "missing")
            with self.assertRaises(ValueError):
                liberoplus.extract_assets(archive, other)


if __name__ == "__main__":
    unittest.main()
