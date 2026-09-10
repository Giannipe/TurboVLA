"""No GPU/network required: consolidated CLI contracts and safety checks."""
import hashlib
import io
import json
import os
import shutil
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cluster/_internal"))
import assets
import evaluate as cluster_eval
import train as cluster_train


class ClusterCLITests(unittest.TestCase):
    def test_libero_uses_torchrun_from_selected_python_environment(self):
        args = cluster_train.parser().parse_args(["--benchmark", "libero", "--output", "/tmp/run", "--gpus", "4"])
        cfg = cluster_train.libero_config(args)
        with patch.object(cluster_train.sys, "executable", "/chosen/env/bin/python"):
            command = cluster_train.libero_command(args, cfg, ROOT)
        self.assertEqual(command[:5], ["/chosen/env/bin/torchrun", "--standalone", "--nnodes=1",
                                     "--nproc_per_node=4", str(ROOT / "experiments/libero/train.py")])

    def test_model_selection_does_not_download_wrong_backbone(self):
        libero = assets.selected("libero", ["models"])
        self.assertIn("dino_b", libero)
        self.assertNotIn("dino_l", libero)
        self.assertNotIn("libero_checkpoint", libero)
        self.assertEqual(set(assets.selected("robotwin", ["datasets"])), {"robotwin_data"})

    def test_unknown_config_keys_rejected(self):
        with self.assertRaises(ValueError):
            assets.overrides({"seed": 42}, ["seedd=7"])
        with self.assertRaises(ValueError):
            assets.overrides({"flag": False}, ["flag=maybe"])

    def test_public_eval_protocol(self):
        args = cluster_eval.parser().parse_args(["--benchmark", "libero", "--checkpoint", "/tmp/trusted.pth", "--store", "/assets"])
        cfg = cluster_eval.libero_config(args, "libero_object", Path("/results"))
        self.assertEqual((cfg["num_trials_per_task"], cfg["seed"], cfg["chunk_size"], cfg["num_open_loop_steps"]), (50, 7, 12, 12))
        self.assertFalse(cfg["allow_hf_download"])
        self.assertEqual(cfg["ckpt_path"], "/tmp/trusted.pth")

    def test_eval_all_plans_four_sequential_suites_without_submitting(self):
        output = io.StringIO()
        with patch.object(assets.subprocess, "run", side_effect=AssertionError("spawned")), redirect_stdout(output):
            cluster_eval.main(["--benchmark", "libero", "--checkpoint", "/tmp/trusted.pth", "--dry-run"])
        commands = [line for line in output.getvalue().splitlines() if "--task_suite_name " in line]
        self.assertEqual(len(commands), 4)
        for command, suite in zip(commands, assets.SUITES):
            self.assertIn(f"--task_suite_name {suite}", command)
            self.assertNotIn("sbatch", command)

    def test_dry_run_does_not_spawn_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "untouched"
            with patch.object(assets.subprocess, "run", side_effect=AssertionError("spawned")), redirect_stdout(io.StringIO()):
                assets.main(["--benchmark", "libero", "--store", str(root), "--dry-run"])
                cluster_train.main(["--benchmark", "libero", "--store", str(root), "--dry-run"])
                cluster_eval.main(["--benchmark", "libero", "--store", str(root), "--checkpoint", "/tmp/test.pth", "--dry-run"])
            self.assertFalse(root.exists())

    def test_cached_download_verification_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = ("all", "models", "test/model", "model", "abc123", "models/test", ["weights.bin"])
            cache = root / spec[5] / ".cache/huggingface/download"
            cache.mkdir(parents=True)
            value = b"test weights"
            path = root / spec[5] / "weights.bin"
            path.write_bytes(value)
            (cache / "weights.bin.metadata").write_text("abc123\n" + hashlib.sha256(value).hexdigest() + "\n0\n")
            entries, origin = assets.expected_files("test", spec, root)
            report = assets.verify_asset("test", spec, root, entries, origin)
            self.assertEqual(report["files"]["weights.bin"]["bytes"], len(value))
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                assets.verify_asset("test", spec, root, entries, origin)

    def test_invalid_index_paths_rejected(self):
        spec = ("all", "models", "test/model", "model", "abc", "models/test", ["*"])
        with self.assertRaises(ValueError):
            assets.verify_asset("test", spec, Path("/tmp"), {"../escape": {"hash": "a", "algorithm": "sha256"}}, "test")

    def test_robotwin_paper_override(self):
        args = cluster_train.parser().parse_args(["--benchmark", "robotwin"])
        cfg = cluster_train.robotwin_config(args)
        self.assertEqual(cfg["trainer.max_train_steps"], 55000)
        self.assertEqual(cfg["datasets.vla_data.per_device_batch_size"] * args.gpus * cfg["trainer.gradient_accumulation_steps"], 192)

    def test_no_silent_resume_or_output_override(self):
        args = cluster_train.parser().parse_args(["--benchmark", "libero", "--output", "/tmp/run", "--set", "resume_mode=all"])
        with self.assertRaises(ValueError):
            cluster_train.libero_config(args)

    def test_eval_does_not_create_a_second_log_file(self):
        args = cluster_eval.parser().parse_args(["--benchmark", "libero", "--checkpoint", "/tmp/test.pth"])
        cfg = cluster_eval.libero_config(args, "libero_goal", Path("/scratch/results/run/libero_goal"))
        self.assertEqual(cfg["log_path"], "")
        self.assertEqual(cfg["result_json_path"], "/scratch/results/run/libero_goal/results.json")

    def test_snapshot_has_all_new_entry_points_and_correct_root(self):
        self.assertEqual(assets.ROOT, ROOT)
        with tempfile.TemporaryDirectory() as directory, patch.object(assets.shutil, "copytree"):
            destination = Path(directory) / "source"
            assets.snapshot(destination)
            for name in ("envs.sh", "assets.sh", "evaluate.sh", "train.sh"):
                self.assertTrue((destination / "scripts/cluster" / name).is_file())
            for name in ("assets.py", "evaluate.py", "train.py"):
                path = destination / "scripts/cluster/_internal" / name
                self.assertTrue(path.is_file())
                self.assertEqual(path.resolve().parents[3], destination)
            self.assertFalse((destination / "scripts/cluster/archive").exists())
            self.assertFalse((destination / "scripts/cluster/_internal/slurm.sh").exists())

    def test_run_names_use_job_names_not_ids_or_timestamps(self):
        with patch.dict(os.environ, {"SLURM_JOB_NAME": "libero-baseline", "SLURM_JOB_ID": "1922639"}):
            self.assertEqual(assets.run_name("train", "libero"), "libero-baseline")
            output = io.StringIO()
            with redirect_stdout(output):
                cluster_train.main(["--benchmark", "libero", "--store", "/assets", "--dry-run"])
            self.assertIn("/assets/training/libero/libero-baseline/checkpoints", output.getvalue())
            self.assertNotIn("1922639", output.getvalue())
        with patch.dict(os.environ, {"SLURM_JOB_NAME": "../invalid"}):
            with self.assertRaises(ValueError):
                assets.run_name("train", "libero")

    def test_attempt_names_are_meaningful_and_sequential(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(assets.attempt_name(output, False), "initial")
            self.assertEqual(assets.attempt_name(output, True), "resume-01")
            (output / "attempts/resume-01").mkdir(parents=True)
            self.assertEqual(assets.attempt_name(output, True), "resume-02")

    def test_training_inherits_slurm_streams_without_duplicating_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline"
            with patch.object(assets, "check_environment"), patch.object(assets, "snapshot"), \
                 patch.object(assets, "record_run", side_effect=lambda path, *_: path.mkdir(parents=True)), \
                 patch.object(cluster_train, "resume_fingerprint", return_value={}), \
                 patch.object(cluster_train.subprocess, "run") as run, redirect_stdout(io.StringIO()):
                cluster_train.main(["--benchmark", "libero", "--output", str(output), "--skip-asset-check"])
            self.assertEqual(run.call_count, 2)  # GPU preflight, then training.
            self.assertNotIn("stdout", run.call_args.kwargs)
            self.assertNotIn("stderr", run.call_args.kwargs)
            self.assertTrue((output / "attempts/initial").is_dir())
            self.assertEqual(list(output.rglob("*.log")), [])
            self.assertEqual(list(output.rglob("logs.json")), [])


class SlurmLauncherTests(unittest.TestCase):
    """Exercise actual shell entry points, including a Slurm spool-copy layout."""

    def shell(self, name, arguments, *, extra_env=None, script=None):
        env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        env.update(TURBOVLA_REPO=str(ROOT), TURBOVLA_PYTHON=sys.executable,
                   TURBOVLA_CONDA_EXE="/nonexistent/conda", PYTHONDONTWRITEBYTECODE="1")
        env.update(extra_env or {})
        return subprocess.run(["bash", str(script or ROOT / f"scripts/cluster/{name}.sh"), *arguments],
                              cwd="/tmp", env=env, capture_output=True, text=True, timeout=30)

    def test_sbatch_headers_and_shell_syntax(self):
        for name in ("envs", "assets", "evaluate", "train"):
            with self.subTest(name=name):
                script = ROOT / f"scripts/cluster/{name}.sh"
                result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                content = script.read_text()
                kind = {"evaluate": "evaluation", "train": "training"}.get(name, name)
                for extension, key in (("out", "output"), ("err", "error")):
                    self.assertIn(f"#SBATCH --{key}=/home/gpepe/ws/logs/turbovla/{kind}/%x.{extension}", content)
                self.assertNotIn("#SBATCH --open-mode", content)
                self.assertNotIn("turbovla_lock_job_logs", content)
                self.assertNotIn("_internal/slurm.sh", content)
                self.assertNotIn("%j", content)
                self.assertNotIn("#SBATCH --array", content)
                self.assertIn("#SBATCH --ntasks=1", content)
                if name in ("evaluate", "train"):
                    self.assertIn(f"#SBATCH --gres=gpu:{1 if name == 'evaluate' else 4}", content)

    def test_help_and_previews_need_no_allocation_or_conda_installation(self):
        for name in ("envs", "assets", "evaluate", "train"):
            with self.subTest(name=name):
                result = self.shell(name, ["--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                args = ["--benchmark=libero", "--dry-run"]
                if name == "evaluate":
                    args += ["--checkpoint", "/tmp/preview.pth"]
                result = self.shell(name, args)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_jobs_cannot_run_accidentally_on_login_node(self):
        for name in ("envs", "assets", "evaluate", "train"):
            with self.subTest(name=name):
                result = self.shell(name, ["--benchmark", "libero"])
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Use sbatch", result.stderr)

    def test_resources_and_legacy_submit_are_not_application_flags(self):
        for name in ("assets", "evaluate", "train"):
            for flag in ("--submit", "--gpus=4", "--gres=gpu:4", "--cpus-per-task=8", "--mem=64G",
                         "--job-name=baseline", "--job-name", "-J", "-Jbaseline"):
                with self.subTest(name=name, flag=flag):
                    result = self.shell(name, ["--help", flag])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("sbatch", result.stderr)

    def test_train_spool_copy_uses_real_repo_and_allocated_gpu_count(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "slurm_script"
            shutil.copyfile(ROOT / "scripts/cluster/train.sh", script)
            result = self.shell("train", ["--benchmark", "libero", "--dry-run"], script=script,
                                extra_env={"SLURM_GPUS_ON_NODE": "2"})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Global batch = 2 GPUs x 8 x 4 = 64", result.stdout)
            self.assertIn("--nproc_per_node=2", result.stdout)
            self.assertIn(str(ROOT / "experiments/libero/train.py"), result.stdout)

    def test_eval_rejects_multiple_gpus_and_launchers_reject_multiple_tasks(self):
        result = self.shell("evaluate", ["--benchmark", "libero", "--checkpoint", "/tmp/preview.pth", "--dry-run"],
                            extra_env={"SLURM_GPUS_ON_NODE": "4"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one GPU", result.stderr)
        result = self.shell("train", ["--benchmark", "libero", "--dry-run"], extra_env={"SLURM_NTASKS": "4"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--ntasks=1", result.stderr)

    def test_env_options_are_order_independent_and_benchmark_specific(self):
        content = (ROOT / "scripts/cluster/envs.sh").read_text()
        self.assertIn("sbatch --partition=gpu_a40 --gres=gpu:1 scripts/cluster/envs.sh --benchmark robotwin --with-flash-attn", content)
        self.assertNotIn("bash scripts/cluster/envs.sh robotwin", content)
        result = self.shell("envs", ["--dry-run", "--benchmark", "robotwin", "--with-flash-attn"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("benchmark=robotwin", result.stdout)
        result = self.shell("envs", ["--benchmark", "robotwin", "--simulator-only", "--dry-run"])
        self.assertNotEqual(result.returncode, 0)

    def test_robotwin_preview_uses_same_batch_entry_point(self):
        result = self.shell("train", ["--benchmark=robotwin", "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Global batch = 4 GPUs x 48 x 1 = 192", result.stdout)
        self.assertIn("--trainer.max_train_steps 55000", result.stdout)

    def test_robotwin_stdio_mode_keeps_server_and_eval_output_without_log_files(self):
        script = (ROOT / "scripts/robotwin/start_eval.sh").read_text()
        function = script[script.index("launch_task_in_slot() {"):script.index("# --- Argument parsing ---")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_policy_server.sh").write_text("echo policy-server-output\n")
            (root / "eval_task.sh").write_text("echo 'Success rate: 1.0'\necho simulator-stderr >&2\n")
            harness = function + '''
kill_descendants() { :; }
kill_policy_servers_on_port() { :; }
wait_for_server() { return 0; }
sleep() { :; }
SLOT_GPUS=(0)
SLOT_PORTS=(7100)
ACTIVE_PIDS=()
TASK_CONFIG=demo_clean
POLICY_NAME=test
CKPT_PATH=/unused/checkpoint
STARVLA_PYTHON=unused
ROBOTWIN_PYTHON=unused
launch_task_in_slot 0 adjust_bottle
wait "${ACTIVE_PIDS[0]}"
'''
            env = dict(os.environ, SCRIPT_DIR=str(root), LOG_DIR=str(root / "must-not-exist"), ROBOTWIN_LOG_TO_STDIO="1")
            result = subprocess.run(["bash", "-c", harness], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("policy-server-output", result.stdout)
            self.assertIn("Success rate: 1.0", result.stdout)
            self.assertIn("simulator-stderr", result.stderr)
            self.assertEqual(list(root.rglob("*.log")), [])


if __name__ == "__main__":
    unittest.main()
