"""Compare the cluster flags to the actual public training entry-point defaults."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from turbovla.training.train_mixed import parse_args_with_mixed_suite_stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cluster/_internal"))
import train as cluster_train


def recipe_arguments(profile, max_steps="80000"):
    cli = ["--benchmark", "libero", "--store", "/assets", "--output", "/run", "--max-steps", max_steps]
    if profile == "paper256":
        cli += ["--batch-size", "16"]
    args = cluster_train.parser().parse_args(cli)
    cfg = cluster_train.libero_config(args)
    command = cluster_train.libero_command(args, cfg, ROOT)
    return command[command.index(str(ROOT / "experiments/libero/train.py")) + 1:]


def parse(arguments):
    with patch.object(sys, "argv", ["train.py", *arguments]):
        return vars(parse_args_with_mixed_suite_stats())


class TrainingRecipeTests(unittest.TestCase):
    def test_upstream128_matches_public_defaults(self):
        actual = parse(recipe_arguments("upstream128"))
        path_options = ["dataset_dir", "dataset_dirs", "stats_path", "stats_key", "dinov3_path",
                        "bert_path", "pretrained_init_ckpt", "checkpoint_dir", "text_layout_path"]
        minimal = []
        for key in path_options:
            minimal.extend([f"--{key}", actual[key]])
        expected = parse(minimal)
        # None becomes max_steps inside train_model; explicit horizon is equivalent.
        expected["lr_schedule_steps"] = expected["max_steps"]
        self.assertEqual(actual, expected)
        self.assertEqual(actual["batch_size"] * actual["grad_accum_steps"] * 4, 128)

    def test_paper_variant_only_changes_batch(self):
        actual = parse(recipe_arguments("paper256"))
        expected = parse(recipe_arguments("upstream128"))
        expected["batch_size"] = 16
        self.assertEqual(actual, expected)

    def test_short_run_config_keeps_full_schedule_and_worker_count(self):
        actual = parse(recipe_arguments("upstream128", "3"))
        self.assertEqual(actual["max_steps"], 3)
        self.assertEqual(actual["lr_schedule_steps"], 80000)
        self.assertEqual(actual["num_workers"], 4)
        self.assertEqual(actual["grad_accum_steps"], 4)


if __name__ == "__main__":
    unittest.main()
