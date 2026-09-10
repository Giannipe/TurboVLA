#!/usr/bin/env python3
"""Validate real LIBERO samples and EMA training checkpoints; not a benchmark."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

SUITES = {"libero_10_no_noops": 379, "libero_goal_no_noops": 428,
          "libero_object_no_noops": 454, "libero_spatial_no_noops": 432}
ROOT = Path(__file__).resolve().parents[2]


def data_check(store):
    import tensorflow_datasets as tfds
    import torch
    from turbovla.data.mixed_suite import LiberoMixedRLDSDataset, vla_collate_fn

    paths = [store / "datasets/libero" / suite / "1.0.0" for suite in SUITES]
    dataset = LiberoMixedRLDSDataset(
        dataset_dir=str(paths[0]), dataset_dirs=[str(p) for p in paths],
        LOCAL_DINOV3_PATH=str(store / "models/dinov3-vitb16"),
        stats_path=str(ROOT / "experiments/libero/configs/libero_all4_stats.json"),
        stats_key="libero_all4_no_noops", chunk_size=12,
        local_files_only=True, expected_image_size=256,
    )
    report = []
    for (suite, count), path in zip(SUITES.items(), paths):
        builder = tfds.builder_from_directory(str(path))
        actual = builder.info.splits["train"].num_examples
        assert actual == count, (suite, actual, count)
        episode = next(iter(tfds.as_numpy(builder.as_dataset(split="train").take(1))))
        steps = list(episode["steps"])
        samples, instructions, states, actions, masks = vla_collate_fn(
            [dataset._build_step_sample(steps, 0, len(steps))])
        assert tuple(samples["dinov3"].shape) == (1, 2, 3, 256, 256)
        assert tuple(states.shape) == (1, 8)
        assert tuple(actions.shape) == (1, 12, 7)
        assert tuple(masks.shape) == (1, 12)
        assert instructions[0].strip()
        assert all(torch.isfinite(t).all().item() for t in
                   (samples["dinov3"], states, actions, masks))
        report.append({"suite": suite, "episodes": actual, "sample_steps": len(steps),
                       "instruction": instructions[0], "sample_validated": True})
    return {"datasets": report, "total_episodes": sum(SUITES.values()),
            "scope": "one real episode/sample per suite; not a full statistics or byte-identity audit"}


def checkpoint_check(path, step, previous):
    import torch
    from turbovla.evaluation.policy import _checkpoint_state_dict

    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    assert ckpt["global_step"] == step
    assert ckpt["ema_decay"] == 0.999
    assert math.isfinite(ckpt["loss"])
    raw, ema = ckpt["model_state_dict"], ckpt["ema_model_state_dict"]
    assert raw.keys() == ema.keys()
    assert _checkpoint_state_dict(ckpt) is ema
    for name in raw:
        assert raw[name].shape == ema[name].shape
        assert torch.isfinite(raw[name]).all().item(), name
        assert torch.isfinite(ema[name]).all().item(), name
    optimizer_steps = {int(value["step"]) for value in ckpt["optimizer_state_dict"]["state"].values()}
    assert optimizer_steps == {step}, optimizer_steps
    assert ckpt["scheduler_state_dict"]["last_epoch"] == step
    args = ckpt["args"]
    assert args["batch_size"] * args["grad_accum_steps"] == 128
    assert args["lr_schedule_steps"] == 80000 and args["warmup_steps"] == 10000
    assert not args["freeze_backbones"] and args["freeze_text_encoder"]
    changed = None
    if previous:
        old = torch.load(previous, map_location="cpu", mmap=True, weights_only=True)
        changed = sum(not torch.equal(raw[name], old["model_state_dict"][name]) for name in raw)
        assert changed > 0, "No weights changed after resuming"
    return {"checkpoint": str(path), "global_step": step, "loss": ckpt["loss"],
            "tensor_count": len(raw), "ema_decay": ckpt["ema_decay"],
            "evaluation_weights": "ema_model_state_dict", "optimizer_steps": sorted(optimizer_steps),
            "raw_tensors_changed_after_resume": changed, "checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["data", "checkpoint", "final"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    store = Path(os.environ.get("TURBOVLA_STORE", "/mnt/beegfs/gpepe/TurboVLA"))
    if args.mode == "data":
        report = data_check(store)
        destination = args.run_dir / "data_check.json"
    elif args.mode == "checkpoint":
        report = checkpoint_check(args.run_dir / "checkpoints" / f"smoke_{args.step}.pth",
                                  args.step, args.previous)
        destination = args.run_dir / f"checkpoint_{args.step}_check.json"
    else:
        reports = [json.loads((args.run_dir / f"checkpoint_{step}_check.json").read_text())
                   for step in (2, 3)]
        assert all(r["checks_passed"] for r in reports)
        resume_log = (args.run_dir / "train_resume.log").read_text()
        assert "restored EMA:" in resume_log and "mode=all" in resume_log
        assert "resume missing keys: 0" in resume_log and "resume unexpected keys: 0" in resume_log
        rollouts = []
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
            result = json.loads((args.run_dir / "evaluation" / f"{suite}.json").read_text())
            assert result["total_episodes"] == 1 and len(result["tasks"]) == 1
            assert result["tasks"][0]["task_id"] == 0
            assert Path(result["ckpt_path"]) == args.run_dir / "checkpoints/smoke_3.pth"
            rollouts.append({"suite": suite, "episodes": 1, "successes": result["total_successes"]})
        report = {"status": "passed", "scope": "pipeline smoke test, NOT paper reproduction",
                  "training_steps": 3, "effective_batch": 128, "gpus": 1,
                  "ema_decay": 0.999, "resume": "model, optimizer, scheduler and EMA restored",
                  "resume_limitation": "upstream does not restore RNG/data-iterator position; not bitwise continuation",
                  "rollouts": rollouts}
        destination = args.run_dir / "smoke_summary.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
