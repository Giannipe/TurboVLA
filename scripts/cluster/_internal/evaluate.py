#!/usr/bin/env python3
"""Evaluate one compatible TurboVLA checkpoint: LIBERO, LIBERO+ or RoboTwin clean50.

Default LIBERO protocol: seed 7, 50 trials/task, chunk/open-loop 12, BF16, EGL.
LIBERO+ reuses that policy configuration with one trial per perturbation variant.
--suite all evaluates the four suites sequentially within the current job.
Checkpoint must be trusted and compatible with the selected upstream evaluator.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import assets

LIBERO_DEFAULTS = dict(
    stats_key="libero_all4_no_noops", normalize_binary_gripper="auto", allow_hf_download=False,
    num_trials_per_task=50, num_steps_wait=10, chunk_size=12, num_open_loop_steps=12,
    env_img_res=256, seed=7, control_mode="relative", precision="bf16",
    dinov3_output_hidden_states=True, save_video=False, task_ids="", max_tasks=-1,
    hidden_dim=256, nheads=8, dim_feedforward=2048, max_text_len=256, text_padding_length=21,
    vla_feature_enhancer_layers=6, enhancer_inner_dim=1024, action_dim=7, state_dim=8,
    num_state_tokens=2, text_dropout=0.0, fusion_dropout=0.0, fusion_droppath=0.1,
    sub_sentence_present=True,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["libero", "robotwin", "liberoplus"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="Any compatible trusted checkpoint; never selected implicitly")
    p.add_argument("--suite", choices=["all", *assets.SUITES], default="all", help="LIBERO suite")
    p.add_argument("--trials", type=int, help="Default 50 LIBERO / 1 LIBERO+ / 100 RoboTwin")
    p.add_argument("--seed", type=int, default=7, help="LIBERO evaluation seed")
    p.add_argument("--chunk-size", type=int, default=12)
    p.add_argument("--open-loop-steps", type=int, default=12)
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16", help="LIBERO precision")
    p.add_argument("--renderer", choices=["egl", "osmesa"], default="egl")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--task-ids", default="", help="LIBERO comma-separated task indices")
    p.add_argument("--stats-path", type=Path)
    p.add_argument("--load-only", action="store_true", help="LIBERO: strict model load without rollout")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override another LIBERO setting; recorded in config.json")
    p.add_argument("--output", type=Path, help="New result root; existing suite results are never overwritten")
    p.add_argument("--skip-asset-check", action="store_true", help="Explicit non-baseline run without pinned asset audit")
    p.add_argument("--robotwin-root", type=Path, default=Path(os.environ["ROBOTWIN_PATH"]) if "ROBOTWIN_PATH" in os.environ else None)
    p.add_argument("--robotwin-python", type=Path, help="Separate simulator environment's Python (not the policy Python)")
    p.add_argument("--tasks", nargs="*", default=[], help="RoboTwin task names; omitted = all clean50")
    assets.add_execution_flags(p, gpus=1)
    return p


def libero_config(args, suite, output, source=assets.ROOT):
    cfg = dict(LIBERO_DEFAULTS)
    cfg.update(num_trials_per_task=args.trials or (1 if args.benchmark == "liberoplus" else 50), seed=args.seed, chunk_size=args.chunk_size,
               num_open_loop_steps=args.open_loop_steps, precision=args.precision,
               save_video=args.save_video, task_ids=args.task_ids)
    cfg = assets.overrides(cfg, args.set)
    simulator_name = "LIBERO-plus" if args.benchmark == "liberoplus" else "LIBERO"
    cfg.update(ckpt_path=str(args.checkpoint), libero_root=str(args.store / "simulators" / simulator_name),
               dinov3_path=str(args.store / "models/dinov3-vitb16"), bert_path=str(args.store / "models/bert-base-uncased"),
               stats_path=str(args.stats_path or source / "experiments/libero/configs/libero_all4_stats.json"),
               task_suite_name=suite, mujoco_gl=args.renderer, pyopengl_platform=args.renderer,
               dry_run_model_load=args.load_only, video_out_path=str(output / "videos"),
               result_json_path=str(output / "results.json"),
               log_path="")  # Upstream keeps StreamHandler, without a second FileHandler.
    assets.require(cfg["num_trials_per_task"] > 0 and cfg["chunk_size"] > 0, "Invalid trials/chunk")
    assets.require(0 < cfg["num_open_loop_steps"] <= cfg["chunk_size"], "Open-loop steps must be in [1, chunk_size]")
    assets.require(not cfg["allow_hf_download"], "Cluster evaluation is offline; use assets.sh first")
    return cfg


def command_for(cfg, source):
    command = [sys.executable, "-u", str(source / "experiments/libero/evaluate.py")]
    for key, value in cfg.items():
        command += [f"--{key}", str(value).lower() if isinstance(value, bool) else str(value)]
    return command


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    args.store = args.store.resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.stats_path = args.stats_path.resolve() if args.stats_path else None
    if args.robotwin_root:
        args.robotwin_root = args.robotwin_root.resolve()
    args.output = (args.output or args.store / "results" / args.benchmark / assets.run_name("evaluate", args.benchmark)).resolve()
    assets.require(args.trials is None or args.trials > 0, "trials must be positive")
    assets.require(args.benchmark == "robotwin" or args.gpus == 1, "LIBERO/LIBERO+ uses one GPU per evaluation job")
    if args.benchmark == "robotwin":
        assets.require(args.robotwin_root and args.robotwin_python, "RoboTwin needs --robotwin-root and --robotwin-python")
        assets.require(not args.set and not args.load_only and not args.task_ids and not args.stats_path,
                       "--set/--load-only/--task-ids/--stats-path are LIBERO-only")
        assets.require(args.suite == "all" and args.seed == 7 and args.chunk_size == 12
                       and args.open_loop_steps == 12 and not args.save_video,
                       "LIBERO-only flags changed; RoboTwin uses its own upstream rollout configuration")
    else:
        assets.require(not args.tasks and not args.robotwin_root and not args.robotwin_python,
                       "RoboTwin-only options used with LIBERO")
    suites = list(assets.SUITES) if args.suite == "all" else [args.suite]
    if not args.dry_run:
        assets.require(args.checkpoint.is_file(), f"Checkpoint missing: {args.checkpoint}")
        if args.benchmark == "robotwin":
            assets.require(args.robotwin_root.is_dir() and args.robotwin_python.is_file(),
                           "RoboTwin simulator or its Python executable is missing")
        assets.check_environment(args.benchmark)
        if not args.skip_asset_check:
            assets.verify(args.benchmark, ["models"], args.store)
        if args.benchmark == "libero":
            assets.libero_simulator(args.store, download=False)
        elif args.benchmark == "liberoplus":
            import liberoplus
            liberoplus.prepare(args.store, download=False)
    for suite in suites if args.benchmark != "robotwin" else ["clean50"]:
        output = args.output / suite
        source = assets.ROOT if args.dry_run else output / "source"
        env = assets.runtime_env(args.store, source)
        env.update(MUJOCO_GL=args.renderer, PYOPENGL_PLATFORM=args.renderer)
        if args.benchmark == "liberoplus":
            env["LIBERO_CONFIG_PATH"] = str(args.store / "config/liberoplus")
        if args.benchmark != "robotwin":
            cfg = libero_config(args, suite, output, source)
            command = command_for(cfg, source)
            if args.benchmark == "liberoplus":
                print(f"LIBERO+ evaluation only: {cfg['num_trials_per_task']} rollout(s)/variant; "
                      f"text_padding_length={cfg['text_padding_length']} (long instructions can be truncated)", flush=True)
        else:
            cfg = {"checkpoint": str(args.checkpoint), "trials": args.trials or 100, "tasks": args.tasks or ["all"]}
            env.update(ROBOTWIN_PATH=str(args.robotwin_root), ROBOTWIN_PYTHON=str(args.robotwin_python.resolve()),
                       ROBOTWIN_TEST_NUM=str(cfg["trials"]), ROBOTWIN_LOG_TO_STDIO="1",
                       DINOV3_MODEL_PATH=str(args.store / "models/dinov3-vitl16"),
                       ROBOTWIN_USE_BF16="1" if args.precision == "bf16" else "0")
            command = ["bash", str(source / "scripts/robotwin/evaluate.sh"), str(args.checkpoint), *args.tasks]
        print(shlex.join(command), flush=True)
        print(f"Results: {output}; pinned asset audit: {not args.skip_asset_check}", flush=True)
        if args.dry_run:
            if args.benchmark == "robotwin":
                print(json.dumps(cfg, indent=2))
            continue
        assets.record_run(output, command, cfg)
        assets.snapshot(source)
        if args.benchmark == "liberoplus":
            import liberoplus
            assets.write_json(output / "benchmark.json", {"benchmark": "liberoplus", "revision": liberoplus.REVISION,
                "suite_counts": liberoplus.COUNTS, "classification_ids": "one-based; rollout task_id is zero-based",
                "text_padding_length": cfg["text_padding_length"]})
        # These caches are node-local and automatically removed after the run.
        assets.write_json(output / "checkpoint.json", {"path": str(args.checkpoint), "sha256": assets.digest(args.checkpoint)})
        with tempfile.TemporaryDirectory(prefix="turbovla-eval-") as cache:
            env.update(MPLCONFIGDIR=str(Path(cache) / "matplotlib"), NUMBA_CACHE_DIR=str(Path(cache) / "numba"))
            subprocess.run(command, cwd=source, env=env, check=True)
        if args.benchmark != "robotwin" and not args.load_only:
            result = json.loads((output / "results.json").read_text())
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
