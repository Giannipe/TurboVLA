#!/usr/bin/env python3
"""One cluster training launcher, with explicit upstream defaults and overrides.

LIBERO: 4 GPUs x batch 8 x accumulation 4 = 128, 80k updates, EMA 0.999.
RoboTwin: 4 GPUs x batch 48 = 192, 55k updates (root README/paper override).
--dry-run never starts training. Submit the public train.sh with sbatch.
"""
from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import assets

LIBERO_DEFAULTS = dict(
    dataset_split="train", stats_key="libero_all4_no_noops", normalize_binary_gripper="auto",
    batch_size=8, grad_accum_steps=4, lr=5e-5, head_lr=5e-5, dinov3_lr=5e-5,
    weight_decay=1e-10, head_weight_decay=1e-10, dinov3_weight_decay=1e-10,
    precision="fp32", dinov3_precision="bf16_autocast", max_steps=80000,
    lr_schedule_steps=80000, warmup_steps=10000, min_lr_ratio=1.0, save_steps=1000,
    log_freq=20, max_grad_norm=1.0, save_final=True, num_workers=4, shuffle_buffer=512,
    step_mix_buffer_size=64, expected_image_size=256, hidden_dim=256, nheads=8,
    dim_feedforward=2048, max_text_len=256, text_padding_length=21,
    vla_feature_enhancer_layers=6, enhancer_inner_dim=1024, action_dim=7, chunk_size=12,
    state_dim=8, num_state_tokens=2, text_dropout=0.0, fusion_dropout=0.0,
    fusion_droppath=0.1, seed=42, allow_hf_download=False, shuffle_steps_within_episode=True,
    freeze_backbones=False, freeze_text_encoder=True, require_feature_enhancer_preload=True,
    load_text_projection_from_init=True, require_text_proj_preload=True,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=["libero", "robotwin"], required=True)
    p.add_argument("--output", type=Path, help="New run directory, or existing run with --resume")
    p.add_argument("--resume", action="store_true", help="LIBERO only: restore latest model/optimizer/scheduler/EMA")
    p.add_argument("--batch-size", type=int, help="Per GPU: default 8 LIBERO / 48 RoboTwin")
    p.add_argument("--grad-accum-steps", type=int, help="Default 4 LIBERO / 1 RoboTwin")
    p.add_argument("--max-steps", type=int, help="Default 80000 LIBERO / 55000 RoboTwin; LIBERO LR horizon stays 80000")
    p.add_argument("--warmup-steps", type=int, help="Default 10000 LIBERO / 1000 RoboTwin")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate for all trainable groups")
    p.add_argument("--save-steps", type=int, help="Default 1000 LIBERO / 5000 RoboTwin")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, help="Per GPU process: default 4 LIBERO / 8 RoboTwin")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="LIBERO: any known setting; RoboTwin: existing YAML dotted keys (advanced)")
    p.add_argument("--skip-asset-check", action="store_true", help="Non-baseline/custom assets; bypass pinned byte audit")
    assets.add_execution_flags(p, gpus=4)
    return p


def libero_config(args, source=assets.ROOT):
    data = args.store / "datasets/libero"
    cfg = dict(LIBERO_DEFAULTS)
    cfg.update(dataset_dir=str(data / "libero_10_no_noops/1.0.0"),
               dataset_dirs=",".join(str(data / f"{s}_no_noops/1.0.0") for s in ("libero_10", "libero_goal", "libero_object", "libero_spatial")),
               stats_path=str(source / "experiments/libero/configs/libero_all4_stats.json"),
               text_layout_path=str(source / "experiments/libero/configs/online_text_layout.json"),
               dinov3_path=str(args.store / "models/dinov3-vitb16"), bert_path=str(args.store / "models/bert-base-uncased"),
               pretrained_init_ckpt=str(args.store / "models/groundingdino/groundingdino_swint_ogc.pth"),
               checkpoint_dir=str(args.output / "checkpoints"), checkpoint_prefix="turbovla_step",
               resume_mode="all" if args.resume else "none", seed=args.seed,
               lr=args.lr, head_lr=args.lr, dinov3_lr=args.lr)
    for name in ("batch_size", "grad_accum_steps", "max_steps", "warmup_steps", "save_steps", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            cfg[name] = value
    cfg = assets.overrides(cfg, args.set)
    # Resume/output are safety controls, not hidden --set escape hatches.
    assets.require(cfg["resume_mode"] == ("all" if args.resume else "none"), "Use --resume, not --set resume_mode")
    assets.require(cfg["checkpoint_dir"] == str(args.output / "checkpoints"), "Use --output, not --set checkpoint_dir")
    for name in ("batch_size", "grad_accum_steps", "max_steps", "save_steps", "log_freq", "chunk_size"):
        assets.require(cfg[name] > 0, f"{name} must be positive")
    assets.require(cfg["num_workers"] >= 0 and cfg["warmup_steps"] >= 0, "Invalid workers/warmup")
    assets.require(cfg["lr_schedule_steps"] > cfg["warmup_steps"], "LR horizon must exceed warmup")
    assets.require(not cfg["allow_hf_download"], "Offline launcher: download assets first")
    return cfg


def libero_command(args, cfg, source):
    # Use the torchrun entry point from the same environment as this launcher.
    # This is equivalent to python -m torch.distributed.run, not a new trainer.
    torchrun = Path(sys.executable).with_name("torchrun")
    command = [str(torchrun), "--standalone", "--nnodes=1",
               f"--nproc_per_node={args.gpus}", str(source / "experiments/libero/train.py")]
    for key, value in cfg.items():
        if isinstance(value, bool):
            flag = key if value else ("train_text_encoder" if key == "freeze_text_encoder" else "no_" + key)
            command.append("--" + flag)
        else:
            command += ["--" + key, str(value)]
    return command


def robotwin_config(args):
    cfg = {"datasets.vla_data.per_device_batch_size": args.batch_size if args.batch_size is not None else 48,
           "trainer.gradient_accumulation_steps": args.grad_accum_steps if args.grad_accum_steps is not None else 1,
           "trainer.max_train_steps": args.max_steps if args.max_steps is not None else 55000,
           "trainer.num_warmup_steps": args.warmup_steps if args.warmup_steps is not None else 1000,
           "trainer.save_interval": args.save_steps if args.save_steps is not None else 5000,
           "datasets.vla_data.num_workers": args.num_workers if args.num_workers is not None else 8,
           "trainer.ema_decay": 0.999, "trainer.ema_device": "cuda", "seed": args.seed}
    for group in ("base", "text_encoder", "vision_encoder", "vision_language_interaction", "vision_projection", "action_head"):
        cfg[f"trainer.learning_rate.{group}"] = args.lr
    # Use the upstream YAML schema, without importing the GPU runtime.
    import yaml
    base = yaml.safe_load(
        (assets.ROOT / "experiments/robotwin/configs/taskbalanced_all50.yaml").read_text()
    )
    for value in args.set:
        key, sep, raw = value.partition("=")
        assets.require(sep, "--set requires KEY=VALUE")
        node = base
        for part in key.split("."):
            assets.require(isinstance(node, dict) and part in node, f"Unknown RoboTwin YAML key: {key}")
            node = node[part]
        cfg[key] = yaml.safe_load(raw)
    for key in ("datasets.vla_data.per_device_batch_size", "trainer.gradient_accumulation_steps", "trainer.max_train_steps", "trainer.save_interval"):
        assets.require(cfg[key] > 0, f"{key} must be positive")
    return cfg


def robotwin_command(args, cfg, source):
    command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
               str(source / "experiments/robotwin/configs/deepspeed_zero2.yaml"), "--num_processes", str(args.gpus),
               "--main_process_port", "29630", str(source / "third_party/starvla_runtime/starVLA/training/train_turbovla.py"),
               "--config_yaml", str(source / "experiments/robotwin/configs/taskbalanced_all50.yaml"),
               "--run_root_dir", str(args.output.parent), "--run_id", args.output.name]
    for key, value in cfg.items():
        command += ["--" + key, json.dumps(value) if not isinstance(value, str) else value]
    return command


def resume_fingerprint(args, cfg):
    # Paths refer to the original checkout here, not per-attempt snapshot paths.
    comparable = {k: v for k, v in cfg.items() if k not in ("max_steps", "resume_mode")}
    sources = {}
    for directory in (assets.ROOT / "turbovla", assets.ROOT / "experiments/libero"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                sources[str(path.relative_to(assets.ROOT))] = assets.digest(path)
    return {"gpus": args.gpus, "config": comparable, "source_sha256": sources,
            "packages": {name: version(name) for name in ("torch", "transformers", "tensorflow", "tensorflow-datasets", "numpy")}}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    args.store = args.store.resolve()
    assets.require(args.gpus > 0 and args.lr > 0, "GPUs and LR must be positive")
    assets.require(not args.resume or (args.output and args.benchmark == "libero"), "--resume needs --output and is LIBERO-only; RoboTwin resume unvalidated")
    args.output = (args.output or args.store / "training" / args.benchmark / assets.run_name("train", args.benchmark)).resolve()
    cfg = libero_config(args) if args.benchmark == "libero" else robotwin_config(args)
    batch_key, accum_key = ("batch_size", "grad_accum_steps") if args.benchmark == "libero" else ("datasets.vla_data.per_device_batch_size", "trainer.gradient_accumulation_steps")
    print(f"Global batch = {args.gpus} GPUs x {cfg[batch_key]} x {cfg[accum_key]} = {args.gpus * cfg[batch_key] * cfg[accum_key]}")
    builder = libero_command if args.benchmark == "libero" else robotwin_command
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        print(shlex.join(builder(args, cfg, assets.ROOT)))
        return
    if args.benchmark == "libero":
        torchrun = Path(sys.executable).with_name("torchrun")
        assets.require(torchrun.is_file() and os.access(torchrun, os.X_OK),
                       f"Missing executable {torchrun}; check PyTorch in this Python environment")
    assets.check_environment(args.benchmark, training=True)
    subprocess.run([sys.executable, "-c", f'import torch; assert torch.cuda.device_count() == {args.gpus}, "Unexpected number of visible GPUs"'], check=True)
    if not args.resume:
        assets.require(not args.output.exists(), f"Refusing to overwrite {args.output}")
    else:
        assets.require(args.output.is_dir(), f"Missing run: {args.output}")
    with assets.locked(args.output / ".training.lock"):
        audit = assets.verify(args.benchmark, ["models", "datasets"], args.store) if not args.skip_asset_check else {"scope": "custom: byte audit skipped"}
        fingerprint = resume_fingerprint(args, cfg) if args.benchmark == "libero" else cfg
        # The asset audit includes actual hashes and catches on-disk changes during resume.
        fingerprint["asset_audit"] = {
            name: {key: value for key, value in details.items() if key != "index_source"}
            for name, details in audit.get("assets", {}).items()
        }
        fingerprint["skip_asset_check"] = args.skip_asset_check
        path = args.output / "resume_contract.json"
        if args.resume:
            assets.require(path.is_file() and json.loads(path.read_text()) == fingerprint, "Resume contract changed: code/config/GPU count/assets/packages differ")
            checkpoints = list((args.output / "checkpoints").glob(cfg["checkpoint_prefix"] + "_*.pth"))
            assets.require(checkpoints, "No checkpoint to resume; refusing a silent fresh start")
        else:
            assets.write_json(path, fingerprint)
        attempt = args.output / "attempts" / assets.attempt_name(args.output, args.resume)
        source = attempt / "source"
        runtime_cfg = libero_config(args, source) if args.benchmark == "libero" else cfg
        command = builder(args, runtime_cfg, source)
        assets.record_run(attempt, command, runtime_cfg)
        assets.snapshot(source)
        env = assets.runtime_env(args.store, source)
        env.update(DINOV3_MODEL_PATH=str(args.store / "models/dinov3-vitl16"))
        if args.benchmark == "robotwin":
            env.update(PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", NCCL_SOCKET_IFNAME="^lo,docker0,virbr0,veth",
                       NCCL_IB_DISABLE="1", TORCH_NCCL_BLOCKING_WAIT="1", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
        print(shlex.join(command), flush=True)
        with tempfile.TemporaryDirectory(prefix="turbovla-train-") as cache:
            env.update(MPLCONFIGDIR=str(Path(cache) / "matplotlib"), NUMBA_CACHE_DIR=str(Path(cache) / "numba"))
            # Inherit stdout/stderr: Slurm is the only persistent log writer.
            subprocess.run(command, cwd=source, env=env, check=True)
        print(f"Training target completed. Outputs: {args.output}")


if __name__ == "__main__":
    main()
