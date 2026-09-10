#!/usr/bin/env python3
"""One downloader/verifier for TurboVLA. Also shared, dependency-light CLI utilities.

Pinned public assets are verifiable; private paper-run files are not. Nothing
downloads at import time. --dry-run is read-only and offline.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
LIBERO_REV = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EPISODES = dict(zip(SUITES, (432, 454, 428, 379)))
CONFIG_HASHES = {
    "libero_all4_stats.json": "b6a26530cceee6748481995067bfa5570a48a4d22990730dc187fbfa1cc24827",
    "online_text_layout.json": "b87d34038ad6ba099445e9ec9999d1f7b24a00dd4e000a8275d1d5e7735b6432",
}

# name: benchmark, component, HF repository, type, revision, local path, patterns
CATALOG = {
    "dino_b": ("libero", "models", "facebook/dinov3-vitb16-pretrain-lvd1689m", "model", "5931719e67bbdb9737e363e781fb0c67687896bc", "models/dinov3-vitb16", ["config.json", "preprocessor_config.json", "model.safetensors"]),
    "dino_l": ("robotwin", "models", "facebook/dinov3-vitl16-pretrain-lvd1689m", "model", "ea8dc2863c51be0a264bab82070e3e8836b02d51", "models/dinov3-vitl16", ["config.json", "preprocessor_config.json", "model.safetensors"]),
    "bert": ("all", "models", "google-bert/bert-base-uncased", "model", "86b5e0934494bd15c9632b12f734a8a67f723594", "models/bert-base-uncased", ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]),
    "grounding": ("all", "models", "ShilongLiu/GroundingDINO", "model", "84311ae61139581d0e62eca0bad610ad14e70aef", "models/groundingdino", ["groundingdino_swint_ogc.pth"]),
    "libero_data": ("libero", "datasets", "openvla/modified_libero_rlds", "dataset", "a7c9ae18499b6eea8a32f78a9302327b752b1b5f", "datasets/libero", ["libero_*_no_noops/1.0.0/*"]),
    "robotwin_data": ("robotwin", "datasets", "StarVLA/RoboTwin-Clean", "dataset", "070d3b86d7db06f924702a82725dba0f4d89a433", "datasets/robotwin/Clean", ["*/data/*", "*/videos/*", "*/meta/*"]),
    "libero_checkpoint": ("libero", "checkpoints", "H-EmbodVis/TurboVLA", "model", "cb5300544693013164c4bb251a13036002a55c81", "pretrained/TurboVLA-unified-cb53005", ["checkpoints/libero/turbovla_libero.pth", "config.json", "libero_all4_stats.json", "README.md"]),
    "robotwin_checkpoint": ("robotwin", "checkpoints", "H-EmbodVis/TurboVLA", "model", "f7b0f53afa248408d20748f2579e446c7ce4119e", "pretrained/TurboVLA", ["checkpoints/robotwin/steps_55000_ema_model.safetensors", "config.yaml", "dataset_statistics.json"]),
    "robotwin_sim_assets": ("robotwin", "simulators", "TianxingChen/RoboTwin2.0", "dataset", "9dc9299c163db059931898a9f0852098a61155a1", "simulator_assets/robotwin", ["background_texture.zip", "embodiments.zip", "objects.zip"]),
}
HASH_LOCKS = {
    "models/groundingdino/groundingdino_swint_ogc.pth": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
    "models/dinov3-vitb16/model.safetensors": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    "models/bert-base-uncased/model.safetensors": "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
    "pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth": "d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab",
    "pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors": "d0183df6bafd44507b6c797da5c5ab080ef8446cde4a8127d7280546d9f7c034",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def store_default():
    value = os.environ.get("TURBOVLA_STORE")
    if value:
        return Path(value).expanduser().resolve()
    return Path(os.environ.get("SCRATCH_FLASH", "/mnt/beegfs/gpepe")) / "TurboVLA"


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def digest(path, algorithm="sha256"):
    h = hashlib.sha1() if algorithm == "git-blob-sha1" else hashlib.sha256()
    if algorithm == "git-blob-sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def selected(benchmark, components):
    return {name: spec for name, spec in CATALOG.items()
            if (benchmark == "all" or spec[0] in (benchmark, "all")) and spec[1] in components}


def expected_files(name, spec, store, online=False):
    _, _, repo, kind, revision, local, patterns = spec
    index_path = store / "manifests/asset_indexes" / f"{name}-{revision}.json"
    if online:
        from huggingface_hub import HfApi
        info = HfApi().repo_info(repo_id=repo, repo_type=kind, revision=revision, files_metadata=True)
        require(info.sha == revision, f"Remote revision mismatch: {repo}")
        entries = {}
        for item in info.siblings:
            if not any(fnmatch.fnmatch(item.rfilename, p) for p in patterns):
                continue
            lfs = item.lfs
            sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            entries[item.rfilename] = {"hash": sha or item.blob_id, "algorithm": "sha256" if sha else "git-blob-sha1"}
        require(entries, f"No remote assets selected: {repo}")
        for pattern in patterns:
            require(any(fnmatch.fnmatch(key, pattern) for key in entries), f"Missing remote pattern: {repo}/{pattern}")
        return entries, "remote-pinned-index"
    if index_path.is_file():
        data = json.loads(index_path.read_text())
        require(data["revision"] == revision and data["repo"] == repo, f"Invalid index {index_path}")
        return data["files"], "saved-remote-index"
    # Migration from the old downloader. Cache metadata stores commit and ETag.
    cache = store / local / ".cache/huggingface/download"
    entries = {}
    for path in sorted(cache.rglob("*.metadata")):
        relative = str(path.relative_to(cache))[:-len(".metadata")]
        if any(fnmatch.fnmatch(relative, p) for p in patterns):
            lines = path.read_text().splitlines()
            require(lines[0] == revision, f"Unexpected cached revision: {path}")
            value = lines[1].strip('"')
            require(len(value) in (40, 64), f"Unknown HF hash: {path}")
            entries[relative] = {"hash": value, "algorithm": "sha256" if len(value) == 64 else "git-blob-sha1"}
    require(entries, f"No verification metadata for {name}; run assets.sh with --online or download")
    # Fixed model/weight files must not disappear unnoticed from local metadata.
    for pattern in patterns:
        require(any(fnmatch.fnmatch(key, pattern) for key in entries), f"Missing expected pattern: {name}/{pattern}")
    return entries, "download-cache (RoboTwin completeness needs --online)"


def verify_asset(name, spec, store, entries, origin):
    files = {}
    for relative, expected in sorted(entries.items()):
        relpath = Path(relative)
        require(not relpath.is_absolute() and ".." not in relpath.parts, "Unsafe asset index path")
        path = store / spec[5] / relative
        actual = digest(path, expected["algorithm"])
        require(actual == expected["hash"], f"Hash mismatch: {path}; refusing to silently replace it")
        pinned = HASH_LOCKS.get(str(Path(spec[5]) / relative))
        if pinned:
            require(actual == pinned, f"Public weight lock mismatch: {path}")
        files[relative] = {**expected, "bytes": path.stat().st_size}
    if name == "libero_data":
        for suite, count in EPISODES.items():
            prefix = f"{suite}_no_noops/1.0.0"
            require(f"{prefix}/features.json" in entries, f"Missing features: {suite}")
            info = json.loads((store / spec[5] / prefix / "dataset_info.json").read_text())
            split = next(s for s in info["splits"] if s["name"] == "train")
            require(sum(map(int, split["shardLengths"])) == count, f"Episode count mismatch: {suite}")
            shards = len(split["shardLengths"])
            for i in range(shards):
                key = f"{prefix}/{info['name']}-train.tfrecord-{i:05d}-of-{shards:05d}"
                require(key in files, f"Missing RLDS shard: {key}")
    if name == "robotwin_data":
        tree = ast.parse((ROOT / "experiments/robotwin/data_registry/data_config.py").read_text())
        tasks = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "_CLEAN50_TASKS" for t in n.targets))
        require(len(tasks) == 50, "Expected 50 RoboTwin tasks")
        for task in tasks:
            require(f"{task}/meta/info.json" in files, f"Missing RoboTwin task: {task}")
    if name == "libero_checkpoint":
        require(digest(store / spec[5] / "libero_all4_stats.json") == CONFIG_HASHES["libero_all4_stats.json"], "Release statistics changed")
    return {"repo": spec[2], "revision": spec[4], "local_dir": spec[5], "index_source": origin, "files": files}


def verify(benchmark, components, store, online=False):
    report = {"scope": "Pinned public release, not private paper-run artifacts", "assets": {}}
    for name, spec in selected(benchmark, components).items():
        entries, origin = expected_files(name, spec, store, online)
        report["assets"][name] = verify_asset(name, spec, store, entries, origin)
        print(f"[verified] {name}: {len(entries)} files", flush=True)
    if benchmark in ("libero", "all"):
        for name, expected in CONFIG_HASHES.items():
            require(digest(ROOT / "experiments/libero/configs" / name) == expected, f"Released config changed: {name}")
    return report


def libero_simulator(store, download):
    target = store / "simulators/LIBERO"
    if not target.exists() and download:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", LIBERO_REV], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    require(actual == LIBERO_REV, f"LIBERO at {actual}, expected {LIBERO_REV}; not changing existing checkout")
    dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"], text=True)
    require(not dirty, "Modified LIBERO tracked files; baseline cannot be verified")
    root = target / "libero/libero"
    for name in ("bddl_files", "init_files", "assets"):
        require((root / name).is_dir(), f"Incomplete simulator: {root / name}")
    if download:
        marker = target / "libero/__init__.py"
        if not marker.exists():
            marker.write_text("# Package marker for the upstream nested editable package.\n")
        cfg = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
               "init_states": str(root / "init_files"), "assets": str(root / "assets"),
               "datasets": str(store / "datasets/libero_raw")}
        path = store / "config/libero/config.yaml"
        # JSON is valid YAML. Preserve existing configuration, validate instead of overwrite.
        if not path.exists():
            write_json(path, cfg)
        print("LIBERO downloaded; install into the policy env with envs.sh --benchmark libero --simulator-only")
    import yaml
    configuration = yaml.safe_load((store / "config/libero/config.yaml").read_text())
    for key, value in {"benchmark_root": root, "bddl_files": root / "bddl_files",
                       "init_states": root / "init_files", "assets": root / "assets"}.items():
        require(Path(configuration[key]).resolve() == value.resolve(), f"LIBERO config path mismatch: {key}")
    return {"path": str(target), "revision": actual}


def add_execution_flags(parser, *, gpus=0):
    parser.add_argument("--store", type=Path, default=store_default())
    parser.add_argument("--dry-run", action="store_true", help="Show plan/commands; no writes, network, or jobs")
    # Internal only: the .sh wrapper derives this from the Slurm allocation.
    parser.add_argument("--gpus", type=int, default=gpus, help=argparse.SUPPRESS)


def runtime_env(store, source=ROOT):
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{source}:{source}/third_party/vla_adapter:{source}/third_party/starvla_runtime",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_DISABLE_TELEMETRY="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2", TF_CPP_MIN_LOG_LEVEL="2",
               LIBERO_CONFIG_PATH=str(store / "config/libero"),
               BERT_MODEL_PATH=str(store / "models/bert-base-uncased"),
               TURBOVLA_INIT_CKPT=str(store / "models/groundingdino/groundingdino_swint_ogc.pth"),
               ROBOTWIN_DATA_ROOT=str(store / "datasets/robotwin"), STARVLA_PYTHON=sys.executable,
               WANDB_MODE=os.environ.get("WANDB_MODE", "disabled"))
    return env


def run_name(kind, benchmark):
    """Use the user's Slurm name, not scheduler IDs, hashes or timestamps."""
    name = os.environ.get("SLURM_JOB_NAME") or f"turbovla-{kind}-{benchmark}"
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name),
            "Use --job-name with letters, digits, dots, hyphens or underscores, without spaces/slashes")
    return name


def attempt_name(output, resume):
    if not resume:
        return "initial"
    index = 1
    while (output / "attempts" / f"resume-{index:02d}").exists():
        index += 1
    return f"resume-{index:02d}"


def overrides(config, values):
    config = dict(config)
    for assignment in values:
        key, sep, value = assignment.partition("=")
        require(sep and key in config, f"Unknown --set key: {key}")
        default = config[key]
        if isinstance(default, bool):
            require(value.lower() in ("true", "false"), f"{key} requires true/false")
            config[key] = value.lower() == "true"
        elif isinstance(default, (int, float)):
            config[key] = type(default)(value)
        else:
            config[key] = value
    return config


def check_environment(benchmark, training=False):
    from importlib.metadata import version
    expected = {"torch": "2.3.1", "torchvision": "0.18.1", "transformers": "4.56.0",
                "tensorflow": "2.20.0", "tensorflow-datasets": "4.9.3"} if benchmark == "libero" else {
                    "torch": "2.6.0", "torchvision": "0.21.0", "transformers": "4.57.0"}
    for name, wanted in expected.items():
        require(version(name).split("+")[0] == wanted, f"Expected {name}=={wanted}, got {version(name)}")
    if benchmark == "robotwin" and training:
        require(version("flash-attn") == "2.7.4.post1", "Install reference FlashAttention with envs.sh")


def snapshot(destination):
    """Copy only runtime code; archives/results/assets are deliberately excluded."""
    for relative in ("turbovla", "experiments", "third_party", "scripts/robotwin"):
        src = ROOT / relative
        shutil.copytree(src, destination / relative, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "scripts/cluster/_internal").mkdir(parents=True, exist_ok=True)
    for name in ("assets.sh", "evaluate.sh", "train.sh", "envs.sh"):
        shutil.copyfile(ROOT / "scripts/cluster" / name, destination / "scripts/cluster" / name)
    for name in ("assets.py", "evaluate.py", "train.py"):
        shutil.copyfile(ROOT / "scripts/cluster/_internal" / name, destination / "scripts/cluster/_internal" / name)


def record_run(directory, command, config, source=ROOT):
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config)
    write_json(directory / "execution.json", {"job_name": os.environ.get("SLURM_JOB_NAME"),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "started_at": datetime.now(timezone.utc).isoformat()})
    (directory / "command.txt").write_text(shlex.join(command) + "\n")
    for filename, cmd in (("git-head.txt", ["git", "-C", str(source), "rev-parse", "HEAD"]),
                          ("pip-freeze.txt", [sys.executable, "-m", "pip", "freeze"])):
        (directory / filename).write_text(subprocess.check_output(cmd, text=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["libero", "robotwin", "all"], required=True)
    parser.add_argument("--components", nargs="+", choices=["models", "datasets", "checkpoints", "simulators"],
                        default=["models", "datasets", "checkpoints", "simulators"])
    parser.add_argument("--verify-only", action="store_true", help="Read-only verification, no payload downloads")
    parser.add_argument("--online", action="store_true", help="For verification: query complete pinned HF file index")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--report", type=Path, help="Optional verification report (explicit write also in verify-only mode)")
    add_execution_flags(parser)
    args = parser.parse_args(argv)
    args.store = args.store.resolve()
    require(args.max_workers > 0, "max-workers must be positive")
    if args.dry_run:
        for name, spec in selected(args.benchmark, args.components).items():
            print(f"{name}: {spec[2]} @ {spec[4]} -> {args.store / spec[5]}")
        if "simulators" in args.components:
            print(f"LIBERO source (if selected): {LIBERO_REV}; RoboTwin simulator installation remains separate")
        return
    os.environ.update(HF_HUB_CACHE=str(args.store / "cache/huggingface/hub"),
                      HF_XET_CACHE=str(args.store / "cache/huggingface/xet"), HF_HUB_DISABLE_TELEMETRY="1")
    if not args.verify_only:
        # Empty destination sends current hf_xet diagnostics to Slurm's streams.
        os.environ["HF_XET_LOG_DEST"] = ""
        os.environ.pop("HF_XET_LOG_FILE", None)
    # HF_HOME/token location deliberately unchanged.
    if args.verify_only:
        report = verify(args.benchmark, args.components, args.store, args.online)
    else:
        from huggingface_hub import snapshot_download
        report = {"scope": "Pinned public release, not private paper-run artifacts", "assets": {}}
        with locked(args.store / ".download_assets.lock"):
            for name, spec in selected(args.benchmark, args.components).items():
                entries, origin = expected_files(name, spec, args.store, online=True)
                # Preserve any existing modified assets; missing files can be resumed.
                for relative, entry in entries.items():
                    path = args.store / spec[5] / relative
                    if path.is_file():
                        require(digest(path, entry["algorithm"]) == entry["hash"], f"Existing asset differs: {path}")
                snapshot_download(repo_id=spec[2], repo_type=spec[3], revision=spec[4],
                                  local_dir=str(args.store / spec[5]), allow_patterns=list(entries), max_workers=args.max_workers)
                report["assets"][name] = verify_asset(name, spec, args.store, entries, origin)
                write_json(args.store / "manifests/asset_indexes" / f"{name}-{spec[4]}.json",
                           {"repo": spec[2], "revision": spec[4], "files": entries})
        if args.benchmark in ("libero", "all"):
            for name, expected in CONFIG_HASHES.items():
                require(digest(ROOT / "experiments/libero/configs" / name) == expected, f"Released config changed: {name}")
    if "simulators" in args.components:
        if args.benchmark in ("libero", "all"):
            report["libero_simulator"] = libero_simulator(args.store, not args.verify_only)
        if args.benchmark in ("robotwin", "all"):
            print("RoboTwin: ZIP assets verified; simulator checkout, extraction and simulator env are separate prerequisites.")
    report["status"] = "passed"
    if args.report or not args.verify_only:
        write_json(args.report or args.store / "manifests" / f"assets-{args.benchmark}-{stamp()}.json", report)
    print(f"PASSED: {len(report['assets'])} asset groups; public-source identity only")


if __name__ == "__main__":
    main()
