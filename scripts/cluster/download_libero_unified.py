#!/usr/bin/env python3
"""Download/check the pinned September LIBERO release, leaving old assets intact."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

REVISION = "cb5300544693013164c4bb251a13036002a55c81"
CHECKPOINT = "checkpoints/libero/turbovla_libero.pth"
SHA256 = "d031ad7be05a2f5d04afb3194ed26b0cb46083685edee7a5e145078a37d26bab"
REPO_ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    store = Path(os.environ.get("TURBOVLA_STORE", str(Path(os.environ.get(
        "SCRATCH_FLASH", "/mnt/beegfs/gpepe")) / "TurboVLA")))
    target = store / "pretrained" / f"TurboVLA-unified-{REVISION[:7]}"
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="H-EmbodVis/TurboVLA", revision=REVISION, local_dir=str(target),
            allow_patterns=[CHECKPOINT, "README.md", "config.json", "libero_all4_stats.json"],
            max_workers=2,
        )
    checkpoint = target / CHECKPOINT
    actual = digest(checkpoint)
    if actual != SHA256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {actual}")
    local_stats = REPO_ROOT / "experiments/libero/configs/libero_all4_stats.json"
    if digest(target / "libero_all4_stats.json") != digest(local_stats):
        raise ValueError("New release statistics differ from baseline: investigate before evaluating")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "transformers", "mujoco", "robosuite")}
    if versions["transformers"] != "4.56.0":
        raise ValueError(f"Expected transformers 4.56.0, got {versions['transformers']}")
    if args.verify_only:
        manifest = json.loads((target / "verified.json").read_text())
        if manifest["sha256"] != actual or manifest["revision"] != REVISION:
            raise ValueError("Verification manifest mismatch")
        print(json.dumps({"checkpoint": str(checkpoint), "sha256": actual,
                          "runtime_versions": versions}, indent=2), flush=True)
        return

    # Verify the hash before parsing the official PyTorch archive. No pickle globals allowed.
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or "ema_model_state_dict" in payload:
        raise ValueError("Unexpected checkpoint layout: review the loader selection")
    manifest = {
        "repo_id": "H-EmbodVis/TurboVLA", "revision": REVISION,
        "checkpoint": str(checkpoint), "sha256": actual,
        "size_bytes": checkpoint.stat().st_size,
        "checkpoint_keys": list(payload), "tensor_count": len(state),
        "loader_key": "model_state_dict", "runtime_versions": versions,
        "stats_sha256": digest(local_stats),
        "provenance_note": "Authors describe this as the 34k EMA export. The archive stores model_state_dict; no EMA is computed or keys renamed locally.",
    }
    temporary = target / "verified.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(target / "verified.json")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
