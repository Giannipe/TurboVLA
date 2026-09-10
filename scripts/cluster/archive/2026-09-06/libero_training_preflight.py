#!/usr/bin/env python3
"""Read-only offline asset audit. Optionally write its report to --output.

Checks against pinned Hugging Face download metadata, NOT the authors' private
training files. Hashes every expected RLDS shard; does not recompute statistics.
No torch/TensorFlow imports, so this can also run on the login node.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

MODELS = {
    "dinov3-vitb16": ("5931719e67bbdb9737e363e781fb0c67687896bc",
                     ["config.json", "preprocessor_config.json", "model.safetensors"]),
    "bert-base-uncased": ("86b5e0934494bd15c9632b12f734a8a67f723594",
                          ["config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors"]),
    "groundingdino": ("84311ae61139581d0e62eca0bad610ad14e70aef", ["groundingdino_swint_ogc.pth"]),
}
WEIGHT_HASHES = {
    "models/dinov3-vitb16/model.safetensors": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    "models/bert-base-uncased/model.safetensors": "68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3",
    "models/groundingdino/groundingdino_swint_ogc.pth": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
}
DATA_REVISION = "a7c9ae18499b6eea8a32f78a9302327b752b1b5f"
EPISODES = {"libero_10": 379, "libero_goal": 428, "libero_object": 454, "libero_spatial": 432}
CONFIG_HASHES = {
    "libero_all4_stats.json": "b6a26530cceee6748481995067bfa5570a48a4d22990730dc187fbfa1cc24827",
    "online_text_layout.json": "b87d34038ad6ba099445e9ec9999d1f7b24a00dd4e000a8275d1d5e7735b6432",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path, git_blob=False):
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_download(root, relative, revision):
    path = root / relative
    metadata = root / ".cache/huggingface/download" / (relative + ".metadata")
    lines = metadata.read_text().splitlines()
    require(lines[0] == revision, f"Unexpected revision: {path}")
    etag = lines[1].strip('"')
    require(len(etag) in (40, 64), f"Unsupported HF hash: {path}")
    digest = file_hash(path, git_blob=len(etag) == 40)
    require(digest == etag, f"Corrupt or modified asset: {path}")
    return {"hash": digest, "algorithm": "git-blob-sha1" if len(etag) == 40 else "sha256",
            "bytes": path.stat().st_size, "revision": revision}


def audit(store, repo):
    versions = {}
    for name, expected in {"torch": "2.3.1", "torchvision": "0.18.1", "transformers": "4.56.0",
                           "tensorflow": "2.20.0", "tensorflow-datasets": "4.9.3", "numpy": "1.26.4"}.items():
        versions[name] = importlib.metadata.version(name)
        require(versions[name].split("+")[0] == expected, f"Unexpected {name}: {versions[name]}")
    files = {}
    for directory, (revision, names) in MODELS.items():
        for name in names:
            key = f"models/{directory}/{name}"
            files[key] = check_download(store / "models" / directory, name, revision)
            if key in WEIGHT_HASHES:
                require(files[key]["hash"] == WEIGHT_HASHES[key], f"Weight lock mismatch: {key}")
    for name, expected in CONFIG_HASHES.items():
        path = repo / "experiments/libero/configs" / name
        require(file_hash(path) == expected, f"Released configuration modified: {path}")
        files[f"configs/{name}"] = {"sha256": expected}
    data_root = store / "datasets/libero"
    suites = {}
    for suite, expected in EPISODES.items():
        prefix = f"{suite}_no_noops/1.0.0"
        for name in ("dataset_info.json", "features.json"):
            relative = f"{prefix}/{name}"
            files[f"datasets/{relative}"] = check_download(data_root, relative, DATA_REVISION)
        info = json.loads((data_root / prefix / "dataset_info.json").read_text())
        split = next(item for item in info["splits"] if item["name"] == "train")
        require(sum(map(int, split["shardLengths"])) == expected, f"Episode count mismatch: {suite}")
        count = len(split["shardLengths"])
        for index in range(count):
            relative = f"{prefix}/{info['name']}-train.tfrecord-{index:05d}-of-{count:05d}"
            files[f"datasets/{relative}"] = check_download(data_root, relative, DATA_REVISION)
        suites[suite] = {"episodes": expected, "shards": count}
    sources = {}
    for directory in (repo / "turbovla", repo / "experiments/libero"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json"):
                sources[str(path.relative_to(repo))] = file_hash(path)
    for name in ("libero_training_recipe.sh", "libero_training_preflight.py", "train_libero.sbatch"):
        relative = f"scripts/cluster/{name}"
        sources[relative] = file_hash(repo / relative)
    return {"status": "passed", "scope": "All expected downloaded training assets, not private author artifacts",
            "versions": versions, "suites": suites, "files": files, "source_sha256": sources,
            "statistics": "released values retained; not independently recomputed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.store, args.repo)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "scope", "versions", "suites")}, indent=2))


if __name__ == "__main__":
    main()
