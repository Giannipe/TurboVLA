"""Pinned LIBERO+ simulator preparation and read-only benchmark audits.

No policy/training changes. Imported only for --benchmark liberoplus.
"""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile

import assets

REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
COUNTS = dict(zip(assets.SUITES, (2402, 2518, 2591, 2519)))
# The pinned public ZIP retains the authors' build-directory prefix.
# Accept this exact layout and the documented assets/ layout, not arbitrary prefixes.
ZIP_PREFIX = PurePosixPath("inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0")
EXTRACTED_FILE_COUNT = 448799
EXTRACTED_BYTES = 8953192654


def pip_inventory(python):
    """Inspect metadata even when pip cannot start; never import the broken CLI."""
    code = """
import importlib.metadata as metadata
import json
import sys
from pathlib import Path
rows = [d for d in metadata.distributions() if d.metadata['Name'].lower() == 'pip']
assert all(Path(d.locate_file('')).resolve().is_relative_to(Path(sys.prefix).resolve()) for d in rows)
try:
    import pip
    version = pip.__version__
except (ImportError, AttributeError):
    version = None
print(json.dumps({'prefix': sys.prefix, 'versions': sorted(d.version for d in rows), 'import_version': version}))
"""
    return json.loads(subprocess.check_output([str(python), "-I", "-B", "-c", code], text=True))


def ensure_clone_pip(source_python, target_python):
    """Repair only a distinct Conda clone, using working pip's public --python CLI.

    Conda's package record can predate a pip self-upgrade. Cloning then leaves
    two pip versions overlaid. Uninstall their records with the source runner,
    then install one prefetched wheel matching the working source version.
    """
    source_python, target_python = Path(source_python).resolve(), Path(target_python).resolve()
    assets.require(source_python != target_python, "Refusing to repair the source environment")
    source, target = pip_inventory(source_python), pip_inventory(target_python)
    source_root, target_root = Path(source["prefix"]).resolve(), Path(target["prefix"]).resolve()
    assets.require(source_root != target_root and not target_root.is_relative_to(source_root)
                   and not source_root.is_relative_to(target_root), "Pip repair requires separate environments")
    assets.require((source_root / "conda-meta").is_dir() and (target_root / "conda-meta").is_dir(),
                   "Pip repair is restricted to existing Conda environments")
    assets.require(source["versions"] == [source["import_version"]], "Source pip metadata is inconsistent")
    subprocess.run([str(source_python), "-I", "-B", "-m", "pip", "--version"], check=True)
    if target["versions"] == [target["import_version"]]:
        probe = subprocess.run([str(target_python), "-I", "-B", "-m", "pip", "--version"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            print("[pip-clone] Existing pip is coherent; no repair needed", flush=True)
            return
    version = source["import_version"]
    assets.require(re.fullmatch(r"\d+(?:\.\d+)+", version), "Unexpected source pip version")
    runner = [str(source_python), "-I", "-B", "-m", "pip", "--python", str(target_python),
              "--disable-pip-version-check"]
    print(f"[pip-clone] Repairing only {target_root}: {target['versions']} -> {version}", flush=True)
    with tempfile.TemporaryDirectory(prefix="turbovla-pip-repair-") as directory:
        # Fetch first: a network failure must not remove the installed files.
        subprocess.run([*runner, "download", "--no-deps", "--only-binary=:all:",
                        "--dest", directory, f"pip=={version}"], check=True)
        wheels = list(Path(directory).glob("pip-*.whl"))
        assets.require(len(wheels) == 1, "Expected exactly one pip wheel")
        for _ in range(3):
            if not target["versions"]:
                break
            previous = len(target["versions"])
            subprocess.run([*runner, "uninstall", "--yes", "pip"], check=True)
            target = pip_inventory(target_python)
            assets.require(len(target["versions"]) < previous, "Pip uninstall made no progress; inspect clone")
        assets.require(not target["versions"], "Pip metadata remains; inspect clone")
        subprocess.run([*runner, "install", "--no-index", "--no-deps", str(wheels[0])], check=True)
    target = pip_inventory(target_python)
    assets.require(target["versions"] == [version] and target["import_version"] == version,
                   "Repaired pip metadata mismatch")
    subprocess.run([str(target_python), "-I", "-B", "-m", "pip", "--version"], check=True)
    assets.require(pip_inventory(source_python) == source, "Source pip changed unexpectedly")
    print("[pip-clone] Repair complete; source pip unchanged", flush=True)


def resolved_bddl(name):
    # ControlEnv resolves these virtual perturbation names before opening a file.
    return name.split("_view_")[0] if "_view_" in name and "_initstate_" in name else name


def audit_tasks(root):
    tree = ast.parse((root / "benchmark/libero_suite_task_map.py").read_text())
    task_map = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets))
    categories = json.loads((root / "benchmark/task_classification.json").read_text())
    totals = Counter()
    for suite, count in COUNTS.items():
        names, rows = task_map[suite], categories[suite]
        assets.require(len(names) == len(rows) == count and len(set(names)) == count, f"Task count mismatch: {suite}")
        for i, (name, row) in enumerate(zip(names, rows)):
            assets.require(row["id"] == i + 1 and row["name"] == name, f"Classification mismatch: {suite}/{i}")
            assets.require((root / "bddl_files" / suite / f"{resolved_bddl(name)}.bddl").is_file(),
                           f"Missing resolved BDDL: {suite}/{name}")
            # Mirror the official get_task_init_states path selection (not plain name existence).
            relative = Path(suite) / f"{name}.pruned_init"
            if "_language_" in name:
                relative = Path(suite) / f"{name.split('_language_')[0]}.pruned_init"
            elif "_view_" in name:
                relative = Path(suite) / f"{name.split('_view_')[0]}.pruned_init"
            else:
                relative = Path(suite) / (re.sub(r"_(table|tb)_\d+", "", name) + ".pruned_init")
                if "_light_" in name:
                    relative = Path(suite) / f"{name.split('_light_')[0]}.pruned_init"
                if "_add_" in name or "_level" in name:
                    relative = Path("libero_newobj") / suite / f"{name}.pruned_init"
            assets.require((root / "init_files" / relative).is_file(), f"Missing init state: {relative}")
            totals[row["category"]] += 1
    return {"suite_counts": COUNTS, "category_counts": dict(totals), "total_tasks": sum(COUNTS.values())}


def safe_members(archive):
    """Return original ZipInfo objects and validated, normalized destination paths.

    Keep ZipInfo.filename intact: ZipFile.open checks it against the ZIP header.
    """
    members = []
    destinations = set()
    for item in archive.infolist():
        relative = PurePosixPath(item.filename)
        assets.require(not relative.is_absolute() and ".." not in relative.parts
                       and "\\" not in item.filename and "\x00" not in item.orig_filename,
                       f"Unsafe ZIP path: {item.filename}")
        assets.require(not stat.S_ISLNK(item.external_attr >> 16), f"ZIP symlink: {item.filename}")
        if relative.parts and relative.parts[0] == "__MACOSX":
            continue
        if relative.is_relative_to(ZIP_PREFIX):
            relative = relative.relative_to(ZIP_PREFIX)
        assets.require(relative.parts and relative.parts[0] == "assets", f"Unexpected ZIP root: {item.filename}")
        assets.require(relative != PurePosixPath("assets") or item.is_dir(), "ZIP assets root must be a directory")
        assets.require(relative not in destinations, f"Duplicate ZIP destination: {relative}")
        destinations.add(relative)
        members.append((item, relative))
    assets.require(any(not item.is_dir() for item, _ in members), "Empty asset archive")
    return members


def extract_assets(archive, root):
    """Extract under root/assets without overwrites; reading each member verifies CRC."""
    members = safe_members(archive)  # Validate every destination before any writes.
    target = root / "assets"
    assets.require(not target.exists() and not target.is_symlink(),
                   "LIBERO+ assets directory already exists; inspect partial extraction before retry")
    print(f"Extracting {len(members)} LIBERO+ entries ({sum(i.file_size for i, _ in members)} bytes)", flush=True)
    files = {}
    for item, relative in members:
        destination = root / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(item) as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        assets.require(destination.stat().st_size == item.file_size, f"Extracted size mismatch: {relative}")
        files[relative.as_posix()] = item.file_size
    return files


def prepare(store, download=False):
    # Serialize clone/extraction/config writes too, not only the HF download.
    # Read-only evaluation preflight must not create a lock or alter the store.
    if download:
        with assets.locked(store / ".prepare_liberoplus.lock"):
            return _prepare(store, download=True)
    return _prepare(store, download=False)


def extraction_files(completion):
    """Reject incomplete/malformed manifests for the pinned public archive."""
    assets.require(completion.get("zip_sha256") == assets.HASH_LOCKS["simulator_assets/liberoplus/assets.zip"],
                   "LIBERO+ extraction marker mismatch")
    files = completion.get("files")
    assets.require(isinstance(files, dict) and len(files) == EXTRACTED_FILE_COUNT,
                   "LIBERO+ extraction manifest file count mismatch")
    for name, size in files.items():
        rel = PurePosixPath(name)
        assets.require(not rel.is_absolute() and ".." not in rel.parts and len(rel.parts) > 1
                       and rel.parts[0] == "assets" and "\\" not in name and "\x00" not in name
                       and name == rel.as_posix(), "Unsafe extraction manifest path")
        assets.require(type(size) is int and size >= 0, f"Invalid extracted size: {name}")
    assets.require(sum(files.values()) == EXTRACTED_BYTES, "LIBERO+ extraction manifest byte count mismatch")
    return files


def _prepare(store, download=False):
    target = store / "simulators/LIBERO-plus"
    if not target.exists() and download:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                        "https://github.com/sylvestf/LIBERO-plus.git", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", REVISION], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", REVISION], check=True)
        subprocess.run(["git", "-C", str(target), "sparse-checkout", "set", "libero"], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    assets.require(actual == REVISION, f"LIBERO+ revision {actual}, expected {REVISION}; do not update silently")
    dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"], text=True)
    assets.require(not dirty, "Modified LIBERO+ tracked files; baseline not verified")
    root = target / "libero/libero"
    report = {"path": str(target), "revision": actual, **audit_tasks(root)}
    marker = store / "manifests/liberoplus-assets-extracted.json"
    archive_path = store / "simulator_assets/liberoplus/assets.zip"
    expected = assets.HASH_LOCKS["simulator_assets/liberoplus/assets.zip"]
    if download and not marker.exists():
        assets.require(not (root / "assets").exists(),
                       "LIBERO+ assets directory already exists without completion marker; inspect partial extraction before retry")
        assets.require(assets.digest(archive_path) == expected, "LIBERO+ ZIP checksum mismatch")
        with zipfile.ZipFile(archive_path) as archive:
            files = extract_assets(archive, root)
        completion = {"zip_sha256": expected, "files": files}
        extraction_files(completion)
        assets.write_json(marker, completion)
    completion = json.loads(marker.read_text())
    for name, size in extraction_files(completion).items():
        path = root / name
        assets.require(path.is_file() and path.stat().st_size == size, f"Incomplete extracted asset: {path}")
    config = {"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
              "init_states": str(root / "init_files"), "assets": str(root / "assets"),
              "datasets": str(store / "datasets/liberoplus_unused")}
    config_path = store / "config/liberoplus/config.yaml"
    if download:
        # Packaging-only fix for the upstream missing outer package marker (PR #54).
        package_marker = target / "libero/__init__.py"
        if not package_marker.exists():
            package_marker.write_text("# Package marker for the upstream nested editable package.\n")
        if not config_path.exists():
            assets.write_json(config_path, config)
    import yaml
    assets.require(yaml.safe_load(config_path.read_text()) == config, "LIBERO+ config path mismatch")
    print(f"LIBERO+ verified: {report['total_tasks']} tasks; extracted file sizes checked", flush=True)
    return report


def perturbation_probe_image():
    """Non-grayscale RGB fixture for the benchmark's camera-image code path.

    ImageMagick collapses an achromatic PNG to grayscale. Upstream motion_blur
    only handles that special case at 224px, not 256px; do not patch the benchmark
    or silently reshape its output. This probe validates the intended RGB path.
    """
    import numpy as np
    from PIL import Image
    row, column = np.indices((256, 256))
    pixels = np.stack((column, row, (row + column) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels)


def validate_environment(store):
    """CPU-only import/perturbation checks, NOT a MuJoCo rollout or GPU test."""
    import contextlib
    import io
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import fog, motion_blur
    assets.check_environment("liberoplus")
    assets.require(Path(get_libero_path("benchmark_root")).resolve() ==
                   (store / "simulators/LIBERO-plus/libero/libero").resolve(), "Wrong imported LIBERO config")
    assets.require(Path(benchmark.__file__).resolve().is_relative_to(store / "simulators/LIBERO-plus"),
                   "Imported vanilla LIBERO instead of LIBERO+")
    with contextlib.redirect_stdout(io.StringIO()):
        counts = {name: benchmark.get_benchmark_dict()[name]().n_tasks for name in COUNTS}
    assets.require(counts == COUNTS, f"Unexpected suite counts: {counts}")
    picture = perturbation_probe_image()
    for function in (fog, motion_blur):
        for severity in (1, 10):
            result = function(picture, severity=severity)
            assets.require(result.shape == (256, 256, 3) and np.isfinite(result).all()
                           and result.min() >= 0 and result.max() <= 255,
                           f"Invalid {function.__name__} output at 256 px, severity {severity}: {result.shape}")
    print("LIBERO+ imports, suite counts, RGB fog/motion blur at 256px (severity 1/10) OK", counts)
