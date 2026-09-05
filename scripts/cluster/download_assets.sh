#!/usr/bin/env bash
set -euo pipefail
umask 027

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Download and verify TurboVLA assets under $SCRATCH_FLASH/TurboVLA.

Usage:
  bash scripts/cluster/download_assets.sh [all|models|datasets|simulators] [options]

Options:
  --plan              Print the destination layout and pinned assets; do not write or download.
  --hf-dry-run        Query Hugging Face metadata and sizes without transferring file payloads.
  --max-workers N     Hugging Face download workers (default: 8).
  -h, --help          Show this help.

Environment overrides:
  TURBOVLA_STORE      default: $SCRATCH_FLASH/TurboVLA
  HF_ENV              conda environment providing `hf` (default: turbovla-libero)

Prerequisites:
  1. Accept the terms for both facebook DINOv3 models in a browser.
  2. Authenticate once with `hf auth login`; keep the token outside shared scratch.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

scope="${1:-all}"
case "$scope" in
  all|models|datasets|simulators)
    shift || true
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "unknown scope: $scope"
    ;;
esac

plan=false
hf_dry_run=false
max_workers="${HF_MAX_WORKERS:-8}"
while (($#)); do
  case "$1" in
    --plan)
      plan=true
      ;;
    --hf-dry-run)
      hf_dry_run=true
      ;;
    --max-workers)
      (($# >= 2)) || die "--max-workers requires an integer"
      max_workers="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown option: $1"
      ;;
  esac
  shift
done

[[ "$max_workers" =~ ^[1-9][0-9]*$ ]] || die "invalid --max-workers value: $max_workers"

scratch_flash="${SCRATCH_FLASH:-}"
if [[ -n "${TURBOVLA_STORE:-}" ]]; then
  store="$TURBOVLA_STORE"
else
  [[ -n "$scratch_flash" ]] || die "SCRATCH_FLASH is not defined"
  store="${scratch_flash}/TurboVLA"
fi

release_revision="f7b0f53afa248408d20748f2579e446c7ce4119e"
libero_revision="a7c9ae18499b6eea8a32f78a9302327b752b1b5f"
robotwin_revision="070d3b86d7db06f924702a82725dba0f4d89a433"
robotwin_assets_revision="9dc9299c163db059931898a9f0852098a61155a1"
bert_revision="86b5e0934494bd15c9632b12f734a8a67f723594"
groundingdino_revision="84311ae61139581d0e62eca0bad610ad14e70aef"
groundingdino_sha256="3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
dinov3_b_revision="5931719e67bbdb9737e363e781fb0c67687896bc"
dinov3_l_revision="ea8dc2863c51be0a264bab82070e3e8836b02d51"
robotwin_background_sha256="54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070"
robotwin_embodiments_sha256="6b87d7d55e106d8ff25917e0538eb1e177fc549280e8a742a8cec3cb9f953fc6"
robotwin_objects_sha256="6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27"

release_dir="${store}/pretrained/TurboVLA"
dinov3_b_dir="${store}/models/dinov3-vitb16"
dinov3_l_dir="${store}/models/dinov3-vitl16"
bert_dir="${store}/models/bert-base-uncased"
groundingdino_dir="${store}/models/groundingdino"
libero_dir="${store}/datasets/libero"
robotwin_clean_dir="${store}/datasets/robotwin/Clean"
robotwin_assets_dir="${store}/simulator_assets/robotwin"
manifest_dir="${store}/manifests"

print_plan() {
  cat <<EOF
TurboVLA asset plan
===================
Scope: $scope
Root:  $store

Models/checkpoints:
  $release_dir
    H-EmbodVis/TurboVLA @ $release_revision
  $dinov3_b_dir
    facebook/dinov3-vitb16-pretrain-lvd1689m @ $dinov3_b_revision
  $dinov3_l_dir
    facebook/dinov3-vitl16-pretrain-lvd1689m @ $dinov3_l_revision
  $bert_dir
    google-bert/bert-base-uncased @ $bert_revision
  $groundingdino_dir/groundingdino_swint_ogc.pth
    ShilongLiu/GroundingDINO @ $groundingdino_revision

Datasets:
  $libero_dir
    openvla/modified_libero_rlds @ $libero_revision
  $robotwin_clean_dir
    StarVLA/RoboTwin-Clean @ $robotwin_revision

Simulator assets:
  $robotwin_assets_dir/{background_texture,embodiments,objects}.zip
    TianxingChen/RoboTwin2.0 @ $robotwin_assets_revision

Generated manifest:
  $manifest_dir/hf_revisions.tsv
EOF
}

print_plan
[[ "$plan" == true ]] && exit 0

mkdir -p \
  "${store}/cache/huggingface/hub" \
  "${store}/cache/huggingface/xet" \
  "$release_dir" \
  "$dinov3_b_dir" \
  "$dinov3_l_dir" \
  "$bert_dir" \
  "$groundingdino_dir" \
  "$libero_dir" \
  "$robotwin_clean_dir" \
  "$robotwin_assets_dir" \
  "$manifest_dir"

[[ -w "$store" ]] || die "asset root is not writable: $store"

command -v flock >/dev/null 2>&1 || die "flock is required to prevent concurrent downloads"
exec 9>"${store}/.download_assets.lock"
flock -n 9 || die "another TurboVLA asset download is already using $store"

# Keep large caches on flash storage. HF_HOME is intentionally not changed so
# that the authentication token remains in the user's private home directory.
export HF_HUB_CACHE="${store}/cache/huggingface/hub"
export HF_XET_CACHE="${store}/cache/huggingface/xet"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_DISABLE_TELEMETRY=1

hf_env="${HF_ENV:-turbovla-libero}"
if command -v conda >/dev/null 2>&1 \
  && conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$hf_env"; then
  hf_cmd=(conda run --no-capture-output -n "$hf_env" hf)
  hf_python_cmd=(conda run --no-capture-output -n "$hf_env" python)
elif command -v hf >/dev/null 2>&1; then
  hf_cmd=(hf)
  hf_python_cmd=(python)
else
  die "Hugging Face CLI not found; create the environments first or activate one containing `hf`"
fi

run_hf_download() {
  "${hf_cmd[@]}" download "$@"
}

preflight_hf_repo() {
  "${hf_python_cmd[@]}" -c '
import fnmatch
import sys
from huggingface_hub import HfApi

repo_type, repo_id, revision, *patterns = sys.argv[1:]
info = HfApi().repo_info(
    repo_id=repo_id,
    repo_type=repo_type,
    revision=revision,
    files_metadata=True,
)
siblings = list(info.siblings or [])
selected = [
    item for item in siblings
    if not patterns or any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in patterns)
]
if patterns:
    missing = [
        pattern for pattern in patterns
        if not any(fnmatch.fnmatch(item.rfilename, pattern) for item in siblings)
    ]
    if missing:
        raise SystemExit(f"required remote files are missing from {repo_id}: {missing}")
known_bytes = sum((getattr(item, "size", None) or 0) for item in selected)
unknown_sizes = sum(getattr(item, "size", None) is None for item in selected)
suffix = f", {unknown_sizes} unknown sizes" if unknown_sizes else ""
print(f"{repo_type:7s} {repo_id} @ {info.sha}: {len(selected)} files, {known_bytes / 1024**3:.2f} GiB{suffix}")
' "$@"
}

download_models=false
download_datasets=false
download_simulator_assets=false
case "$scope" in
  all)
    download_models=true
    download_datasets=true
    download_simulator_assets=true
    ;;
  models)
    download_models=true
    ;;
  datasets)
    download_datasets=true
    ;;
  simulators)
    download_simulator_assets=true
    ;;
esac

if [[ "$download_models" == true ]]; then
  if ! "${hf_cmd[@]}" auth whoami >/dev/null 2>&1; then
    die "Hugging Face authentication is required for gated DINOv3 models; run: hf auth login"
  fi

  echo "[models] Checking access to both gated DINOv3 snapshots"
  preflight_hf_repo model facebook/dinov3-vitb16-pretrain-lvd1689m \
    "$dinov3_b_revision" config.json
  preflight_hf_repo model facebook/dinov3-vitl16-pretrain-lvd1689m \
    "$dinov3_l_revision" config.json
fi

if [[ "$hf_dry_run" == true ]]; then
  echo "[dry-run] Remote files and transfer sizes at the pinned revisions"
  if [[ "$download_models" == true ]]; then
    preflight_hf_repo model H-EmbodVis/TurboVLA "$release_revision"
    preflight_hf_repo model facebook/dinov3-vitb16-pretrain-lvd1689m "$dinov3_b_revision"
    preflight_hf_repo model facebook/dinov3-vitl16-pretrain-lvd1689m "$dinov3_l_revision"
    preflight_hf_repo model google-bert/bert-base-uncased "$bert_revision" \
      config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt
    preflight_hf_repo model ShilongLiu/GroundingDINO "$groundingdino_revision" \
      groundingdino_swint_ogc.pth
  fi
  if [[ "$download_datasets" == true ]]; then
    preflight_hf_repo dataset openvla/modified_libero_rlds "$libero_revision"
    preflight_hf_repo dataset StarVLA/RoboTwin-Clean "$robotwin_revision"
  fi
  if [[ "$download_simulator_assets" == true ]]; then
    preflight_hf_repo dataset TianxingChen/RoboTwin2.0 "$robotwin_assets_revision" \
      background_texture.zip embodiments.zip objects.zip
  fi
  echo "Hugging Face dry-run completed; no file payloads were downloaded."
  exit 0
fi

if [[ "$download_models" == true ]]; then
  echo "[models] TurboVLA release"
  run_hf_download H-EmbodVis/TurboVLA \
    --revision "$release_revision" \
    --local-dir "$release_dir" \
    --max-workers "$max_workers"

  echo "[models] DINOv3 ViT-B"
  run_hf_download facebook/dinov3-vitb16-pretrain-lvd1689m \
    --revision "$dinov3_b_revision" \
    --local-dir "$dinov3_b_dir" \
    --max-workers "$max_workers"

  echo "[models] DINOv3 ViT-L"
  run_hf_download facebook/dinov3-vitl16-pretrain-lvd1689m \
    --revision "$dinov3_l_revision" \
    --local-dir "$dinov3_l_dir" \
    --max-workers "$max_workers"

  echo "[models] BERT base uncased (PyTorch files only)"
  run_hf_download google-bert/bert-base-uncased \
    config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt \
    --revision "$bert_revision" \
    --local-dir "$bert_dir" \
    --max-workers "$max_workers"

  echo "[models] GroundingDINO Swin-T OGC initialization"
  run_hf_download ShilongLiu/GroundingDINO \
    groundingdino_swint_ogc.pth \
    --revision "$groundingdino_revision" \
    --local-dir "$groundingdino_dir" \
    --max-workers "$max_workers"
fi

if [[ "$download_datasets" == true ]]; then
  echo "[datasets] Modified LIBERO RLDS (four no-noop suites)"
  run_hf_download openvla/modified_libero_rlds \
    --repo-type dataset \
    --revision "$libero_revision" \
    --local-dir "$libero_dir" \
    --max-workers "$max_workers"

  echo "[datasets] RoboTwin Clean (50 nested LeRobot v2.1 datasets)"
  run_hf_download StarVLA/RoboTwin-Clean \
    --repo-type dataset \
    --revision "$robotwin_revision" \
    --local-dir "$robotwin_clean_dir" \
    --max-workers "$max_workers"
fi

if [[ "$download_simulator_assets" == true ]]; then
  echo "[simulators] RoboTwin textures, embodiments, and objects"
  run_hf_download TianxingChen/RoboTwin2.0 \
    background_texture.zip embodiments.zip objects.zip \
    --repo-type dataset \
    --revision "$robotwin_assets_revision" \
    --local-dir "$robotwin_assets_dir" \
    --max-workers "$max_workers"
fi

if [[ "$download_models" == true ]]; then
  echo "[verify] TurboVLA release checksums"
  (
    cd "$release_dir"
    sha256sum -c CHECKSUMS.sha256
  )

  echo "${groundingdino_sha256}  ${groundingdino_dir}/groundingdino_swint_ogc.pth" | sha256sum -c -

  for required_file in \
    "${release_dir}/checkpoints/libero/object.pth" \
    "${release_dir}/checkpoints/libero/goal.pth" \
    "${release_dir}/checkpoints/libero/spatial.pth" \
    "${release_dir}/checkpoints/libero/long.pth" \
    "${release_dir}/checkpoints/robotwin/steps_55000_ema_model.safetensors" \
    "${release_dir}/config.yaml" \
    "${release_dir}/dataset_statistics.json" \
    "${release_dir}/libero_all4_stats.json" \
    "${dinov3_b_dir}/config.json" \
    "${dinov3_b_dir}/preprocessor_config.json" \
    "${dinov3_b_dir}/model.safetensors" \
    "${dinov3_l_dir}/config.json" \
    "${dinov3_l_dir}/preprocessor_config.json" \
    "${dinov3_l_dir}/model.safetensors" \
    "${bert_dir}/config.json" \
    "${bert_dir}/model.safetensors" \
    "${bert_dir}/tokenizer.json" \
    "${bert_dir}/tokenizer_config.json" \
    "${bert_dir}/vocab.txt"; do
    [[ -f "$required_file" ]] || die "required model file is missing: $required_file"
  done
fi

if [[ "$download_datasets" == true ]]; then
  for suite in libero_10_no_noops libero_goal_no_noops libero_object_no_noops libero_spatial_no_noops; do
    [[ -d "${libero_dir}/${suite}/1.0.0" ]] || die "LIBERO suite is missing: ${suite}/1.0.0"
  done

  robotwin_registry="${repo_root}/experiments/robotwin/data_registry/data_config.py"
  robotwin_task_output="$("${hf_python_cmd[@]}" -c '
import ast
import sys
from pathlib import Path

tree = ast.parse(Path(sys.argv[1]).read_text(encoding="utf-8"))
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "_CLEAN50_TASKS"
        for target in node.targets
    ):
        tasks = ast.literal_eval(node.value)
        if len(tasks) != 50:
            raise SystemExit(f"expected 50 RoboTwin tasks, registry contains {len(tasks)}")
        print("\n".join(tasks))
        break
else:
    raise SystemExit("could not find _CLEAN50_TASKS in the RoboTwin registry")
' "$robotwin_registry")"
  mapfile -t robotwin_tasks <<<"$robotwin_task_output"
  for task_name in "${robotwin_tasks[@]}"; do
    [[ -f "${robotwin_clean_dir}/${task_name}/meta/info.json" ]] \
      || die "RoboTwin task is missing or malformed: Clean/${task_name}"
  done
fi

if [[ "$download_simulator_assets" == true ]]; then
  echo "${robotwin_background_sha256}  ${robotwin_assets_dir}/background_texture.zip" | sha256sum -c -
  echo "${robotwin_embodiments_sha256}  ${robotwin_assets_dir}/embodiments.zip" | sha256sum -c -
  echo "${robotwin_objects_sha256}  ${robotwin_assets_dir}/objects.zip" | sha256sum -c -
fi

# Record only successfully verified scopes. Merge by repository name so that a
# later `models` or `datasets` resume keeps entries from the other scope.
manifest_path="${manifest_dir}/hf_revisions.tsv"
manifest_selected="${manifest_dir}/hf_revisions.selected.tmp.$$"
manifest_tmp="${manifest_path}.tmp.$$"
{
  [[ "$download_models" == true ]] && printf 'model\tH-EmbodVis/TurboVLA\t%s\t%s\n' "$release_revision" "$release_dir"
  [[ "$download_models" == true ]] && printf 'model\tfacebook/dinov3-vitb16-pretrain-lvd1689m\t%s\t%s\n' "$dinov3_b_revision" "$dinov3_b_dir"
  [[ "$download_models" == true ]] && printf 'model\tfacebook/dinov3-vitl16-pretrain-lvd1689m\t%s\t%s\n' "$dinov3_l_revision" "$dinov3_l_dir"
  [[ "$download_models" == true ]] && printf 'model\tgoogle-bert/bert-base-uncased\t%s\t%s\n' "$bert_revision" "$bert_dir"
  [[ "$download_models" == true ]] && printf 'model\tShilongLiu/GroundingDINO\t%s\t%s\n' "$groundingdino_revision" "$groundingdino_dir"
  [[ "$download_datasets" == true ]] && printf 'dataset\topenvla/modified_libero_rlds\t%s\t%s\n' "$libero_revision" "$libero_dir"
  [[ "$download_datasets" == true ]] && printf 'dataset\tStarVLA/RoboTwin-Clean\t%s\t%s\n' "$robotwin_revision" "$robotwin_clean_dir"
  [[ "$download_simulator_assets" == true ]] && printf 'simulator-assets\tTianxingChen/RoboTwin2.0\t%s\t%s\n' "$robotwin_assets_revision" "$robotwin_assets_dir"
} >"$manifest_selected"
{
  [[ -f "$manifest_path" ]] && awk -F '\t' 'NR > 1 && NF >= 4' "$manifest_path"
  cat "$manifest_selected"
} | awk -F '\t' '
  BEGIN {OFS="\t"; print "kind", "repo", "revision", "local_dir"}
  NF >= 4 {
    if (!seen[$2]++) order[++count]=$2
    row[$2]=$0
  }
  END {for (i=1; i<=count; i++) print row[order[i]]}
' >"$manifest_tmp"
mv "$manifest_tmp" "$manifest_path"
rm -f "$manifest_selected"

echo
echo "TurboVLA assets are ready."
du -sh "$store"/* 2>/dev/null || true
cat <<EOF

Runtime paths:
  export TURBOVLA_STORE="$store"
  export ROBOTWIN_DATA_ROOT="$store/datasets/robotwin"
  export BERT_MODEL_PATH="$bert_dir"
  export DINOV3_MODEL_PATH="$dinov3_l_dir"
  export TURBOVLA_INIT_CKPT="$groundingdino_dir/groundingdino_swint_ogc.pth"
EOF
