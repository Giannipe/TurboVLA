#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Create the two Python 3.10 environments used by TurboVLA.

Usage:
  bash scripts/cluster/create_conda_envs.sh [libero|robotwin|all] [options]

Options:
  --with-flash-attn  Install flash-attn in the RoboTwin environment.
                     Run this on a node with a compatible CUDA toolchain, or
                     where the matching prebuilt flash-attn wheel is reachable.
  -h, --help         Show this help.

Environment overrides:
  TURBOVLA_LIBERO_ENV             default: turbovla-libero
  TURBOVLA_ROBOTWIN_ENV           default: turbovla-robotwin
  PYTORCH_LIBERO_INDEX_URL        default: https://download.pytorch.org/whl/cu121
  PYTORCH_ROBOTWIN_INDEX_URL      default: https://download.pytorch.org/whl/cu124
  FLASH_ATTN_MAX_JOBS             default: 8

Existing environments are reused and checked; they are never removed.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

target="${1:-all}"
case "$target" in
  libero|robotwin|all)
    shift || true
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    die "unknown target: $target"
    ;;
esac

with_flash_attn=false
while (($#)); do
  case "$1" in
    --with-flash-attn)
      with_flash_attn=true
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

libero_env="${TURBOVLA_LIBERO_ENV:-turbovla-libero}"
robotwin_env="${TURBOVLA_ROBOTWIN_ENV:-turbovla-robotwin}"
libero_torch_index="${PYTORCH_LIBERO_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
robotwin_torch_index="${PYTORCH_ROBOTWIN_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
flash_attn_max_jobs="${FLASH_ATTN_MAX_JOBS:-8}"

command -v conda >/dev/null 2>&1 || die "conda is not available on PATH"
conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"

env_exists() {
  local name="$1"
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$name"
}

ensure_env() {
  local name="$1"
  if env_exists "$name"; then
    echo "[conda] Reusing existing environment: $name"
  else
    echo "[conda] Creating environment: $name (Python 3.10)"
    conda create --name "$name" python=3.10 pip -y
  fi

  local py_version
  py_version="$(conda run -n "$name" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$py_version" == "3.10" ]] || die "$name uses Python $py_version; Python 3.10 is required"
}

run_in_env() {
  local name="$1"
  shift
  conda run --no-capture-output -n "$name" "$@"
}

check_robotwin_requirements() {
  local name="$1"
  local env_prefix=""
  local check_output=""
  local check_status=0
  local unexpected=""

  # Invoke the environment's interpreter directly here: on a non-zero exit,
  # `conda run` adds its own ERROR wrapper to pip's output and would make the
  # narrowly-scoped exception below look like a second dependency failure.
  env_prefix="$(conda env list | awk -v target="$name" '$1 == target {print $NF; exit}')"
  [[ -n "$env_prefix" && -x "$env_prefix/bin/python" ]] \
    || die "Could not resolve the Python interpreter for Conda environment: $name"
  check_output="$("$env_prefix/bin/python" -m pip check 2>&1)" || check_status=$?
  if ((check_status == 0)); then
    printf '%s\n' "$check_output"
    return 0
  fi

  # pipablepytorch3d 0.7.6 currently publishes a wheel whose internal WHEEL
  # tag says cp311 even when installed for Python 3.10. `pip check` therefore
  # reports it as unsupported. TurboVLA only imports pytorch3d.transforms,
  # which is pure Python; verify that exact runtime surface and reject every
  # other dependency error.
  unexpected="$(printf '%s\n' "$check_output" \
    | sed '/^pipablepytorch3d 0\.7\.6 is not supported on this platform$/d' \
    | sed '/^WARNING: The directory .*cache\/pip/d' \
    | sed '/^[[:space:]]*$/d')"
  [[ -z "$unexpected" ]] || {
    printf '%s\n' "$check_output" >&2
    die "RoboTwin environment has broken requirements"
  }

  echo "[pip-check] Ignoring the known pipablepytorch3d wheel-tag defect after a functional transforms test."
  run_in_env "$name" python -c \
    'import torch; from pytorch3d.transforms import quaternion_to_matrix; out = quaternion_to_matrix(torch.tensor([[1.0, 0.0, 0.0, 0.0]])); assert tuple(out.shape) == (1, 3, 3); print("pytorch3d.transforms OK")'
}

install_common_tools() {
  local name="$1"
  run_in_env "$name" python -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
  run_in_env "$name" python -m pip install --disable-pip-version-check \
    "huggingface_hub[hf_xet]==0.35.3"
}

install_libero() {
  echo "============================================================"
  echo "Installing TurboVLA LIBERO environment: $libero_env"
  echo "Repository: $repo_root"
  echo "PyTorch index: $libero_torch_index"
  echo "============================================================"

  ensure_env "$libero_env"
  install_common_tools "$libero_env"

  run_in_env "$libero_env" python -m pip install --disable-pip-version-check --upgrade \
    torch==2.3.1 torchvision==0.18.1 \
    --index-url "$libero_torch_index"

  # Match the versions recorded in experiments/libero/README.md. Pinning these
  # avoids silently moving to a newer TensorFlow/Transformers release on a
  # later rerun of this setup script.
  run_in_env "$libero_env" python -m pip install --disable-pip-version-check --upgrade \
    numpy==1.26.4 \
    mujoco==2.3.7 \
    matplotlib==3.10.1 \
    transformers==4.56.0 \
    tensorflow==2.20.0 \
    tensorflow-datasets==4.9.3

  run_in_env "$libero_env" python -m pip install --disable-pip-version-check --upgrade \
    --upgrade-strategy only-if-needed \
    -e "${repo_root}[libero]"

  run_in_env "$libero_env" python -m pip check
  run_in_env "$libero_env" python -c \
    'import importlib.metadata as m, torch, turbovla; print("torch", torch.__version__, "cuda_build", torch.version.cuda, "cuda_available", torch.cuda.is_available()); print("transformers", m.version("transformers")); print("tensorflow", m.version("tensorflow")); print("tensorflow-datasets", m.version("tensorflow-datasets")); print("mujoco", m.version("mujoco"))'
}

install_robotwin() {
  echo "============================================================"
  echo "Installing TurboVLA RoboTwin policy environment: $robotwin_env"
  echo "Repository: $repo_root"
  echo "PyTorch index: $robotwin_torch_index"
  echo "============================================================"

  ensure_env "$robotwin_env"
  install_common_tools "$robotwin_env"

  # torchvision 0.21.0 is pinned by pyproject.toml and is the official match
  # for torch 2.6.0. CUDA 12.4 wheels run with the newer drivers used by the
  # cluster's A40/A100/H200 nodes.
  run_in_env "$robotwin_env" python -m pip install --disable-pip-version-check --upgrade \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url "$robotwin_torch_index"

  # scripts/robotwin/requirements.txt records these two versions explicitly.
  run_in_env "$robotwin_env" python -m pip install --disable-pip-version-check --upgrade \
    numpy==1.26.4 transformers==4.57.0

  run_in_env "$robotwin_env" python -m pip install --disable-pip-version-check --upgrade \
    --upgrade-strategy only-if-needed \
    -e "${repo_root}[robotwin]"

  if [[ "$with_flash_attn" == true ]]; then
    echo "[flash-attn] Installing flash-attn==2.7.4.post1"
    run_in_env "$robotwin_env" python -m pip install --disable-pip-version-check ninja packaging
    run_in_env "$robotwin_env" env MAX_JOBS="$flash_attn_max_jobs" \
      python -m pip install --disable-pip-version-check \
      flash-attn==2.7.4.post1 --no-build-isolation
    run_in_env "$robotwin_env" python -c \
      'import flash_attn; print("flash-attn", flash_attn.__version__, "import OK")'
  else
    cat <<EOF
[flash-attn] Deferred.
Complete the RoboTwin training environment from a suitable compute/build node:

  bash scripts/cluster/create_conda_envs.sh robotwin --with-flash-attn
EOF
  fi

  check_robotwin_requirements "$robotwin_env"
  run_in_env "$robotwin_env" python -c \
    'import importlib.metadata as m, torch, turbovla; print("torch", torch.__version__, "cuda_build", torch.version.cuda, "cuda_available", torch.cuda.is_available()); print("torchvision", m.version("torchvision")); print("transformers", m.version("transformers")); print("deepspeed", m.version("deepspeed")); print("accelerate", m.version("accelerate"))'
}

case "$target" in
  libero)
    install_libero
    ;;
  robotwin)
    install_robotwin
    ;;
  all)
    install_libero
    install_robotwin
    ;;
esac

echo
echo "Environment setup completed."
echo "  LIBERO:  conda activate $libero_env"
echo "  RoboTwin: conda activate $robotwin_env"
