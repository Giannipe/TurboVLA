#!/usr/bin/env bash
set -euo pipefail
umask 027

usage() {
  cat <<'EOF'
Clone, pin, install, and configure the external LIBERO simulator for TurboVLA.

Usage:
  bash scripts/cluster/setup_libero_simulator.sh [--plan]

Environment overrides:
  TURBOVLA_STORE       default: $SCRATCH_FLASH/TurboVLA
  TURBOVLA_LIBERO_ENV  default: turbovla-libero
  LIBERO_REVISION      default: 8f1084e3132a39270c3a13ebe37270a43ece2a01

The simulator is installed editable with --no-deps. TurboVLA owns the tested
dependency pins in its Conda environment; LIBERO's legacy requirements file is
intentionally not installed over them.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

plan=false
while (($#)); do
  case "$1" in
    --plan)
      plan=true
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

if [[ -n "${TURBOVLA_STORE:-}" ]]; then
  store="$TURBOVLA_STORE"
else
  [[ -n "${SCRATCH_FLASH:-}" ]] || die "SCRATCH_FLASH is not defined"
  store="${SCRATCH_FLASH}/TurboVLA"
fi

libero_env="${TURBOVLA_LIBERO_ENV:-turbovla-libero}"
libero_url="https://github.com/Lifelong-Robot-Learning/LIBERO.git"
libero_revision="${LIBERO_REVISION:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
libero_root="${store}/simulators/LIBERO"
libero_package_root="${libero_root}/libero/libero"
libero_config_dir="${store}/config/libero"
libero_raw_data_dir="${store}/datasets/libero_raw"
manifest_dir="${store}/manifests"

cat <<EOF
LIBERO simulator plan
=====================
Repository:  $libero_url
Revision:    $libero_revision
Checkout:    $libero_root
Conda env:   $libero_env
Config:      $libero_config_dir/config.yaml
Raw HDF5:    $libero_raw_data_dir
Released RLDS data remains under: $store/datasets/libero
EOF
[[ "$plan" == true ]] && exit 0

command -v conda >/dev/null 2>&1 || die "conda is not available on PATH"
conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"
conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$libero_env" \
  || die "Conda environment does not exist: $libero_env"

mkdir -p "$(dirname "$libero_root")" "$libero_config_dir" "$libero_raw_data_dir" "$manifest_dir"

if [[ ! -e "$libero_root" ]]; then
  git clone "$libero_url" "$libero_root"
elif [[ ! -d "${libero_root}/.git" ]]; then
  die "LIBERO destination exists but is not a Git checkout: $libero_root"
fi

current_revision="$(git -C "$libero_root" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "$current_revision" ]]; then
  git -C "$libero_root" fetch origin "$libero_revision"
  git -C "$libero_root" checkout --detach FETCH_HEAD
elif [[ "$current_revision" != "$libero_revision" ]]; then
  die "LIBERO checkout is at $current_revision, expected $libero_revision; refusing to overwrite it"
fi

if [[ -n "$(git -C "$libero_root" status --porcelain --untracked-files=no)" ]]; then
  die "LIBERO tracked files are modified; restore or preserve those changes before baseline setup"
fi

# The pinned upstream uses a nested libero/libero package but lacks the outer
# package marker. Modern setuptools can otherwise build an empty editable wheel.
# This compatibility shim is deliberately untracked and leaves upstream source
# files unchanged.
if [[ ! -f "${libero_root}/libero/__init__.py" ]]; then
  printf '%s\n' '# Package marker required for modern editable installs.' \
    >"${libero_root}/libero/__init__.py"
fi

conda run --no-capture-output -n "$libero_env" \
  python -m pip install --disable-pip-version-check --no-deps -e "$libero_root"

conda run --no-capture-output -n "$libero_env" python -c \
  'from pathlib import Path; import shutil, robosuite; root = Path(robosuite.__path__[0]); target = root / "macros_private.py"; target.exists() or shutil.copyfile(root / "macros.py", target); print("robosuite macros", target)'

config_tmp="${libero_config_dir}/config.yaml.tmp.$$"
{
  printf 'benchmark_root: "%s"\n' "$libero_package_root"
  printf 'bddl_files: "%s/bddl_files"\n' "$libero_package_root"
  printf 'init_states: "%s/init_files"\n' "$libero_package_root"
  printf 'datasets: "%s"\n' "$libero_raw_data_dir"
  printf 'assets: "%s/assets"\n' "$libero_package_root"
} >"$config_tmp"
mv "$config_tmp" "${libero_config_dir}/config.yaml"

LIBERO_CONFIG_PATH="$libero_config_dir" \
  conda run --no-capture-output -n "$libero_env" python -c \
  'from libero.libero import benchmark, get_libero_path; from libero.libero.envs import OffScreenRenderEnv; suite = benchmark.get_benchmark_dict()["libero_object"](); assert suite.n_tasks == 10; print("LIBERO import OK; tasks", suite.n_tasks); print("assets", get_libero_path("assets"))'

printf 'kind\trepo\trevision\tlocal_dir\n%s\t%s\t%s\t%s\n' \
  simulator "$libero_url" "$libero_revision" "$libero_root" \
  >"${manifest_dir}/libero_simulator_revision.tsv"

cat <<EOF

LIBERO simulator is ready. Export this for every rollout job:
  export LIBERO_CONFIG_PATH="$libero_config_dir"
EOF
