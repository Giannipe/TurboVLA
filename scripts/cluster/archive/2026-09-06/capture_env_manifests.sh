#!/usr/bin/env bash
set -euo pipefail
umask 027

if [[ -n "${TURBOVLA_STORE:-}" ]]; then
  store="$TURBOVLA_STORE"
else
  [[ -n "${SCRATCH_FLASH:-}" ]] || {
    echo "ERROR: SCRATCH_FLASH is not defined" >&2
    exit 1
  }
  store="${SCRATCH_FLASH}/TurboVLA"
fi

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is not available on PATH" >&2
  exit 1
}

manifest_dir="${store}/manifests/environments"
mkdir -p "$manifest_dir"

for env_name in \
  "${TURBOVLA_LIBERO_ENV:-turbovla-libero}" \
  "${TURBOVLA_ROBOTWIN_ENV:-turbovla-robotwin}"; do
  conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq "$env_name" || {
    echo "ERROR: Conda environment does not exist: $env_name" >&2
    exit 1
  }

  explicit_tmp="${manifest_dir}/${env_name}.conda-explicit.txt.tmp.$$"
  pip_tmp="${manifest_dir}/${env_name}.pip-freeze.txt.tmp.$$"
  conda list -n "$env_name" --explicit >"$explicit_tmp"
  conda run -n "$env_name" python -m pip freeze --all >"$pip_tmp"
  mv "$explicit_tmp" "${manifest_dir}/${env_name}.conda-explicit.txt"
  mv "$pip_tmp" "${manifest_dir}/${env_name}.pip-freeze.txt"
done

printf 'Captured environment manifests in %s\n' "$manifest_dir"
