#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
batch_script="${script_dir}/evaluate_libero_official.sbatch"

command -v sbatch >/dev/null 2>&1 || {
    echo "ERROR: sbatch is not available" >&2
    exit 1
}

mkdir -p /home/gpepe/ws/logs

run_id="${TURBOVLA_LIBERO_EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
export TURBOVLA_LIBERO_EVAL_RUN_ID="$run_id"

echo "Submitting the four official TurboVLA LIBERO evaluations"
echo "run_id=${run_id}"

cd "$repo_root"
for suite in libero_spatial libero_object libero_goal libero_10; do
    echo "[submit] ${suite}"
    sbatch "$batch_script" "$suite"
done

store="${TURBOVLA_STORE:-${SCRATCH_FLASH:-/mnt/beegfs/gpepe}/TurboVLA}"
result_dir="${TURBOVLA_LIBERO_EVAL_ROOT:-${store}/results/libero_official_eval/${run_id}}"
cat <<EOF

Submitted four independent jobs (no Slurm array).
Result directory:
  ${result_dir}

After all four jobs finish:
  python scripts/cluster/summarize_libero_official.py "${result_dir}"
EOF
