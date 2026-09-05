#!/bin/bash
set -euo pipefail
cd /home/gpepe/ws/TurboVLA
mkdir -p /home/gpepe/ws/logs
export TURBOVLA_STORE="${TURBOVLA_STORE:-${SCRATCH_FLASH:-/mnt/beegfs/gpepe}/TurboVLA}"
export TURBOVLA_LIBERO_EVAL_RUN_ID="${TURBOVLA_LIBERO_EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
result_root="${TURBOVLA_STORE}/results/libero_unified_eval/${TURBOVLA_LIBERO_EVAL_RUN_ID}"
if [[ -e "$result_root" ]]; then
    echo "ERROR: run directory already exists: $result_root" >&2
    exit 1
fi
mkdir -p "$result_root"
echo "run_id=${TURBOVLA_LIBERO_EVAL_RUN_ID}"
download_job="$(sbatch --parsable scripts/cluster/download_libero_unified.sbatch)"
download_job="${download_job%%;*}"
[[ "$download_job" =~ ^[0-9]+$ ]] || exit 1
echo "download_job=${download_job}"
for suite in libero_spatial libero_object libero_goal libero_10; do
    job="$(sbatch --parsable --dependency="afterok:${download_job}" --kill-on-invalid-dep=yes \
        --job-name="eval_turbovla_unified_${suite}" scripts/cluster/evaluate_libero_unified.sbatch "$suite")"
    echo "${suite}=${job}"
done
echo "results=${result_root}"
echo "After completion: python scripts/cluster/summarize_libero_official.py '${result_root}' --unified-checkpoint"
