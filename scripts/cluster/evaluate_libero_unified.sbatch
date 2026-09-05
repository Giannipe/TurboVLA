#!/bin/bash
#SBATCH --job-name=eval_turbovla_unified_libero
#SBATCH --output=/home/gpepe/ws/logs/%x_%j.out
#SBATCH --error=/home/gpepe/ws/logs/%x_%j.err
#SBATCH --time=20:50:00
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail
umask 027
cd /home/gpepe/ws/TurboVLA
store="${TURBOVLA_STORE:-${SCRATCH_FLASH:-/mnt/beegfs/gpepe}/TurboVLA}"
suite="${1:?Pass a LIBERO suite}"
case "$suite" in libero_spatial|libero_object|libero_goal|libero_10) ;; *) exit 2 ;; esac
export TURBOVLA_LIBERO_EVAL_ROOT="${store}/results/libero_unified_eval/${TURBOVLA_LIBERO_EVAL_RUN_ID:?Missing run ID}"
export TURBOVLA_LIBERO_CHECKPOINT="${store}/pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth"
export TURBOVLA_LIBERO_PREFLIGHT=true
# Same GPU, renderer, Python environment and evaluator as the old-release baseline.
export LIBERO_RENDERER=egl
export TURBOVLA_LIBERO_PYTHON=/home/gpepe/miniconda3/envs/turbovla-libero/bin/python
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ -e "${TURBOVLA_LIBERO_EVAL_ROOT}/${suite}.json" || -e "${TURBOVLA_LIBERO_EVAL_ROOT}/logs/${suite}.log" ]]; then
    echo "ERROR: existing suite artifacts; choose a new run ID" >&2
    exit 1
fi
echo "[provenance] model/evaluator code commit: $(git rev-parse HEAD)"
git diff --exit-code HEAD -- turbovla third_party/vla_adapter experiments/libero
"$TURBOVLA_LIBERO_PYTHON" -u scripts/cluster/download_libero_unified.py --verify-only
sha256sum scripts/cluster/evaluate_libero_official.sbatch turbovla/evaluation/policy.py
bash scripts/cluster/evaluate_libero_official.sbatch "$suite"
