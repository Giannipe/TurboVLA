#!/bin/bash
#SBATCH --job-name=turbovla-upstream-preflight
#SBATCH --output=/home/gpepe/ws/logs/%x_%j.out
#SBATCH --error=/home/gpepe/ws/logs/%x_%j.err
#SBATCH --partition=cpu_sapphire
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
umask 027
repo_root="${TURBOVLA_REPO:?Set TURBOVLA_REPO to the updated worktree}"
store="${TURBOVLA_STORE:-${SCRATCH_FLASH:-/mnt/beegfs/gpepe}/TurboVLA}"
python_bin=/home/gpepe/miniconda3/envs/turbovla-libero/bin/python
cd "$repo_root"
# No pip reinstall: the active evaluations keep their original editable install.
export PYTHONPATH="${repo_root}:${repo_root}/third_party/vla_adapter"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/turbovla-upstream-numba-${SLURM_JOB_ID:-manual}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/turbovla-upstream-mpl-${SLURM_JOB_ID:-manual}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
echo "[preflight] upstream worktree=$repo_root commit=$(git rev-parse HEAD)"
"$python_bin" -c 'import os; from pathlib import Path; import turbovla, vla_adapter; from turbovla.evaluation import policy; root=Path(os.environ["TURBOVLA_REPO"]).resolve(); assert Path(turbovla.__file__).resolve().is_relative_to(root); assert Path(vla_adapter.__file__).resolve().is_relative_to(root); assert hasattr(policy, "_RELEASED_LIBERO_EXPORT_SHA256"); print("policy source", policy.__file__); print("rollout source", vla_adapter.__file__)'
"$python_bin" -m unittest discover -s tests -p test_libero_export_compat.py -v
"$python_bin" -u scripts/cluster/download_libero_unified.py --verify-only
"$python_bin" -u experiments/libero/evaluate.py \
    --ckpt_path "${store}/pretrained/TurboVLA-unified-cb53005/checkpoints/libero/turbovla_libero.pth" \
    --dinov3_path "${store}/models/dinov3-vitb16" \
    --bert_path "${store}/models/bert-base-uncased" \
    --stats_path "${repo_root}/experiments/libero/configs/libero_all4_stats.json" \
    --stats_key libero_all4_no_noops --normalize_binary_gripper auto \
    --allow_hf_download false --task_suite_name libero_object \
    --num_trials_per_task 50 --chunk_size 12 --num_open_loop_steps 12 \
    --seed 7 --precision bf16 --dry_run_model_load true \
    --log_path "/home/gpepe/ws/logs/upstream-preflight-${SLURM_JOB_ID:-manual}.log"
echo "[preflight] strict model load passed; no simulator rollouts or training performed"
