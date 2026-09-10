#!/bin/bash
#SBATCH --job-name=turbovla-training-smoke
#SBATCH --output=/home/gpepe/ws/logs/%x_%j.out
#SBATCH --error=/home/gpepe/ws/logs/%x_%j.err
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
umask 027
repo_root="${TURBOVLA_REPO:-/home/gpepe/ws/TurboVLA}"
export TURBOVLA_STORE="${TURBOVLA_STORE:-${SCRATCH_FLASH:-/mnt/beegfs/gpepe}/TurboVLA}"
run_dir="${TURBOVLA_STORE}/training/libero_smoke/job-${SLURM_JOB_ID:?Submit with sbatch}"
python_bin=/home/gpepe/miniconda3/envs/turbovla-libero/bin/python
[[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
mkdir -p "$run_dir/checkpoints" "$run_dir/evaluation"
cd "$repo_root"
export PYTHONPATH="${repo_root}:${repo_root}/third_party/vla_adapter"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
export LIBERO_CONFIG_PATH="${TURBOVLA_STORE}/config/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/turbovla-train-numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/turbovla-train-mpl-${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
echo "[smoke] run_dir=$run_dir git=$(git rev-parse HEAD)"
git diff --stat -- turbovla tests scripts/cluster
git diff -- turbovla tests scripts/cluster > "$run_dir/source_changes.patch"
sha256sum turbovla/training/{trainer,train_mixed,pi05}.py turbovla/data/{mixed_suite,libero_rlds}.py \
    scripts/cluster/{training_smoke_checks.py,train_libero_smoke.sbatch} \
    | tee "$run_dir/source_sha256.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$python_bin" -m unittest discover -s tests -p 'test_training_ema_resume.py' -v
"$python_bin" -u scripts/cluster/training_smoke_checks.py data --run-dir "$run_dir"

datasets="${TURBOVLA_STORE}/datasets/libero"
train_args=(
    --dataset_dir "${datasets}/libero_10_no_noops/1.0.0"
    --dataset_dirs "${datasets}/libero_10_no_noops/1.0.0,${datasets}/libero_goal_no_noops/1.0.0,${datasets}/libero_object_no_noops/1.0.0,${datasets}/libero_spatial_no_noops/1.0.0"
    --stats_path "${repo_root}/experiments/libero/configs/libero_all4_stats.json"
    --stats_key libero_all4_no_noops --normalize_binary_gripper auto
    --dinov3_path "${TURBOVLA_STORE}/models/dinov3-vitb16"
    --bert_path "${TURBOVLA_STORE}/models/bert-base-uncased"
    --pretrained_init_ckpt "${TURBOVLA_STORE}/models/groundingdino/groundingdino_swint_ogc.pth"
    --checkpoint_dir "$run_dir/checkpoints" --checkpoint_prefix smoke
    --batch_size 8 --grad_accum_steps 16 --num_workers 0
    --lr_schedule_steps 80000 --warmup_steps 10000 --save_steps 2 --log_freq 1
    --precision fp32 --dinov3_precision bf16_autocast --seed 42
    --no_allow_hf_download --no_freeze_backbones --freeze_text_encoder
)
# 1 GPU x 8 samples x 16 accumulation = 128, but this does NOT test 4-GPU DDP.
# Preserve the 80k LR schedule even though this plumbing test stops at step 3.
"$python_bin" -u experiments/libero/train.py "${train_args[@]}" --max_steps 2 --resume_mode none \
    2>&1 | tee "$run_dir/train_initial.log"
"$python_bin" -u scripts/cluster/training_smoke_checks.py checkpoint --run-dir "$run_dir" --step 2
"$python_bin" -u experiments/libero/train.py "${train_args[@]}" --max_steps 3 --resume_mode all \
    2>&1 | tee "$run_dir/train_resume.log"
"$python_bin" -u scripts/cluster/training_smoke_checks.py checkpoint --run-dir "$run_dir" --step 3 \
    --previous "$run_dir/checkpoints/smoke_2.pth"

eval_args=(
    --ckpt_path "$run_dir/checkpoints/smoke_3.pth"
    --libero_root "${TURBOVLA_STORE}/simulators/LIBERO"
    --dinov3_path "${TURBOVLA_STORE}/models/dinov3-vitb16"
    --bert_path "${TURBOVLA_STORE}/models/bert-base-uncased"
    --stats_path "${repo_root}/experiments/libero/configs/libero_all4_stats.json"
    --stats_key libero_all4_no_noops --normalize_binary_gripper auto --allow_hf_download false
    --num_trials_per_task 1 --task_ids 0 --chunk_size 12 --num_open_loop_steps 12
    --num_steps_wait 10 --env_img_res 256 --seed 7 --control_mode relative
    --mujoco_gl egl --pyopengl_platform egl --precision bf16 --save_video false
)
"$python_bin" -u experiments/libero/evaluate.py "${eval_args[@]}" --task_suite_name libero_object \
    --dry_run_model_load true --log_path "$run_dir/evaluation/preflight.log"
for suite in libero_spatial libero_object libero_goal libero_10; do
    "$python_bin" -u experiments/libero/evaluate.py "${eval_args[@]}" --task_suite_name "$suite" \
        --result_json_path "$run_dir/evaluation/${suite}.json" \
        --log_path "$run_dir/evaluation/${suite}.log"
done
"$python_bin" -u scripts/cluster/training_smoke_checks.py final --run-dir "$run_dir"
echo "[smoke] PASSED: $run_dir/smoke_summary.json"
