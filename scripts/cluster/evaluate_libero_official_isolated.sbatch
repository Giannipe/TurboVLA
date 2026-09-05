#!/bin/bash
#SBATCH --job-name=eval_turbovla_isolated_libero
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

repo_root="${TURBOVLA_REPO:-/home/gpepe/ws/TurboVLA}"
store="${TURBOVLA_STORE:-${SCRATCH_FLASH:-/mnt/beegfs/gpepe}/TurboVLA}"
python_bin="${TURBOVLA_LIBERO_PYTHON:-/home/gpepe/miniconda3/envs/turbovla-libero/bin/python}"
suite="${1:-}"

case "$suite" in
    libero_spatial) checkpoint_name="spatial.pth" ;;
    libero_object)  checkpoint_name="object.pth" ;;
    libero_goal)    checkpoint_name="goal.pth" ;;
    libero_10)      checkpoint_name="long.pth" ;;
    *)
        echo "Usage: sbatch $0 {libero_spatial|libero_object|libero_goal|libero_10}" >&2
        exit 2
        ;;
esac

run_id="${TURBOVLA_LIBERO_EVAL_RUN_ID:-isolated-job-${SLURM_JOB_ID:-manual}}"
checkpoint="${store}/pretrained/TurboVLA/checkpoints/libero/${checkpoint_name}"
dinov3_path="${store}/models/dinov3-vitb16"
bert_path="${store}/models/bert-base-uncased"
stats_path="${repo_root}/experiments/libero/configs/libero_all4_stats.json"
libero_root="${store}/simulators/LIBERO"

if [[ -n "${TURBOVLA_LIBERO_EVAL_ROOT:-}" ]]; then
    result_root="$TURBOVLA_LIBERO_EVAL_ROOT"
else
    result_root="${store}/results/libero_official_eval_isolated/${run_id}"
fi

renderer="${LIBERO_RENDERER:-egl}"
case "$renderer" in
    egl|osmesa) ;;
    *)
        echo "ERROR: LIBERO_RENDERER must be egl or osmesa, got: ${renderer}" >&2
        exit 1
        ;;
esac

for required in \
    "$python_bin" \
    "$checkpoint" \
    "${dinov3_path}/model.safetensors" \
    "${bert_path}/model.safetensors" \
    "$stats_path" \
    "${libero_root}/libero/libero" \
    "${repo_root}/scripts/libero/aggregate_results.py"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: required path is missing: ${required}" >&2
        exit 1
    fi
done

task_result_dir="${result_root}/tasks/${suite}"
task_log_dir="${result_root}/logs/${suite}"
video_dir="${result_root}/videos"
numba_cache="${TMPDIR:-/tmp}/turbovla-numba-${SLURM_JOB_ID:-manual}"
matplotlib_cache="${TMPDIR:-/tmp}/turbovla-matplotlib-${SLURM_JOB_ID:-manual}"
mkdir -p \
    /home/gpepe/ws/logs \
    "$task_result_dir" \
    "$task_log_dir" \
    "$video_dir" \
    "$numba_cache" \
    "$matplotlib_cache"

export LIBERO_CONFIG_PATH="${store}/config/libero"
export NUMBA_CACHE_DIR="$numba_cache"
export MPLCONFIGDIR="$matplotlib_cache"
export MUJOCO_GL="$renderer"
export PYOPENGL_PLATFORM="$renderer"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

echo "[job] host=$(hostname)"
echo "[job] started=$(date --iso-8601=seconds)"
echo "[job] slurm_job_id=${SLURM_JOB_ID:-manual} run_id=${run_id}"
echo "[job] suite=${suite} checkpoint=${checkpoint}"
echo "[job] isolation=one Python process per task"
echo "[job] result_root=${result_root}"
echo "[job] renderer=${renderer}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

cd "$repo_root"
task_results=()
for task_id in {0..9}; do
    task_result="${task_result_dir}/task_${task_id}.json"
    task_log="${task_log_dir}/task_${task_id}.log"
    task_results+=("$task_result")

    if [[ -e "$task_result" ]]; then
        echo "ERROR: refusing to overwrite existing task result: ${task_result}" >&2
        exit 1
    fi

    echo "[task ${task_id}] started=$(date --iso-8601=seconds)"
    "$python_bin" -u experiments/libero/evaluate.py \
        --ckpt_path "$checkpoint" \
        --libero_root "$libero_root" \
        --dinov3_path "$dinov3_path" \
        --bert_path "$bert_path" \
        --stats_path "$stats_path" \
        --stats_key libero_all4_no_noops \
        --normalize_binary_gripper auto \
        --allow_hf_download false \
        --task_suite_name "$suite" \
        --task_ids "$task_id" \
        --num_trials_per_task 50 \
        --num_steps_wait 10 \
        --chunk_size 12 \
        --num_open_loop_steps 12 \
        --env_img_res 256 \
        --seed 7 \
        --control_mode relative \
        --mujoco_gl "$renderer" \
        --pyopengl_platform "$renderer" \
        --precision bf16 \
        --dinov3_output_hidden_states true \
        --save_video false \
        --video_out_path "$video_dir" \
        --result_json_path "$task_result" \
        --log_path "$task_log"
    echo "[task ${task_id}] finished=$(date --iso-8601=seconds) result=${task_result}"
done

suite_result="${result_root}/${suite}.json"
if [[ -e "$suite_result" ]]; then
    echo "ERROR: refusing to overwrite existing suite result: ${suite_result}" >&2
    exit 1
fi

"$python_bin" scripts/libero/aggregate_results.py \
    "${task_results[@]}" \
    --output "$suite_result"

echo "[job] suite_result=${suite_result}"
echo "[job] finished=$(date --iso-8601=seconds)"
