#!/bin/bash
#SBATCH --job-name=turbovla-libero-train
#SBATCH --output=/home/gpepe/ws/logs/%x_%j.out
#SBATCH --error=/home/gpepe/ws/logs/%x_%j.err
#SBATCH --time=20:50:00
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
# Usage: sbatch .../train_libero.sbatch [upstream128|paper256] [full|ddp-smoke] [new|resume] [run_dir]
# Does not submit follow-up jobs or evaluations automatically.
set -euo pipefail
umask 027
[[ $# -le 4 ]] || { echo 'Expected at most four arguments; see usage in this file' >&2; exit 2; }
recipe="${1:-upstream128}"
mode="${2:-full}"
operation="${3:-new}"
repo_root="${TURBOVLA_REPO:-/home/gpepe/ws/TurboVLA}"
export TURBOVLA_STORE="${TURBOVLA_STORE:-${SCRATCH_FLASH:-/mnt/beegfs/gpepe}/TurboVLA}"
python_bin=/home/gpepe/miniconda3/envs/turbovla-libero/bin/python
[[ "$recipe" == upstream128 || "$recipe" == paper256 ]] || { echo 'Invalid recipe' >&2; exit 2; }
case "$mode" in full) max_steps=80000 ;; ddp-smoke) max_steps=3 ;; *) echo 'Invalid mode' >&2; exit 2 ;; esac
[[ "$operation" == new || "$operation" == resume ]] || { echo 'Invalid operation' >&2; exit 2; }
job_id="${SLURM_JOB_ID:?Submit with sbatch}"
run_dir="${4:-$TURBOVLA_STORE/training/libero_${recipe}_${mode}/job-$job_id}"
[[ "$run_dir" == /* ]] || { echo 'run_dir must be absolute' >&2; exit 2; }
if [[ "$operation" == new ]]; then
    [[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
    mkdir -p "$run_dir/checkpoints" "$run_dir/attempts"
    resume_mode=none
else
    [[ $# -eq 4 && -f "$run_dir/recipe.txt" && -f "$run_dir/assets.json" ]] || {
        echo 'Resume requires an existing run_dir as fourth argument' >&2; exit 2;
    }
    [[ "$(< "$run_dir/recipe.txt")" == "$recipe $mode" ]] || { echo 'Recipe mismatch' >&2; exit 1; }
    compgen -G "$run_dir/checkpoints/turbovla_step_*.pth" >/dev/null || { echo 'No checkpoint to resume' >&2; exit 1; }
    resume_mode=all
fi
exec 9>"$run_dir/.training.lock"
flock -n 9 || { echo 'Another process is using this run_dir' >&2; exit 1; }
attempt="$run_dir/attempts/$job_id"
mkdir "$attempt"
source_root="$attempt/source"
mkdir -p "$source_root/experiments" "$source_root/scripts/cluster"
# Freeze the code used by this attempt, including the local EMA-resume fix.
cp -a "$repo_root/turbovla" "$source_root/"
cp -a "$repo_root/experiments/libero" "$source_root/experiments/"
cp "$repo_root/scripts/cluster/"{libero_training_recipe.sh,libero_training_preflight.py,train_libero.sbatch} "$source_root/scripts/cluster/"
export PYTHONPATH="$source_root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$attempt/matplotlib" NUMBA_CACHE_DIR="$attempt/numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
cd "$source_root"
echo "run_dir=$run_dir recipe=$recipe mode=$mode operation=$operation"
git -C "$repo_root" rev-parse HEAD > "$attempt/git_head.txt"
git -C "$repo_root" diff -- turbovla scripts/cluster > "$attempt/local_changes.patch"
"$python_bin" -m pip freeze > "$attempt/pip-freeze.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$attempt/gpus.txt"
"$python_bin" -c 'import torch; assert torch.cuda.device_count() == 4, "Exactly four visible GPUs required"; assert torch.cuda.is_bf16_supported(), "BF16 required"'
"$python_bin" -u scripts/cluster/libero_training_preflight.py --store "$TURBOVLA_STORE" --repo "$source_root" --output "$attempt/assets.json"
if [[ "$operation" == new ]]; then
    cp "$attempt/assets.json" "$run_dir/assets.json"
    printf '%s %s\n' "$recipe" "$mode" > "$run_dir/recipe.txt"
else
    # Excludes paths/timestamps, so an identical snapshot can be resumed elsewhere.
    cmp "$run_dir/assets.json" "$attempt/assets.json" || { echo 'Code, assets or environment changed: refusing resume' >&2; exit 1; }
fi
source scripts/cluster/libero_training_recipe.sh
command=("$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4
         experiments/libero/train.py "${train_args[@]}")
printf '%q ' "${command[@]}" > "$attempt/command.sh"
printf '\n' >> "$attempt/command.sh"
"${command[@]}" 2>&1 | tee "$attempt/train.log"
echo "Completed target $max_steps. Checkpoints: $run_dir/checkpoints"
