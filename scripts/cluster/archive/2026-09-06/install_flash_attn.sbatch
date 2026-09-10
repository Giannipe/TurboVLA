#!/usr/bin/env bash
#SBATCH --job-name=turbovla-flash
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=turbovla-flash-%j.out
#SBATCH --error=turbovla-flash-%j.out

set -euo pipefail

repo_root="${TURBOVLA_REPO:-/home/gpepe/ws/TurboVLA}"

module load nvhpc-hpcx-cuda12/25.1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FLASH_ATTN_MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

echo "[job] host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version

cd "$repo_root"
bash scripts/cluster/create_conda_envs.sh robotwin --with-flash-attn
