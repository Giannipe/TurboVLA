#!/bin/bash
#SBATCH --job-name=turbovla-unified-assets
#SBATCH --output=/home/gpepe/ws/logs/%x_%j.out
#SBATCH --error=/home/gpepe/ws/logs/%x_%j.err
#SBATCH --partition=cpu_sapphire
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
umask 027
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
cd /home/gpepe/ws/TurboVLA
/home/gpepe/miniconda3/envs/turbovla-libero/bin/python -u scripts/cluster/download_libero_unified.py
