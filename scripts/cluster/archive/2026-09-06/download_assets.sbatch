#!/usr/bin/env bash
#SBATCH --job-name=turbovla-assets
#SBATCH --output=/home/gpepe/ws/logs/turbovla-assets-%j.out
#SBATCH --error=/home/gpepe/ws/logs/turbovla-assets-%j.err
#SBATCH --partition=cpu_sapphire
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

repo_root="${TURBOVLA_REPO:-/home/gpepe/ws/TurboVLA}"
scope="${1:-all}"

export SCRATCH_FLASH="${TURBOVLA_SCRATCH_FLASH:-/mnt/beegfs/gpepe}"

echo "[job] host=$(hostname)"
echo "[job] started=$(date --iso-8601=seconds)"
echo "[job] repository=$repo_root"
echo "[job] scope=$scope"
echo "[job] scratch=$SCRATCH_FLASH"

cd "$repo_root"
bash scripts/cluster/download_assets.sh "$scope" \
  --max-workers "${HF_MAX_WORKERS:-8}"

echo "[job] finished=$(date --iso-8601=seconds)"
