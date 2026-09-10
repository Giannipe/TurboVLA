#!/bin/bash
#SBATCH --job-name=turbovla-assets
#SBATCH --output=/home/gpepe/ws/logs/turbovla/assets/%x.out
#SBATCH --error=/home/gpepe/ws/logs/turbovla/assets/%x.err
#SBATCH --partition=cpu_sapphire
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
# Usage: sbatch [SLURM options] scripts/cluster/assets.sh --benchmark libero [application flags]
# Preview without allocation: bash scripts/cluster/assets.sh --benchmark libero --dry-run [...]
set -euo pipefail
umask 027
export PYTHONDONTWRITEBYTECODE=1
repo_root="${TURBOVLA_REPO:-/home/gpepe/ws/TurboVLA}"
# sbatch executes a spool copy: do not derive repo_root from BASH_SOURCE.
benchmark=libero
preview=false
arguments=("$@")
for ((index=0; index<${#arguments[@]}; index++)); do
  case "${arguments[index]}" in
    --help|-h|--dry-run) preview=true ;;
    --benchmark)
      ((index + 1 < ${#arguments[@]})) || { echo "--benchmark requires a value" >&2; exit 2; }
      benchmark="${arguments[index+1]}"
      ;;
    --benchmark=*) benchmark="${arguments[index]#*=}" ;;
    --submit)
      echo "Already a batch script: use sbatch $0, without --submit." >&2
      exit 2
      ;;
    --job-name|--job-name=*|-J|-J?*)
      echo "Job names go BEFORE the script: sbatch --job-name=NAME $0 ..." >&2
      exit 2
      ;;
    --gpus|--gpus=*|--gres|--gres=*|--partition|--partition=*|--mem|--mem=*|--cpus|--cpus=*|--cpus-per-task|--cpus-per-task=*|--time|--time=*|--nodes|--nodes=*|--ntasks|--ntasks=*)
      echo "Slurm resources go BEFORE the script: sbatch --gres=gpu:N --cpus-per-task=N --mem=... --time=... $0 ..." >&2
      exit 2
      ;;
  esac
done
if [[ -z "${SLURM_JOB_ID:-}" && "$preview" != true ]]; then
  echo "Use sbatch scripts/cluster/assets.sh ...; bash is only for --help/--dry-run." >&2
  exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-1}" != 1 || "${SLURM_NTASKS:-1}" != 1 ]]; then
  echo "This asset launcher requires --nodes=1 --ntasks=1." >&2
  exit 2
fi
case "$benchmark" in
  libero|all) environment="${TURBOVLA_LIBERO_ENV:-turbovla-libero}" ;;
  robotwin) environment="${TURBOVLA_ROBOTWIN_ENV:-turbovla-robotwin}" ;;
  *) echo "Unknown benchmark: $benchmark" >&2; exit 2 ;;
esac
python_bin="${TURBOVLA_PYTHON:-/home/gpepe/miniconda3/envs/$environment/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing interpreter $python_bin; run envs.sh first." >&2; exit 1; }
[[ -f "$repo_root/scripts/cluster/_internal/assets.py" ]] || { echo "Invalid TURBOVLA_REPO: $repo_root" >&2; exit 1; }
cd "$repo_root"
exec "$python_bin" -u scripts/cluster/_internal/assets.py "${arguments[@]}"
