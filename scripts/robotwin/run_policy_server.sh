#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/starvla_runtime:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/robotwin/run_policy_server.sh <ckpt_path> [gpu_id] [port]" >&2
    exit 1
fi

your_ckpt="$1"
gpu_id="${2:-${ROBOTWIN_SERVER_GPU:-0}}"
port="${3:-${ROBOTWIN_SERVER_PORT:-5694}}"
star_vla_python="${STARVLA_PYTHON:-${star_vla_python:-python}}"

use_bf16_flag=()
if [[ "${ROBOTWIN_USE_BF16:-1}" != "0" ]]; then
    use_bf16_flag+=(--use_bf16)
fi

# Released checkpoints carry their original Hugging Face model IDs in
# config.yaml. When local paths are provided, override those IDs at runtime so
# evaluation remains offline and uses the exact snapshots staged on the cluster.
config_override_flags=(
    --cfg-option "framework.initialization.load_pretrained=false"
)
if [[ -n "${BERT_MODEL_PATH:-}" ]]; then
    if [[ ! -d "${BERT_MODEL_PATH}" ]]; then
        echo "[ERROR] BERT_MODEL_PATH does not exist: ${BERT_MODEL_PATH}" >&2
        exit 1
    fi
    config_override_flags+=(
        --cfg-option "framework.text.bert_path=${BERT_MODEL_PATH}"
        --cfg-option "framework.text.local_files_only=true"
    )
fi
if [[ -n "${DINOV3_MODEL_PATH:-}" ]]; then
    if [[ ! -d "${DINOV3_MODEL_PATH}" ]]; then
        echo "[ERROR] DINOV3_MODEL_PATH does not exist: ${DINOV3_MODEL_PATH}" >&2
        exit 1
    fi
    config_override_flags+=(
        --cfg-option "framework.vision.model_path=${DINOV3_MODEL_PATH}"
        --cfg-option "framework.vision.local_files_only=true"
    )
fi

echo "[INFO] Starting RoboTwin policy server"
echo "[INFO] checkpoint: ${your_ckpt}"
echo "[INFO] gpu: ${gpu_id}"
echo "[INFO] port: ${port}"
[[ -z "${BERT_MODEL_PATH:-}" ]] || echo "[INFO] local BERT: ${BERT_MODEL_PATH}"
[[ -z "${DINOV3_MODEL_PATH:-}" ]] || echo "[INFO] local DINOv3: ${DINOV3_MODEL_PATH}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${star_vla_python}" "${REPO_ROOT}/third_party/starvla_runtime/deployment/model_server/server_policy.py" \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${config_override_flags[@]}" \
    "${use_bf16_flag[@]}"
