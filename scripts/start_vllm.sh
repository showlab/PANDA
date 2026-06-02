#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local Qwen2.5-VL model directory}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

args=(
  --model "$MODEL_PATH"
  --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --limit-mm-per-prompt 'image=300,video=2'
  --trust-remote-code
)
if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  args+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
fi

exec python -m vllm.entrypoints.openai.api_server "${args[@]}"
