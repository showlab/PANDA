#!/usr/bin/env bash
set -euo pipefail

: "${VIDEO_ROOT:?Set VIDEO_ROOT}"
: "${ANNOTATION_FILE:?Set ANNOTATION_FILE}"
: "${DATASET:?Set DATASET to ucf, xd, ub, or csad}"
CHUNK_TOTAL="${CHUNK_TOTAL:-1}"
if [[ ! "$CHUNK_TOTAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "CHUNK_TOTAL must be a positive integer" >&2
  exit 2
fi
VLM_PREFIX="${VLM_PREFIX:-qwen2_5_7b_v}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"
RUN_ID="${RUN_ID:-run-$(date -u +%Y%m%dT%H%M%SZ)-$$}"

case "$RUN_ID" in
  ""|*[!A-Za-z0-9._-]*)
    echo "RUN_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
    ;;
esac

for arg in "$@"; do
  case "$arg" in
    --video-root|--video-root=*|--annotation-file|--annotation-file=*|--dataset|--dataset=*|\
    --output-dir|--output-dir=*|--chunk-idx|--chunk-idx=*|--chunk-total|--chunk-total=*|\
    --vlm-api|--vlm-api=*|--run-id|--run-id=*)
      echo "Cannot override reserved worker argument: $arg" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
RUN_OUTPUT_DIR="$OUTPUT_DIR/$RUN_ID"
if ! mkdir "$RUN_OUTPUT_DIR"; then
  echo "Run output directory already exists: $RUN_OUTPUT_DIR" >&2
  exit 2
fi

pids=()
for ((idx=0; idx<CHUNK_TOTAL; idx++)); do
  python -m panda.evaluation.runner "$@" \
    --video-root "$VIDEO_ROOT" \
    --annotation-file "$ANNOTATION_FILE" \
    --dataset "$DATASET" \
    --output-dir "$RUN_OUTPUT_DIR" \
    --run-id "$RUN_ID" \
    --chunk-idx "$idx" \
    --chunk-total "$CHUNK_TOTAL" \
    --vlm-api "${VLM_PREFIX}${idx}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
