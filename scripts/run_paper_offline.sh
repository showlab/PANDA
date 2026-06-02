#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  echo "The paper profile accepts no trailing arguments: $*" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="${PANDA_PAPER_MODE:-offline}"
case "$MODE" in
  offline) PERCEPTION_COUNT=300; PERCEPTION_SAMPLING=paper_uniform; SMOOTHING=mean ;;
  online) PERCEPTION_COUNT=10; PERCEPTION_SAMPLING=legacy_initial; SMOOTHING=none ;;
  *) echo "PANDA_PAPER_MODE must be offline or online" >&2; exit 2 ;;
esac
PROFILE_ARGS=(--profile "paper-$MODE")
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.0-flash}"
if [[ "$GEMINI_MODEL" != "gemini-2.0-flash" ]]; then
  if [[ "${PANDA_ALLOW_MODEL_SUBSTITUTION:-0}" != "1" ]]; then
    echo "Changed Gemini model requires PANDA_ALLOW_MODEL_SUBSTITUTION=1" >&2
    exit 2
  fi
  PROFILE_ARGS+=(--allow-model-substitution)
fi

: "${VIDEO_ROOT:?Set VIDEO_ROOT}"
: "${ANNOTATION_FILE:?Set ANNOTATION_FILE}"
: "${DATASET:?Set DATASET to ucf, xd, ub, or csad}"
: "${RUN_ID:?Set an explicit RUN_ID}"
: "${PANDA_RULES_DIR:?Set PANDA_RULES_DIR to the local rules/index directory}"
: "${EMBEDDING_MODEL_PATH:?Set EMBEDDING_MODEL_PATH to the local embedding model directory}"
: "${VLM_MODEL_PATH:?Set VLM_MODEL_PATH to the local VLM model directory}"
: "${VLM_MODEL_MANIFEST:?Set VLM_MODEL_MANIFEST to its local content manifest}"
: "${EMBEDDING_MODEL_MANIFEST:?Set EMBEDDING_MODEL_MANIFEST to its local content manifest}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for Gemini planning and reflection}"
: "${TAVILY_API_KEY:?Set TAVILY_API_KEY for the web-search reflection tool}"
: "${VLM_MODEL_SHA256:?Set VLM_MODEL_SHA256 to the verified model content-manifest digest}"
: "${EMBEDDING_MODEL_SHA256:?Set EMBEDDING_MODEL_SHA256 to the verified embedding content-manifest digest}"

if [[ ! -d "$PANDA_RULES_DIR" ]]; then
  echo "PANDA_RULES_DIR must be an existing local directory: $PANDA_RULES_DIR" >&2
  exit 2
fi
if [[ ! -d "$EMBEDDING_MODEL_PATH" ]]; then
  echo "EMBEDDING_MODEL_PATH must be an existing local directory: $EMBEDDING_MODEL_PATH" >&2
  exit 2
fi

case "$DATASET" in
  ucf)
    RULES_JSON="ucf_anomaly_rules_20.json"
    RULES_INDEX="ucf_anomaly_rules_faiss_20.index"
    ;;
  xd)
    RULES_JSON="xd_anomaly_rules_20.json"
    RULES_INDEX="xd_anomaly_rules_faiss_20.index"
    ;;
  ub)
    RULES_JSON="ub_train_anomaly_rules_20.json"
    RULES_INDEX="ub_train_anomaly_rules_faiss_20.index"
    ;;
  csad)
    RULES_JSON="CSAD_anomaly_rules_20.json"
    RULES_INDEX="CSAD_anomaly_rules_faiss_20.index"
    ;;
  *)
    echo "DATASET must be one of: ucf, xd, ub, csad" >&2
    exit 2
    ;;
esac

for asset in "$RULES_JSON" "$RULES_INDEX"; do
  if [[ ! -f "$PANDA_RULES_DIR/$asset" ]]; then
    echo "Required dataset-specific RAG asset does not exist: $PANDA_RULES_DIR/$asset" >&2
    exit 2
  fi
done

VERIFY_PYTHON="${PANDA_VERIFY_PYTHON:-python3}"
"$VERIFY_PYTHON" "$SCRIPT_DIR/verify_content_manifest.py" \
  --root "$VLM_MODEL_PATH" \
  --manifest "$VLM_MODEL_MANIFEST" \
  --expected-sha256 "$VLM_MODEL_SHA256" >/dev/null
"$VERIFY_PYTHON" "$SCRIPT_DIR/verify_content_manifest.py" \
  --root "$EMBEDDING_MODEL_PATH" \
  --manifest "$EMBEDDING_MODEL_MANIFEST" \
  --expected-sha256 "$EMBEDDING_MODEL_SHA256" >/dev/null

# All model-backed tools are selectable, so validate every tool asset before
# creating a run or starting workers. No model dependency is imported here.
"$VERIFY_PYTHON" -m panda.tool_assets >/dev/null

PAPER_OUTPUT_DIR="${PAPER_OUTPUT_DIR:-./paper_offline_results}"
RUN_OUTPUT_DIR="$PAPER_OUTPUT_DIR/$RUN_ID"

OUTPUT_DIR="$PAPER_OUTPUT_DIR" "$SCRIPT_DIR/run_eval_chunks.sh" "${PROFILE_ARGS[@]}" \
  --fps 1 \
  --window 5 \
  --perception-frames-num "$PERCEPTION_COUNT" \
  --perception-sampling "$PERCEPTION_SAMPLING" \
  --rag-backend faiss \
  --num-rules 5 \
  --vlm-temperature 0.2 \
  --vlm-top-p 0.5 \
  --planning-provider gemini \
  --reflection-provider gemini

panda-merge \
  --glob "$RUN_OUTPUT_DIR/results_gpu*.json" \
  --output "$RUN_OUTPUT_DIR/merged.json" \
  --smoothing "$SMOOTHING" \
  --smoothing-window 10 \
  --smoothing-boundary reflect
