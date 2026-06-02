#!/usr/bin/env bash
# Published initial-ten-frame protocol; no offline smoothing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PANDA_PAPER_MODE=online
export PAPER_OUTPUT_DIR="${PAPER_OUTPUT_DIR:-./paper_online_results}"
exec bash "$SCRIPT_DIR/run_paper_offline.sh" "$@"
