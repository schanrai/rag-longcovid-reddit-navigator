#!/usr/bin/env bash
# Run Phase 1c embedding in the background; survives closing the terminal window.
# Logs to reports/embed_eval.log — use: tail -f reports/embed_eval.log
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${ROOT}/reports/embed_eval.log"
mkdir -p "${ROOT}/reports"
if [[ ! -d "${ROOT}/.venv" ]]; then
  echo "Missing .venv — create it and pip install -r requirements.txt first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
{
  echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") starting embed_eval ====="
  echo "args: $*"
} >>"$LOG"
nohup python3 "${ROOT}/src/embed_eval.py" "$@" >>"$LOG" 2>&1 &
echo "Started embed_eval in background (PID $!)"
echo "Log: $LOG"
echo "Watch: tail -f $LOG"
