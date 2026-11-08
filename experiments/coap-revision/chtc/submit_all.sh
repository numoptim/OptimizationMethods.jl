#!/bin/bash
# Submits every chtc/jobs_*.sub (each has at most 10,000 jobs). Run from anywhere:
#   bash experiments/coap-revision/chtc/submit_all.sh [--dry-run]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # experiments/coap-revision
mkdir -p chtc/logs results/raw
for f in chtc/julia-1.13.2-linux-x86_64.tar.gz chtc/code.tar.gz chtc/depot.tar.gz; do
  [ -f "$f" ] || { echo "ERROR: missing $f (see CHTC_RUNBOOK.md steps 1-2)"; exit 1; }
done
for sub in chtc/jobs_*.sub; do
  n=$(grep -c . "${sub%.sub}.txt")
  echo "$sub: $n jobs"
  if [ "${1:-}" = "--dry-run" ]; then
    condor_submit -dry-run /dev/null "$sub" > /dev/null && echo "  dry run ok"
  else
    condor_submit "$sub"
  fi
done
