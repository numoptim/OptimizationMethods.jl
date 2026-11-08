#!/bin/bash
# Submits every chtc/jobs_*.sub (each has at most 10,000 jobs), skipping jobs whose output
# directory results/raw/job_<id> already exists (e.g. the CHTC pilot jobs). Run from anywhere:
#   bash experiments/coap-revision/chtc/submit_all.sh [--dry-run]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # experiments/coap-revision
mkdir -p chtc/logs results/raw
for f in chtc/julia-1.13.3-linux-x86_64.tar.gz chtc/code.tar.gz chtc/depot.tar.gz; do
  [ -f "$f" ] || { echo "ERROR: missing $f (see CHTC_RUNBOOK.md steps 1-2)"; exit 1; }
done
for sub in chtc/jobs_[0-9]*.sub; do
  base="${sub%.sub}"
  todo="${base}_todo"
  while read -r id; do
    [ -z "$id" ] && continue
    [ -d "results/raw/job_$id" ] || echo "$id"
  done < "$base.txt" > "$todo.txt"
  n=$(grep -c . "$todo.txt" || true)
  echo "$sub: $(grep -c . "$base.txt") jobs listed, $n to submit (the rest already have output)"
  [ "$n" -eq 0 ] && continue
  sed -e "s#^queue job from .*#queue job from $todo.txt#" "$sub" > "$todo.sub"
  if [ "${1:-}" = "--dry-run" ]; then
    echo "  dry run: would run condor_submit $todo.sub"
  else
    condor_submit "$todo.sub"
  fi
done
