#!/bin/bash
# Reruns the jobs listed (one job id per line) in a file written by collect.jl:
#   bash chtc/resubmit.sh results/merged/resubmit_jobs.txt [--dry-run]
# Existing output of those jobs is moved to results/raw_invalid/ (not deleted), then the jobs are
# submitted with the settings of chtc/jobs_1.sub. Run it only when `condor_q` shows no jobs of yours.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # experiments/coap-revision
LIST="${1:?usage: resubmit.sh results/merged/resubmit_jobs.txt [--dry-run]}"
[ -f "$LIST" ] || { echo "ERROR: $LIST not found"; exit 1; }
for f in chtc/julia-1.13.3-linux-x86_64.tar.gz chtc/code.tar.gz chtc/depot.tar.gz; do
  [ -f "$f" ] || { echo "ERROR: missing $f (see CHTC_RUNBOOK.md steps 1-2)"; exit 1; }
done
mkdir -p chtc/logs results/raw results/raw_invalid
stamp=$(date -u +%Y%m%dT%H%M%SZ)
TODO=chtc/resubmit_todo
grep -E '^[0-9]+$' "$LIST" | sort -un > "$TODO.txt" || true
n=$(grep -c . "$TODO.txt" || true)
if [ "$n" -eq 0 ]; then echo "$LIST lists no jobs; nothing to do"; exit 0; fi
echo "$n job(s) to rerun: $(tr '\n' ' ' < "$TODO.txt")"
if [ "${2:-}" = "--dry-run" ]; then
  echo "dry run: would move their old output to results/raw_invalid/ and run condor_submit $TODO.sub"
  exit 0
fi
while read -r id; do
  if [ -d "results/raw/job_$id" ]; then
    mv "results/raw/job_$id" "results/raw_invalid/job_${id}_$stamp"
  fi
done < "$TODO.txt"
sed -e "s#^queue job from .*#queue job from $TODO.txt#" chtc/jobs_1.sub > "$TODO.sub"
condor_submit "$TODO.sub"
