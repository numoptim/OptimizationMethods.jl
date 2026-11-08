#!/bin/bash
# Prints the max and mean job runtime (hours), peak memory and disk of the jobs of one
# cluster, checked against the 24 h target:
#   bash experiments/coap-revision/chtc/job_runtimes.sh CLUSTER_ID
# (CLUSTER_ID is printed by condor_submit: "N job(s) submitted to cluster CLUSTER_ID".)
# `-` instead of CLUSTER_ID reads lines "ClusterId ProcId WallSeconds MemoryMB DiskKB JobStatus" from stdin.
set -euo pipefail
ARG="${1:?usage: job_runtimes.sh CLUSTER_ID}"
if [ "$ARG" = "-" ]; then
  src() { cat; }
else
  src() { condor_history -constraint "ClusterId == $ARG" -af ClusterId ProcId RemoteWallClockTime MemoryUsage DiskUsage JobStatus; }
fi
src | awk '
  $6 == 4 { n++; h = $3 / 3600; s += h; if (h > mx) mx = h
            if ($4 + 0 > mem) mem = $4 + 0; if ($5 + 0 > dsk) dsk = $5 + 0 }
  $6 != 4 { bad++ }
  END {
    if (n == 0) { print "no completed jobs found"; exit 1 }
    printf "completed jobs: %d   not completed (held/removed/other): %d\n", n, bad
    printf "runtime hours: max %.2f   mean %.2f   (target <= 24)\n", mx, s / n
    printf "peak memory: %.0f MB   peak disk: %.2f GB\n", mem, dsk / 1048576
    if (mx <= 24) print "RUNTIME OK (<= 24 h)"; else { print "RUNTIME TOO LONG (> 24 h): lower G_max/F_max in src/jobs.jl or shrink TARGET_WORST_H and regenerate"; exit 2 }
  }'
