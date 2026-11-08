#!/bin/bash
# Executable of every experiment job. Argument: JOB_ID.
# Inputs (transferred): julia tarball, code.tar.gz, depot.tar.gz.
set -euo pipefail
JOB_ID="$1"
JULIA_VERSION=1.13.3
tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz
tar -xzf code.tar.gz
tar -xzf depot.tar.gz
export PATH="$PWD/julia-${JULIA_VERSION}/bin:$PATH"
export JULIA_DEPOT_PATH="$PWD/depot"
export JULIA_NUM_THREADS=1
cd repo/experiments/coap-revision
mkdir -p "$_CONDOR_SCRATCH_DIR/out"
julia --project=. src/run_job.jl "$JOB_ID" --out "$_CONDOR_SCRATCH_DIR/out"
# hand the finished job directory back (HTCondor transfers job_<id>)
mv "$_CONDOR_SCRATCH_DIR/out/job_${JOB_ID}" "$_CONDOR_SCRATCH_DIR/job_${JOB_ID}"
