#!/bin/bash
# Run INSIDE the interactive build job (after `condor_submit -i chtc/build.sub`):
#   bash build_env.sh
# Builds depot.tar.gz (pre-instantiated, precompiled Julia 1.13.3 depot) in $PWD and
# prints its size. Then `exit` the interactive job; the tarball is transferred back to
# the submit node ONLY if you copy it out: use the last line of this script's output.
set -euo pipefail
JULIA_VERSION=1.13.3
tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz
tar -xzf code.tar.gz
export PATH="$PWD/julia-${JULIA_VERSION}/bin:$PATH"
export JULIA_DEPOT_PATH="$PWD/depot"
mkdir -p "$JULIA_DEPOT_PATH"
cd repo/experiments/coap-revision
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
# smoke test: the package and the experiment code load
julia --project=. -e 'include("src/runner.jl"); println("runner loads, Julia ", VERSION)'
cd ../../..
tar -czf depot.tar.gz depot
ls -lh depot.tar.gz
echo "NEXT: type exit; HTCondor returns depot.tar.gz to your submit directory (see CHTC_RUNBOOK.md)."
echo "If it is larger than 100MB, ask chtc@cs.wisc.edu for SQUID/staging access (CHTC Julia guide)."
