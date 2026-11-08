#!/bin/bash
# Runs src/collect.jl on the submit node using the portable Julia + depot (set up once
# in $HOME/coap-julia). All arguments are passed to collect.jl, e.g.
#   bash experiments/coap-revision/chtc/collect.sh
#   bash experiments/coap-revision/chtc/collect.sh --pilot
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."          # experiments/coap-revision
J="$HOME/coap-julia"
if [ ! -x "$J/julia-1.13.2/bin/julia" ]; then
  mkdir -p "$J"
  tar -xzf chtc/julia-1.13.2-linux-x86_64.tar.gz -C "$J"
  tar -xzf chtc/depot.tar.gz -C "$J"
fi
export PATH="$J/julia-1.13.2/bin:$PATH"
export JULIA_DEPOT_PATH="$J/depot"
julia --project=. src/collect.jl "$@"
