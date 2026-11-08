#!/bin/bash
# Run ONCE on the CHTC submit node, from the repository root or anywhere inside it:
#   bash experiments/coap-revision/chtc/make_tarballs.sh
# Creates, in experiments/coap-revision/chtc/:
#   julia-1.13.2-linux-x86_64.tar.gz   (portable Julia, downloaded if missing)
#   code.tar.gz                        (tracked files of the current commit, prefix repo/)
set -euo pipefail
JULIA_VERSION=1.13.2
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel)"
cd "$here"
if [ ! -f "julia-${JULIA_VERSION}-linux-x86_64.tar.gz" ]; then
  wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-${JULIA_VERSION}-linux-x86_64.tar.gz"
fi
git -C "$root" archive --format=tar.gz --prefix=repo/ HEAD -o "$here/code.tar.gz"
# the experiments Manifest.toml is committed with `git add -f` (see STATUS.md); check it is in the archive
tar -tzf code.tar.gz | grep -q 'repo/experiments/coap-revision/Manifest.toml' || { echo "ERROR: experiments Manifest.toml is not committed"; exit 1; }
ls -lh julia-${JULIA_VERSION}-linux-x86_64.tar.gz code.tar.gz
