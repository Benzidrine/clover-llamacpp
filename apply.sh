#!/usr/bin/env bash
# Apply the Clover patch series to a llama.cpp checkout.
#
#   ./apply.sh /path/to/llama.cpp            # expects the PR branch checked out
#   ./apply.sh /path/to/llama.cpp 4838ffb    # or an explicit base commit
#
# The patches are cut against PR #31306 at 4838ffb; use `git am -3` if you are on a
# slightly different revision.
set -euo pipefail

LLAMA_DIR=${1:?usage: ./apply.sh /path/to/llama.cpp [base-commit]}
BASE=${2:-4838ffb}

HERE=$(cd "$(dirname "$0")" && pwd)

cd "$LLAMA_DIR"

if ! git rev-parse --verify --quiet "$BASE" >/dev/null; then
    echo "error: base commit $BASE not found in $LLAMA_DIR" >&2
    echo "       fetch the PR branch first:" >&2
    echo "       git fetch origin pull/31306/head:clover-31306 && git checkout clover-31306" >&2
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "error: $LLAMA_DIR has uncommitted changes; commit or stash them first" >&2
    exit 1
fi

git am "$HERE"/patches/*.patch

echo
echo "applied. build with:"
echo "  cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON"
echo "  cmake --build build -j --target llama-diffusion-cli"
