#!/usr/bin/env bash
# Run Clover-35B-A3B-Preview with llama-diffusion-cli.
#
#   MODEL=/path/to/Clover-35B-A3B-Preview-Q4_K_M.gguf ./run.sh prompt.txt
#
# Knobs (env):
#   MODEL   path to the GGUF                          (default ./Clover-35B-A3B-Preview-Q4_K_M.gguf)
#   BIN     path to llama-diffusion-cli               (default ./build/bin/llama-diffusion-cli)
#   UB      generation length in positions            (default 512)
#   TEMP    sampling temperature, 0 = greedy          (default 0.0)
#   CONF    confidence threshold for resolving a slot (default 0.95, driver default)
#   NGL     layers offloaded to the GPU               (default 99)
#
# Note: generation length is -ub, not -n. The driver ignores -n.
set -euo pipefail

PROMPT=${1:?usage: ./run.sh prompt.txt}

MODEL=${MODEL:-./Clover-35B-A3B-Preview-Q4_K_M.gguf}
BIN=${BIN:-./build/bin/llama-diffusion-cli}
UB=${UB:-512}
TEMP=${TEMP:-0.0}
NGL=${NGL:-99}

if [ ! -f "$MODEL" ]; then
    echo "error: model not found at $MODEL (set MODEL=...)" >&2
    exit 1
fi

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not built (see README: cmake --build build --target llama-diffusion-cli)" >&2
    exit 1
fi

if [ -n "${CONF:-}" ]; then
    export LLAMA_DIFFUSION_CONF_THRESHOLD="$CONF"
fi

exec "$BIN" \
    -m "$MODEL" \
    -ngl "$NGL" -c 4096 \
    -f "$PROMPT" \
    -ub "$UB" \
    --diffusion-block-length 32 \
    --diffusion-algorithm 4 \
    --diffusion-steps 128 \
    --temp "$TEMP"
