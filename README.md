# Clover-35B-A3B in llama.cpp

Patches that make [Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
— a 35B-A3B **block-diffusion** MoE — actually run and produce coherent text in
[llama.cpp](https://github.com/ggml-org/llama.cpp).

Upstream [PR #31306](https://github.com/ggml-org/llama.cpp/pull/31306) adds the `clover_moe`
architecture, its GGUF metadata and its graph. As of base commit `4838ffb` it loads the
weights and then cannot run inference on them. These four patches take it from there to
working output, about **100 tok/s** for a 480-token reply on one RTX 3090 with Q4_K_M.

This is a working port, not an upstream-quality contribution — see [Limitations](#limitations).

## Why it does not work out of the box

| # | Fault | Symptom |
|---|---|---|
| 1 | `llm_arch_is_diffusion()` never whitelists `clover_moe` | loads all 21 GB, then `unsupported model for diffusion` |
| 2 | the block schedule re-feeds the whole sequence every step and never commits a block | fluent but unrelated text (a fabricated `<|im_start|>user` turn) |
| 3 | `shift_logits` falls back to `true` | every masked position samples its left neighbour's distribution |
| 4 | the expert top-k runs as a custom CPU sort, twice per layer | 82 graph splits, ~80 host round-trips per pass |

Only (1) is a one-line fix. (2) is the real work: the driver has to follow the reference
sampler's structure, not just clear the cache between steps.

## Patches

| Patch | File | Effect |
|---|---|---|
| `0001` | `src/llama-arch.cpp` | registers `clover_moe` as a diffusion architecture |
| `0002` | `src/models/clover-moe.cpp` | moves the expert sort to the backend, 82 → 2 graph splits |
| `0003` | `examples/diffusion/diffusion.cpp` | commit prompt → preview block → commit block, plus the transfer policy |
| `0004` | `examples/diffusion/diffusion-cli.cpp` | `shift_logits` default and the second sequence slot |

What `0003` changes, in the order the denoising loop runs:

1. **Commit the whole prompt** through the causal path (`llama_set_causal_attn(ctx, true)`)
   into sequence 0.
2. **Preview a block** of exactly `block_length` mask tokens at the positions continuing
   after the prompt, with causal attention off. The preview runs on a *scratch* sequence
   copied from sequence 0, because the Gated-DeltaNet recurrent state supports a full reset
   only, and the cache rejects re-running positions it already holds.
3. **Commit the resolved block** by replaying it through the causal path on the scratch
   sequence and copying it back, so the next block starts from a correct state.
4. **Resolve every pending slot** whose confidence clears a threshold, falling back to the
   single most confident slot — rather than transferring a fixed count per step. This is
   what lets a 32-slot block converge in ~10 passes instead of 32.
5. Never sample the mask token; report a step budget consistent with the block schedule.

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# the PR branch the patches are cut against
git fetch origin pull/31306/head:clover-31306
git checkout clover-31306          # or: git checkout 4838ffb

git am /path/to/clover-llamacpp/patches/*.patch

cmake -B build -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j --target llama-diffusion-cli
```

`-DLLAMA_BUILD_EXAMPLES=ON` matters: `llama-diffusion-cli` is not built by default.

## Get a model

```bash
# Q4_K_M, 21.17 GB
huggingface-cli download maternion/Clover-35B-A3B-Preview-GGUF \
  Clover-35B-A3B-Preview-Q4_K_M.gguf --local-dir .
```

## Run

```bash
./build/bin/llama-diffusion-cli \
  -m Clover-35B-A3B-Preview-Q4_K_M.gguf \
  -ngl 99 -c 4096 \
  -f prompt.txt \
  -ub 512 \
  --diffusion-block-length 32 \
  --diffusion-algorithm 4 \
  --diffusion-steps 128 \
  --temp 0.0
```

or `./run.sh prompt.txt`.

**`-ub` is the generation length**, not `-n`; `-n` is ignored by this driver. Positions
beyond `n_input + blocks*32` stay masked, which is the `masks left unresolved` count.

Environment knobs:

- `LLAMA_DIFFUSION_CONF_THRESHOLD` (default `0.95`) — the confidence above which a slot is
  resolved in the current pass. Lower it to resolve more per pass (faster, lower quality),
  raise it for the opposite. It exists as an env var because `--diffusion-eps` is mutually
  exclusive with `--diffusion-block-length` in this CLI.

## Measured

`prompt.txt` (`The capital of France is`, 13 tokens after the chat template), greedy,
RTX 3090, `-ngl 99`, timing excludes model load:

| `-ub` | blocks | generated | wall | tok/s |
|---|---:|---:|---:|---:|
| 128 | 3 | 96 | 1.49 s | 64.4 |
| 256 | 7 | 224 | 2.52 s | 89.0 |
| 512 | 15 | 480 | 4.72 s | 101.6 |

Throughput rises with length because prompt prefill and one commit pass per block amortize.
Before the four patches the same 128-position run took 5.66 s and produced garbage.

Output:

```
The user is asking for the capital of France. This is a straightforward
factual question. The capital of France is Paris. I should provide a
clear, direct answer.

The capital of France is **Paris**.
```

## Testing a single pass

`tools/clover_probe.cpp` runs one forward pass and prints the top-k at each position, which
is far quicker to reason about than a 128-token denoising loop. It can commit a prompt
causally and then preview a block after it (the reference arrangement):

```bash
g++ -O2 -std=c++17 tools/clover_probe.cpp \
  -I llama.cpp/include -I llama.cpp/ggml/include \
  -L llama.cpp/build/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,$PWD/llama.cpp/build/bin -o clover_probe

PROBE_PREFILL=1 ./clover_probe model.gguf "The capital of France is" 32 6
# gen  0:  [Paris 21.19]  [the 21.03]  [a 20.31] ...
```

`PROBE_DUMP=1` prints per-layer hidden-state statistics, `PROBE_CAUSAL=1` uses the ordinary
causal path as a control.

## Limitations

- Cut against PR #31306 at `4838ffb`. Upstream may fix some of this differently; apply to a
  moving branch and expect conflicts.
- Tested with `Q4_K_M` only. Other quants are untouched by these patches but unmeasured.
- No stopping on `<|im_end|>`: the loop always fills the `-ub` budget, so trim the text
  yourself.
- Quality has only been spot-checked on short prompts, not evaluated systematically.
- `--diffusion-algorithm` values other than `4` (confidence-based) were not exercised with
  the new transfer policy.

## One thing not to do

The reference model threads a `valid` mask through the GDN layer, and it is tempting to add
a per-token validity mask to the graph so masked slots are excluded from the convolution and
the block softmax. **Do not.** In the reference sampler, `valid` is `attention_mask`, and the
block loop passes `ones_like(block)` — mask tokens are attended to normally, and unresolved
slots are tracked in a separate `pending` tensor. Implementing the gating collapsed a block
that predicted `Paris` at 21.19 into a constant `7` token at ~9.

## Attribution

- Model: [Diffuspace/Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
- GGUF: [maternion/Clover-35B-A3B-Preview-GGUF](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF)
- Upstream work: llama.cpp PR #31306
- Patches here are MIT, matching llama.cpp.
