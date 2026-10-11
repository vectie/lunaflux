# MiniMax installed-original matched-layer replay — 2026-10-11

## Fifty independent native layer comparisons

The diagnostic `encode-text-matched-layers` mode now reuses the existing native
observation queues and supplies original `hidden-0` through `hidden-49` before
their corresponding layers. Each output is independent: native layer N does
not become the input of native layer N+1. Kernels, weights, positions, precision
laws and BF16 publication remain unchanged. Ordinary serving has no extra
copies or synchronization. Outputs are diagnostic local-layer comparisons,
not a propagated hidden-state-50 encoder or media-quality result.

The host bound accounts separately for 50 original inputs and 51 snapshots.
Input extent and predecessor-index regressions pass. Physical comparison now
completed for all 50 layers; this diagnostic alone does not fix the remaining
encoder gap.

On `.179`, 140 literal original token IDs and original predecessor tensors
produce 51 native snapshots. Embedding matches all 716,800 BF16 values exactly.
The largest independent layer-wide relative L2 is **0.1854224% at layer 1**.
Layer 7 has the largest independent per-row error: **0.8251653% at row 97**.
Later layer-wide norms are much smaller, but large features dominate those
norms; they are not evidence that every row matches. This separates local
layer discrepancies from the previously recorded accumulated encoder error.

The first-layer operator trace begins differing at Q projection. Input RMSNorm
is bitwise identical. Q/K/V relative L2 is 0.0112170% / 0.0474268% /
0.0006329%; downstream attention, output and final residual are 0.0734488% /
0.1196788% / 0.1854224%. These latter values include propagated differences;
matched-intermediate operator replays are necessary before assigning causes.
The operator reference is the separate installed-ATen reconstruction, not an
unmodified installed-original layer capture.

Both native diagnostics terminate successfully with empty stderr and release
the GPU. The full local suite passes 5,013/5,013. No production arithmetic,
acceptance tolerance or media-quality result is changed by this measurement.

New retained roots:

- `/tmp/lunaflux-minimax-text-encoder-original-matched-long-20261011-v1`
- `/tmp/lunaflux-minimax-matched-long-comparison-20261011-v1`
- `/tmp/lunaflux-minimax-text-encoder-matched-first-operators-20261011-v1`
- `/tmp/lunaflux-minimax-aten-reference-matched-first-long-20261011-v1`
- `/tmp/lunaflux-minimax-matched-first-operator-comparison-20261011-v1`

## Matched first-layer operators and precision attribution

The separate ATen reconstruction now consumes each literal native intermediate
independently, rather than propagating its own preceding operator. On 140 real
original token IDs, input RMSNorm, Q/K normalization, both rotary transforms and
both residual additions match bit-for-bit. This localizes differences away from
those normalization/layout/publication boundaries on this input. The reconstruction
is not an unmodified installed-original individual-operator capture.

| Operator, identical inputs | Relative L2 | Exact BF16 / values |
| --- | --- | --- |
| Q projection | 0.0112170% | 1,144,230 / 1,146,880 |
| K projection | 0.0474268% | 142,948 / 143,360 |
| V projection | 0.000632876% | 143,350 / 143,360 |
| Attention | 0.0735295% | 1,115,895 / 1,146,880 |
| Output projection | 0.0271678% | 713,342 / 716,800 |
| SwiGLU activation | 0.0230035% | 3,571,010 / 3,584,000 |
| Down projection | 0.0412059% | 710,166 / 716,800 |

A projection precision replay on those same intermediate bytes compares native
and original BF16 output with F64 accumulation followed by BF16 publication.
Native versus F64 relative L2 is 0.00116923% for Q, 0.000000996722% for K,
0.000202730% for V, 0.0110001% for output and 0.00852634% for down. Original
BF16 versus F64 errors are respectively 0.0112758%, 0.0474245%, 0.000599594%,
0.0262432% and 0.0411539%. Disabling reduced-precision BF16 reduction produces
the same original BF16 output at this geometry.

Thus a blanket increase in accumulation precision is **not** a justified fix
for faithful original-output agreement: native is often closer to the F64 oracle
already. Different reduction/publication arithmetic remains observable, but
neither this observation nor a small local norm proves propagated encoder or
media quality acceptable. Existing mean/probability/compensated-dot experiments
remain unselected. No numerical tolerance is relaxed.

All reference/precision executions terminate and release their GPU owners.
The offline adapter is separate from the production MoonBit runtime.

Additional retained reports:

- `/tmp/lunaflux-minimax-matched-first-replay-comparison-20261011-v1/operators.tsv`
- `/tmp/lunaflux-minimax-projection-precision-first140-20261011-v1/precision.tsv`

```sh
moon run scripts/run-minimax-text-encoder.mbtx NEW_ROOT SUFFIX COMMITTED_ARM_EXECUTABLE STAGED_WEIGHTS EXACT_ROW_AOT 179 ORIGINAL_IDS_I32 matched ORIGINAL_HIDDEN_DIRECTORY
```

The offline original encoder adapter can now consume an explicit BF16 hidden
state and run exactly one selected installed `MiniMaxH3Qwen3VLTextDecoderLayer`.
It preserves original packed weight loaders, rotary positions, normalization,
attention and MLP arithmetic. It does not recompute a reference prefix before
the selected layer, cast the input to a different dtype, or use LunaFlux as its
own oracle. Only the selected original layer resides on the GPU. Production
execution and arithmetic are unchanged.

## Real Spark validation

On `.179`, original layer 7 receives original hidden-6 from the prior 140-token
installed-original run. Both the retained input and output are **716,800/716,800
BF16 values bitwise identical** to the prior original hidden-6/hidden-7. This
validates the replay's input, layer index, original weights and numerical path.

A second serial replay substitutes literal hidden-6 from the explicitly
**unselected reciprocal-mean experiment**. That experiment is not silently
treated as the production encoder; its original input/output files remain
unchanged. The retained replay input also matches all 716,800 supplied values.

| Comparison at hidden-7 | Global relative L2 | Row 5 | Worst row |
| --- | --- | --- | --- |
| Native experimental layer versus original layer, identical native input | 0.0925733% | 0.0901600% | 0.692476%, row 15 |
| Original versus original, different prior inputs | 0.7619351% | 3.0420645% | 4.3043028%, row 12 |

These norms are not additive. The direct installed-original replay supports
the earlier localization: propagated prior-input differences can dominate the
local discrepancy of this layer. It neither identifies the earliest faulty
operator nor proves the twelve-case encoder or audiovisual quality correct.
Matching the original more closely by a blanket rounding change is not justified
by this result; the prior mean and probability ablations remain unselected.

Both retained reference runs exit zero without OOM and release the GPU, within
8-GiB/no-swap Docker and user-unit limits. Complete CUDA residency/peak was not
measured. Installed-image version warnings are retained; they do not replace
the demonstrated original-input/output identity. No timing claim is made.
An earlier native-input `v1` was dispatched before the prior unit was confirmed
terminal, then retired; it is preserved but excluded. The numerical comparison
uses a fresh `v2` after exact prior-unit retirement and idle-GPU observation.

Roots:

- `/tmp/lunaflux-minimax-upstream-encoder-layer7-original-input-20261011-v1`
- `/tmp/lunaflux-minimax-upstream-encoder-layer7-native-mean-input-20261011-v2`
- `/tmp/lunaflux-minimax-original-layer7-local-error-20261011-v2.txt`
- `/tmp/lunaflux-minimax-original-layer7-input-sensitivity-20261011-v2.txt`

```sh
moon run scripts/run-minimax-upstream-encoder.mbtx prepare NEW_ROOT LITERAL_IDS_I32 1 --replay-layer 7 LITERAL_HIDDEN_6_BF16
moon run scripts/run-minimax-upstream-encoder.mbtx run NEW_ROOT
# Inspect only after this exact user unit/container is terminal.
moon run scripts/run-minimax-upstream-encoder.mbtx inspect NEW_ROOT
```

Three MoonBit runner regressions pass. The adapter compiles with
`-Wall -Wextra -Werror` in the installed ARM image and passes the real replay
identity checks above. Native warning-denied checking passes with the existing
toolchain migration exclusions; no exclusions or production dependencies were
added.
