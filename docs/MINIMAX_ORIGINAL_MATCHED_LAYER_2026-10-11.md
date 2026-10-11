# MiniMax installed-original matched-layer replay — 2026-10-11

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
