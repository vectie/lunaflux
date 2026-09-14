# Historical interval specialization

This continues the exact-arithmetic work in
[ATTENTION_IF_CONVERSION_2026-09-14.md](ATTENTION_IF_CONVERSION_2026-09-14.md).
It does not establish framework or end-to-end parity.

## Implementation

The split K/V transfer lowering partially evaluates the immutable read-view
union for complete historical tiles. When `kb + key_tile <= first_position`,
K addresses come exclusively from the paged view. Mixed boundary tiles retain
the original selection, and V consumes the previously computed address product.
No model identity or Qwen shape check is involved. The helper is shared by
the specialized and general branches, rather than maintaining two kernels.

The softmax rescaling alpha uses the existing bit-masked exact `expf` selection,
preserving positive zero for an infinite previous maximum. Approximate `__expf`
was measured separately but is **not** enabled: it changed BF16 outputs by up to
0.000488281 and did not eliminate the remaining performance gap.

## GPU measurements

RTX 5060 Ti, CUDA 13.1.115, generated c322, Q64/K64/D128, 128 threads.
Six independent processes with five alternating old/new trials per workload.
The 16 workload combinations and all 480 comparisons passed bitwise equality
and unchanged KV checks. Registers decreased from 233 to 225, with no compiler
reported stack or spills.

| Query tokens / rows / history base | Previous exact c322, µs | This change, µs | Reduction |
| --- | ---: | ---: | ---: |
| 1528 / 1 / 0 | 331.11 | 325.59 | 1.7% |
| 2048 / 8 / 2048 | 1034.75 | 1019.17 | 1.5% |
| 1528 / 8 / 4096 | 1457.31 | 1429.76 | 1.9% |

Previous values are from the preceding campaign, not interleaved in this run.
The interleaved reference is the original c318: 459.98, 1645.77, and 2590.30 µs
respectively. Therefore the much larger cumulative improvement against c318
must not be attributed to this change alone.

The previously measured vLLM/SGLang attention baselines still remain faster.
Those are not fresh end-to-end measurements, and their launch/layout caveats
from the preceding report still apply.

Remote measurement directory:
`/tmp/lunaflux-instruction-compiler3-timing-20260914-r1`.
Generated source: `/tmp/lunaflux-instruction-compiler3-20260914.cu`.
The runtime selection and full-model benchmark must be verified independently;
emitting an improved c322 does not prove the deployed graph selects it.

Validation: warning-denied native check, source tests 32/32, parallel AOT tests
4/4, full native suite 3739/3739, and all nine memcheck/racecheck/synccheck runs
across the three representative workloads passed. The full native build emits
existing Clang allocation-probe attribute warnings; MoonBit checks passed.
