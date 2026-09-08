# Block-tiled ordered fold pipeline

The projection compiler now describes a two-stage immutable operand pipeline
for the materialized MLP-down fold. It groups four independent 16-row maps into
a 64-row block and transfers 32 reduction elements per stage. Each output dot
still consumes its 16-element matrix reductions in ascending order. Single-row
execution shares each input load across four independent output folds.

CUDA lowering realizes this plan with double-buffered asynchronous shared
copies, cooperative input loading, and register accumulators. Model builders
and scheduling contain no CUDA-specific additions. The implementation covers
the split BF16 MLP-down path and, in the follow-up below, its gate/up producer.
QKV and other dense GEMMs have **not** all been converted to this pipeline.

At least 256 tokens are required before grouping rows. Complete 64-row blocks
use the pipeline; any remaining rows use the existing 16-row realization.
This avoids computing a full 64-row tile for a one-row remainder. Reduction
extents must be at least 512 and divisible by 64. Existing AOT companion launch
dimensions remain valid because the consumer traverses its work grid by stride.

## Paired GPU results, 2026-09-09

RTX 5060 Ti, sm120, CUDA 13.1.115; BF16 input/output widths 1024 and intermediate
width 3072. These timings include **both gate/up and down**, not just the
modified kernel, and exclude model loading and serving overhead. Median of
three paired trials, each using 20 captured repetitions after warmup:

| Tokens | Previous MLP µs | Pipeline MLP µs | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 17.130 | 16.928 | 1.01× |
| 8 | 56.856 | 56.714 | 1.00× |
| 65 | 127.498 | 131.021 | 0.97× |
| 128 | 181.258 | 185.117 | 0.98× |
| 255 | 359.376 | 368.555 | 0.98× |
| 256 | 360.272 | 297.530 | 1.21× |
| 257 | 360.173 | 309.235 | 1.16× |
| 512 | 664.467 | 530.010 | 1.25× |
| 513 | 666.269 | 531.598 | 1.25× |
| 1023 | 1307.643 | 1043.424 | 1.25× |
| 1024 | 1307.624 | 1045.843 | 1.25× |

The full vector also includes 2, 7, 17, 63, 64, 127, and 129. Output and
intermediate workspace are bit-exact against the previous implementation;
unused allocation tails remain untouched. Memcheck, racecheck, initcheck,
and synccheck pass, including 255/256/257 and 513-token cases. Explicit resource
release is checked by the probe.

Small/mid-sized inputs can be about 2–3% slower in this combined executable.
This is not an across-the-board speedup or an end-to-end serving result.
Separate low-row specialization and extension to the other matrix families
remain work. No new vLLM/SGLang comparison was run for this increment.

Rejected experiments included register-only lookahead (4–8% slower on long
inputs), a 16-row shared pipeline, and larger transfer blocks. The selected
32-element transfer reduces shared storage versus 64-element transfers and
substantially improves throughput. Failed experiments are not enabled.

Raw selected campaign: `/dev/shm/lunaflux-pipeline-20260909-r7` on the test host;
downloaded copy: `/private/tmp/lunaflux-pipeline-results-20260909-r7`.
Both generated CUDA sources, cubins, harness, compiler logs, three timing
trials, and four sanitizer logs are retained there.

## Sibling gate/up pipeline follow-up

The same immutable pipeline description now covers the product of the gate
and up ordered folds. A 64-row macro tile shares each staged input between
both folds and each weight fragment between four independent row fragments.
Two 32-element transfer slots overlap operand copying with computation;
16-element reduction order and the existing BF16 SiLU materialization remain
unchanged. Entirely unobserved row fragments are skipped on partial blocks.
The generic compiler selects this transformation from
`FuseSiblingInputTraversals`, not a Qwen model identity. CUDA asynchronous
copies and warp ownership remain private lowering details.

Paired r9 campaign, same hardware, shape, vector, and timing method as above.
The baseline here already includes the committed MLP-down pipeline. Median
of three trials, combined gate/up plus down:

| Tokens | Down-only pipeline µs | Both pipelines µs | Additional speedup |
| ---: | ---: | ---: | ---: |
| 256 | 297.626 | 261.342 | 1.14× |
| 257 | 309.288 | 289.150 | 1.07× |
| 512 | 530.006 | 463.997 | 1.14× |
| 513 | 531.755 | 479.862 | 1.11× |
| 1024 | 1046.155 | 912.408 | 1.15× |

All 18 token lengths retain bit-exact outputs and intermediate workspace and
untouched allocation tails. All four sanitizers pass. The primary producer
uses 122 registers, 24 KiB static shared memory, and no register spills.
The smaller 32-row/K16 experiment was slower on the principal long shapes
(although faster at 257 tokens); it is not selected. This remains one measured
shape, not proof that the chosen tile is optimal for every model or device.
Inputs below 256 tokens retain their prior path.

Raw campaign: `/dev/shm/lunaflux-pipeline-20260909-r9`; downloaded copy:
`/private/tmp/lunaflux-pipeline-results-20260909-r9`.
Full-serving benchmarking and extension to QKV, dense output projections,
and vocabulary projections remain separate work.
