# Equal work kernel and route comparison

The retained Spark traces establish that LunaFlux is slower than vLLM on several
identical logical query/history vectors. A prior identical-operand kernel A/B
also establishes that removing one redundant dependent load reduces execution
time. These results support kernel implementation and executable route selection
as real optimization targets. They do not prove that hand-written kernel code
alone causes the entire serving gap, or assign every millisecond to an
instruction-level cause.

This is a new analysis of retained 4096-input, 64-output, C16 traces, after the
affine prefill repair. It is not a new GPU benchmark. Hardware, models and pinned
baseline versions are described in the
[serving report](BENCHMARK_AFFINE_POSITION_REPAIR_2026-10-03.md).
No production source, binaries, containers or runtime routes were changed.

## Logical work is equal against vLLM

The new analyzer reads actual sorted query/history vectors, not capacity grids.
For a row with q queries and p past tokens, the causal attention work measure
is `q*p + q*(q+1)/2` query-key pairs per query head. This counts mathematical
work, not physically executed MMA operations, padding or memory transactions.

| Measured window ledger | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Steps | 96 | 96 | 73 |
| Query tokens | 66,544 | 66,544 | 66,560 |
| Query tokens in rows with more than one query | 65,536 | 65,536 | 65,536 |
| Query tokens in one-query rows | 1,008 | 1,008 | 1,024 |
| Causal query-key pairs per query head | 138,411,520 | 138,411,520 | 138,478,080 |
| Pure C16 steps | 44 | 32 | 64 |

LunaFlux and vLLM execute equal aggregate logical token and attention work.
Their distribution among pure and mixed steps differs. Equality of total work
does not establish equal cache behavior, scheduling or physically padded work.
One-query rows are not classified as decode solely from their query length.

The SGLang window includes sixteen additional one-query tokens and four orphan
kernel calls totaling 15.040 us. Its chunking differs; this ledger is not an
exact SGLang work match. The analyzer retains these differences rather than
subtracting them by assumption.

## Equal work chains are slower

Ten exact query/history vectors occur in both LunaFlux and vLLM, covering ten
steps per engine. There are no exact common vectors with SGLang. These ten
matches cover only 10 of 96 LunaFlux steps, so their ratios must not be
extrapolated to the entire window.

The following values are sums of observed CUDA kernel durations over the
28-layer chains, not exclusive wall time or dependency critical paths.
Cross-framework operands, arithmetic laws, physical paging and cache states
are not identical.

| Exact logical work | Chain | LunaFlux ms | vLLM ms |
| --- | --- | ---: | ---: |
| One row, q2048 p2048 | Attention | 25.076 | 21.970 |
| One row, q2048 p2048 | Projection and postops | 41.480 | 32.536 |
| q2047 p0 plus q1 p4096 | Attention | 27.339 | 10.450 |
| q2047 p0 plus q1 p4096 | Projection and postops | 40.637 | 32.423 |
| q2047 p2047 plus q1 p4097 | Attention | 42.463 | 22.967 |

For q2048 p2048, LunaFlux has 112 projection/postop calls versus 225 in the
reference, yet the chain activity is 27.5% higher. Fewer kernels do not ensure a
faster chain. Reference kernels that cannot be reliably assigned remain
unmapped: 1.966 ms for this step. Even charging all that activity to the reference
projection chain gives 34.502 ms, still below LunaFlux's 41.480 ms. This
conservative comparison does not depend on pretending unknown GEMM symbols
identify output or down.

The first q2048 p0 step has only 109 observed LunaFlux projection/postop calls;
it is not used for a complete projection-chain conclusion. The final single-row
match also contains substantial unmapped reference activity. All raw groups and
call counts remain in the output instead of silently treating missing calls as
equivalent full chains.

## Mixed attention selects a costly decode companion

For the exact q2047 p0 plus q1 p4096 step, the trace decomposes LunaFlux attention
as follows:

| Selected attention component | Calls | Kernel activity ms |
| --- | ---: | ---: |
| Normal matrix prefill | 28 | 9.226 |
| Unpartitioned blockwise decode | 28 | 18.113 |
| Whole LunaFlux attention chain | 56 | 27.339 |
| Whole vLLM attention chain | 28 | 10.450 |

The observed LunaFlux decode launch is grid32x8x1, block64, 78 registers and
33,040 bytes shared memory. The frozen generated decode source maps x to decode
row and y to KV head, rejects `local_row >= decode_count`, and has one partition.
With one decode row and eight KV heads, only eight CTAs have active row work on
the 48-SM GPU. This is a source-derived active work count, not a new hardware
utilization measurement. Raising the capacity grid does not create more useful
decode work.

The owning lowering is
[source_blockwise_decode.mbt](../kernels/luna_cuda_attention_tile_source/source_blockwise_decode.mbt).
The startup mixed graph inserts a disjoint decode companion after prefill in
[attention_phase_prepare.mbt](../engine/device_step/attention_phase_prepare.mbt).
The route admission in
[measured_attention_routes.mbt](../engine/device_step/measured_attention_routes.mbt)
restricts ordinary and partitioned decode route IDs to pure DecodeGraph buckets;
those IDs are not automatically usable as mixed graph alternatives.

Across the seven exact common mixed vectors, LunaFlux attention activity exceeds
vLLM by 16.092 to 20.120 ms per step. This directly locates substantial activity
in mixed execution. It does not prove that replacing the companion with a
partitioned implementation will recover the complete excess: that requires a
same-work, same-law full mixed-graph A/B. Prefill and decode have disjoint output
row domains; two kernel families alone are not proof of duplicate computation.

This is an execution decomposition and selection issue as well as a kernel
implementation issue. It is not evidence that functional compiler architecture
is inherently slower or that another IR layer alone will solve it.

## An implementation change has a causal A B result

The previous paired affine-position experiment removed a redundant dependent
position equality load. It preserved arithmetic, operands, launch geometry and
allocated register residency. Six correctness cases were bitwise identical;
memory, race and synchronization checks passed.

Unprofiled median event time fell from 935.664 to 853.285 us, an 8.8% reduction.
Allocated registers remained 232 and residency remained two blocks per SM.
This is a concrete intervention proving that implementation-level load dependency
removal can accelerate the kernel; it is stronger than inferring speed from
instruction count or occupancy. Details and limitations are in the
[affine repair report](BENCHMARK_AFFINE_POSITION_REPAIR_2026-10-03.md).

It does not establish an 8.8% framework improvement. Ordinary long64 completion
time improved by 1.18%, and its package refreshed route measurements, so that
serving result is not an isolated same-route causal A/B.

## Reproduction and remaining uncertainty

[compare_equal_work.mbtx](../benchmarks/gpu_pipeline/compare_equal_work.mbtx)
is a diagnostic MoonBit tool. It passes formatting, strict native warning-denied
checking and self-tests for row work, history bounds, exact intersections,
occurrence normalization and whole-chain categorization. It never modifies
input traces and uses CreateNew for output.

Inputs are `affine-trace-v2/work-shapes.json` and the vLLM/SGLang work tables
under `capture/` in the previously downloaded, verified affine archive.
The archive SHA256 and reproduction paths are retained in the serving report.
The new detailed result is
`/private/tmp/lunaflux-equal-work-20261003.Z7aVuHu2/comparison-detailed.json`.

What is established: aggregate logical work equality against vLLM, slower
matched chains, a costly selected mixed decode companion, and one causal
same-law implementation improvement. What remains unproven: the instruction or
memory cause of each remaining chain excess, a complete causal allocation of
the 17% serving gap, and exact equal-work claims against SGLang. No new GPU
workload was necessary for this analysis.
