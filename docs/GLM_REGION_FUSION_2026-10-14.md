# GLM independent-region optimization

The retained shape-aware implementation (`3864529e`) is the baseline. Its
selected trace attributes 23.562 ms of decode KDA time to projections and
137.602 ms of prefill to routed experts. Profiled NCCL time includes peer
waiting and must not be presented as pure network cost.

## Implementation boundary

1. Represent independent same-input projections as a pure segmented matrix
   domain. Fuse their launches without changing each output's arithmetic,
   publication, or weight ownership. Recurrent semantics bind six independent
   outputs; only CUDA lowering knows the thread/block mapping.
2. Represent bounded indirect expert tiles independently of maximum capacity.
   Use a bounded grid over live groups, retaining the exact ordered matrix
   accumulation and explicit shared-memory producer/reader lifetimes. Measure
   grouping, gate/up, down and packing separately.
3. Batch independent F32 collective buffers without adding their values before
   reduction. Preserve separate BF16 rounding and expert-combination order.
   Storage, readiness and publication cuts must be explicit startup plans.

Comparison is external to compiler legality. No payload hashes, tolerance
changes, runtime JIT, or reference-dependent source selection are introduced.

## Bounded validation and experiment budget

At most two schedules per region; unit/source tests, deterministic matched
GPU checks and sanitizer/leak gates for changed kernels. Ablate the changes
against the retained runtime before two final native cohorts and one fresh
vLLM cohort. Record actual token IDs/counts, selected symbols, first-token and
completion times, and memory minima; differing output lengths are not equal
work. Keep all numerical and timing regressions visible.

Both Sparks have a 100 GiB per-task ceiling, zero task swap, and an independent
16 GiB available-host-memory reserve. Never overlap GPU owners. A reserve
breach stops the experiment; it does not authorize reducing the reserve.

## Component measurements and first ablation

Actual generated-source probe `/tmp/lunaflux-glm-regions-20261014-v3`
compares the separate and fused six-projection chains. Ragged K=33 and
full K=4096 checks cover rows 0, 1, 3, 16, 17, 64 and out-of-capacity 65.
Outputs are bitwise equal; an independent dyadic-input CPU dot oracle also
passes. The pack/unpack transport preserves arbitrary F32 bit patterns.
Memcheck/full leak/race/sync checks report zero errors. For the full-width
fixture, sanitizer coverage is rows 0/1/65, not the entire 64-row geometry.
The ragged fixture has the complete row sweep.

| Live rows | Separate six projections, median ms | Fused median ms |
|---|---:|---:|
| 1 | 0.482421 | 0.464650 |
| 16 | 1.136065 | 0.542472 |
| 64 | 1.147226 | 0.705267 |

Four alternating trials are retained. These are component-chain times, not
full-model gains. Earlier v1 harness namespace compilation failed; v2 exposed
six separate scratch reservations exceeding 48 KiB. The pure segmented plan
now gives mutually exclusive regions one shared scratch lifetime; v3 passes.

FP4 probe `/tmp/lunaflux-glm-fp4-performance-live-tiles-20261014-v1`
uses the retained staged-20261014-v2 source and identical full-width operands,
capacity 64, eight resident experts and eight selected routes. Median complete
chain microseconds are 411.096→297.136 (R1), 449.114→330.254 (R3),
832.827→727.608 (R22), 1719.290→1458.309 (R64). Capacity and trimmed
launches preserve every output/workspace bit. Full-width R1 memcheck/leak/
race/sync checks pass. This synthetic route corpus does not cover all actual
288-expert distributions; the full-model benchmark remains necessary.

The first committed ablation selects fused projections and live expert tiles,
but leaves collective coalescing unselected (132 sites). The coalescing
implementation is available for a separate subsequent TP2 measurement; no
claim about its full-model correctness or benefit is made from pack tests.

## Benchmark host availability

With explicit user authorization, `lunanexa/deployment/qwen3-llm` on the
management cluster was scaled from one replica to zero to free Spark .178.
Deployment UID: `691da90e-1724-4272-8910-93327a105a53`. The prior pod was
`qwen3-llm-db96595dc-g9wm2`; Kubernetes owns its retirement. No workload data
was removed. After retirement, .178 had no GPU compute owner and
126,307,774,464 available host bytes. Pre-existing host swap residency is
separate from the zero-swap limit enforced on each benchmark unit.

Restoring the one-replica workload is reversible, but its previous footprint
left only about 14.1 GiB available. Do not silently change its resources or
restart it concurrently with the paired benchmarks.

## Pinned builds and first full-model result

Implementation `001f5de1` was built cleanly on ARM in
`/tmp/lunaflux-checkpoint-build-glm-regions-20261014-v1`; clean affected
package tests passed 54/54. The new TP2 selection (`9dac199f`) builds in
`/tmp/lunaflux-checkpoint-build-glm-coalesced-20261014-v1`; local binding tests
pass 26/26. Warning-denied native check passes with the existing migration
warning exclusions unchanged. The full local native suite passes 5,156/5,156;
the unrelated dirty migration/TLS files were neither staged nor reverted.

The initial native cohort is
`/tmp/lunaflux-model-bench-glm-regions-20261014-v1`. Both ranks exit zero,
close their contexts and retain the host reserve. AOT compilation finished
before measured requests. Six completion times are .606/.537/.951/.966/.982/
.659 seconds (total 4.701 s versus retained repeat 4.865 s); every generated
token ID is unchanged. First-token times are .335/.348/.333/.358/.364/.478 s.
This 3.4% reduction is one ablation, not proof of a stable universal gain;
the historical first/repeat runs varied. It must not be represented by the
much larger isolated component speedups.

TP2 coalescing joins independent lanes, preserving the same two F32 operands
and the separate consumer rounds. Larger topologies are not selected, because
message size can change a multi-rank reduction tree. Planned rank memory is
98,648,722,228 / 99,918,135,092 bytes, under the 100 GiB limit. The 42 joined
MoE regions add 396,361,728 bytes per rank, replacing 132 collective sites by
90. Full-model results for this second ablation must establish the net effect
of saved collectives versus extra copies.

Fresh vLLM preparation v1 failed before model launch because the obsolete
default SSH socket was used. That directory is retained; v2 uses the current
sockets and identical literal prompt IDs. No container download or model
payload integrity scan is part of this experiment.

The two coalesced cohorts (`glm-coalesced-20261014-v1` and
`glm-coalesced-repeat-20261014-v1` under `/tmp/lunaflux-model-bench-`) finish
in 4.679 and 4.700 s, with all native output IDs unchanged. Their benefit over
the 4.701 s non-coalesced ablation is 0.47% and 0.02%: **no demonstrated
material end-to-end gain from collective batching**. Both ranks explicitly
log `sites=90` and `prepared_replay_graphs=91`, so this is not missing source
propagation. The extra memory/copy cost remains visible. Minimum observed
host available memory across the two cohorts is 23,688,824 KiB (22.59 GiB);
all unit swap observations are zero. Contexts close and both ranks exit zero.

## Selected trace and second expert schedule

The coalesced full-model trace is
`/tmp/lunaflux-model-bench-glm-coalesced-trace-20261014-v1`.
On egress step 3 (prefill), KDA falls 55.597280 to 35.100544 ms, but routed
experts remain 137.601856 to 138.060672 ms. The latter comprises gate/up
74.721536, down 36.826272, grouping 15.895968 and activation global
quantization 8.178176 ms. These are profiled GPU intervals, not unprofiled
request latency. NCCL intervals contain peer waits: their prefill reduction
does not demonstrate faster network transport, and decode NCCL time increases.

The remaining, second expert-schedule trial consumes the existing immutable
`ExpertGroupingPlan.shared_input_staging` and `OperandStorage` plan. One left
operand owner feeds independent gate/up ordered MMA reductions. Weight regions
and scale rows remain distinct; each reduction retains ascending K order and
its original publication. Shared scratch is 11,520 bytes. The CUDA lowering
publishes both right operands together and retires all readers before reuse.
No new model-specific arithmetic, runtime compilation or reference selection
is introduced; the down schedule is unchanged.

Probe `/tmp/lunaflux-glm-fp4-performance-paired-20261014-v1` compares against
the pre-edit live-grid source frozen locally in
`/tmp/lunaflux-glm-fp4-performance-live-frozen-20261014-v1`.
All outputs/workspaces match bitwise for R1/3/22/64 and both capacity/trimmed
grids. An additional 64-logical-expert routing corpus uses eight repeated
synthetic weight banks: it changes grouping occupancy, **not** the full
checkpoint's weight-cache footprint. In that corpus, median R22 chain time
is 2266.910 to 2003.075 us and R64 is 4555.304 to 4194.816 us. R1 is nearly
unchanged (301.350 to 301.736 us); the eight-expert repeated R22 corpus does
not establish improvement. Full-width R1 and diverse R3 capacity-grid
memcheck/full-leak/race/sync checks report zero errors. Diverse R3 has 22 live
groups, exercising repeated work-tile ownership in the bounded grid.

Fresh vLLM `/tmp/lunaflux-reference-bench-vllm-glm-regions-20261014-v2`
completed all six literal requests with HTTP 200. Completion times are
.540517/.392749/.652727/.669396/.672864/.376539 s (3.304792 s total).
The equal-output-count five-request subset is 2.912043 s. Its minimum observed
host reserve is 17.419 GiB. The cached backend emits the existing gate/up
global-scale mismatch warning; speed and text agreement are not correctness
certification. After successful responses, controlled rank-one shutdown
returns 137; the watcher flags the failed unit, but no reserve breach occurred
and both GPU owners retired. Preserve that shutdown result rather than
misreporting it as a clean rank-one exit or an OOM.

The user requested restoration of `qwen3-llm` to one replica after the final
benchmark/profiling owner has retired. Do not restore concurrently with a
benchmark; record actual restoration and readiness separately.
