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
