# Historical page-ID batching — 2026-10-05

## Decision and scope

The source-correlated AKO loop found a serial metadata dependency in the selected
query-owned prefill kernel. Batched page producers reduce kernel time in all six
paired cells on both Sparks: **9.38–11.74% median paired reduction**, with every
individual pair positive (worst pair8.82%). Accept this source optimization after
boundary tests and sanitizer checks. Serving propagation is being measured below;
kernel gains are not whole-engine gains or a refreshed vLLM/SGLang comparison.

The baseline is the frozen explicit `approx-base2-f32-v1` Q64/K64 c30322, not the
strict-exp variant. Numerical expressions, reduction order and approximation
permission are unchanged. Both GPUs are GB10 sm121. GPU179 UUID is
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`; GPU178 UUID is
`GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`.

## Hypothesis and implementation

In the baseline, the complete historical staging loop issues a page-table load,
immediately checks its bounds, broadcasts the ID, then issues the vector copy.
This happens serially for eight vector slots. Matched PC sampling identifies the
load-dependent bounds check `ISETP.GE.U32.AND ...0x4000` as the hottest metadata
site; reducing output multiplies previously did not produce a reliable gain.

The terminal CUDA renderer now consumes the existing backend-neutral
`PagedTileEpochs` ownership relation. When a whole historical tile is page aligned
and its page set fits the subgroup, owner lanes fetch all page IDs together before
the vector consumers. Each lane retains one ID, not an eight-element local array.
Consumers borrow the validated page identity with a subgroup shuffle. Physical
page numbers need not be contiguous. Invalid-page publication, zero filling,
retained V addresses, K/V lifetimes, transfer order and barriers remain intact.
Partial/mixed/current tiles and overfull owner maps retain their checked read maps.

The actual caller passes the immutable selected key-tile extent. No model name,
request-path cache/JIT, profiler query, new heap allocation or device check is
introduced. CUDA lane/shuffle details remain in terminal lowering. This realizes
an existing pure ownership IR rather than adding another unused optimization plan.

Changed production files: `source_query_stage.mbt`, `source_query_owned.mbt` in
`kernels/luna_cuda_attention_tile_source`. Regression tests cover page producers
before consumers, invalid-page publication, owner capacity32 and overflow/tail
fallback, split/dense domain legality, retained V and absence of new barriers.
Existing source snapshots change only for affected split+dense families.

## Finite dual-device measurements

Five alternating pairs per cell, 30 GPU-event repeats per member. Q2048, actual
runtime envelopeQ2048/rows32. The columns are medians in microseconds; paired gain
is the median of individual `1-new/old`, not the ratio of medians.

| Host | Active rows / history | Baseline µs | Batched µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 7,285.71 | 6,408.41 | 11.74% | 11.34% |
| .178 | 2 / 28,672 | 7,200.56 | 6,394.97 | 11.32% | 10.20% |
| .178 | 2 / 8,192 | 2,114.93 | 1,909.31 | 10.03% | 8.82% |
| .179 | 1 / 28,672 | 7,368.02 | 6,621.11 | 10.14% | 9.35% |
| .179 | 2 / 28,672 | 7,396.25 | 6,613.93 | 10.67% | 9.82% |
| .179 | 2 / 8,192 | 2,174.09 | 1,979.52 | 9.38% | 8.89% |

All30 paired observations match the baseline output bitwise. The three irregular
Q129/R2/H128 sanitizer checks report zero errors/hazards; bitwise=true, oracle
maxabs0.000330008, below ceiling0.003. Scoped native checks and affected tests pass
120/120 with the repository's existing warning exclusions20/79/29/25; this is not
a claim that unrelated dirty packages pass warning-denied aggregate checks.

Parallel .178 coverage retains the same pinned aggregate KV capacity; history
decreases as active rows increase. All outputs match bitwise. These are separate
kernel shapes, not an equal-history serving batching curve:

| Q / rows / history | Baseline / batched µs | Median paired reduction | Worst pair | Decision |
| --- | ---: | ---: | ---: | --- |
| 129 / 2 / 127 | 17.180 / 17.159 | 0.12% | −3.51% | Inconclusive |
| 2048 / 4 / 28672 | 7293.46 / 6497.59 | 11.19% | 10.31% | Improved |
| 2048 / 8 / 14336 | 3972.78 / 3582.21 | 9.92% | 8.04% | Improved |
| 2048 / 16 / 7168 | 2963.33 / 2874.07 | 3.12% | 1.36% | Inconclusive at conservative gate |

The optimization is not credited as a short-tail/C16 robust win. Sealed extra
archive `larger-trial178/measurement.tar.gz`, SHA-256
`93fc9a7d29d30c7c51f44a44611a249115d9046e64501baf08a6471d70590780`;
download hash and all manifest entries verify.

## Matched hardware counters

Same Q2048/R2/H28672 launches, baseline then batched, NCU179. These profiled
durations do not replace unprofiled timing above. Counts are executed warp
instructions, not byte counts or elapsed-cycle allocations.

| Metric | Baseline | Batched |
| --- | ---: | ---: |
| Warp instructions | 936,583,040 | 891,530,112 |
| Scalar global load `LDG.E` | 7,382,656 | 960,128 |
| Shared load `LDS` | 18,471,936 | 3,791,872 |
| Branch `BRA` | 23,309,312 | 10,464,256 |
| Register `MOV` | 100,893,696 | 97,258,496 |
| Tensor `HMMA` | 119,668,736 | 119,668,736 |
| Async copies `LDGSTS` | 14,958,592 | 14,958,592 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Registers/thread | 234 | 234 |
| Register/shared-limited resident CTAs | 2 / 2 | 2 / 2 |
| Long-scoreboard / average warp latency | 20.93% | 10.22% |
| Wait / average warp latency | 37.85% | 32.74% |
| Branch-resolving / average warp latency | 3.13% | 0.59% |
| Barrier / average warp latency | 2.40% | 5.29% |
| Issue active | 29.62% | 31.15% |

The metadata instruction chains collapse without changing matrix work, copy
count or barrier count. Bounds checks still wait for page loads: the hottest
remaining check samples increase46,031→54,635, and total sampled not-issued
long-scoreboard counts95,813→105,050. Therefore do not equate the smaller aggregate
stall fraction with every PC having fewer samples, or claim all load/barrier
latency eliminated. The robust timing improvement and executed instruction
reduction support the optimization, not a universal zero-stall claim.

## Exact identities, preservation and limits

Baseline cubin:
`66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`.
New cubin:
`7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57`.
New CUDA source:
`55ceeda9b04e30239ef289004bd64bdafc4b3436dc0d0557d2bde7b1310b9f9e`.
Symbol remains `lunaflux_attention_prefill_tile_compiler_exp2_v1`.
Two offline compilations produce identical cubin hashes; source hash, not an
unchanged symbol or candidate number, distinguishes this implementation.

CUDA13.0.88 nvcc hash:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
One GPU workload per device, paired trials run concurrently across devices.
GPU processes have16GiB MemoryMax, zero swap and600-second deadline; CPU builds
have8GiB and bounded parallelism. The continuous reserve is32GiB MemAvailable;
observed minima exceed122,038,096KiB. No source compilation runs during paired
GPU timing on the same device. Serving children retain their separate64GiB cap
and reserve monitor; a parent CPU cap is not misrepresented as bounding siblings.

Remote roots:

- .179 `/home/wlc004s/lunaflux-ako-page-batch-20261005.gbQSnTxD/experiment`
- .178 `/home/wlc003s/lunaflux-ako-page-batch-20261005.glmkHPPS/experiment`

Local `/tmp/lunaflux-ako-page-batch-20261005.I4DZbBDu`, with report, raw counters,
source/SASS, trials and downloaded immutable archives. Archive hashes:

- .179 `7ebebe93dcca271798d8b8a2d9e9b9e18ed523a12e8652e30116083da282da82`
- .178 `a5723b1529a72f28cc22dd96bcc54f97660c4314ed2cc3a8eafe1c19c40498ee`

Both downloaded hashes and all manifest entries verify. Additional serving and
long-memory outputs are preserved separately; sealed parents are not overwritten.

The additional .178 Q2048/R2/H28672 memcheck also passes with no errors, bitwise
equality and oracle maxabs0.000377474. Zero local bytes,234 registers/thread and
two resident blocks remain. Separate archive `long-check178/measurement.tar.gz`
has hash `6a0a5bf7fed05fbf10c572ee8ae12a596ee772860eef620a085e2912fb7629e6`;
download hash and all manifest entries verify.

## Serving follow-up

New module is bound into a fresh nine-module runtime: other eight modules and
launch geometries are checked unchanged. Fresh route scope
`015b96fb10d16359e7b9fb77c86401a69af408366850fe96469a7aa1fa7b5b66`.
Six scope-bound route cells complete before serving admission. The comparator is
the latest measured exp2 serving bundle, not strict prefill, so earlier exp2 or
partitioned-decode gains cannot be credited twice. The six cells select ordinary
prefill/mixed and partitioned pure C2 decode. Their slow partitioned-prefill
alternatives are not reported as a regression against the old ordinary kernel.

The unprofiled four-start ABBA comparison completes on .179, with identical
request bodies, one warmup and three measured trials per cell per fresh start.
Each side has two fresh starts/six measured trials. All requests produce64 output
tokens. Completion/TTFT below are milliseconds; tok/s counts output tokens only.

| Input / concurrency | Baseline completion | Batched completion | Reduction | Baseline / batched TTFT | Baseline / batched output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16,384 / C1 | 1994.5 | 1965 | 1.48% | 1040 / 1007 | 32.09 / 32.57 |
| 32,512 / C1 | 4520 | 4356 | 3.63% | 3049 / 2874.5 | 14.16 / 14.69 |
| 32,512 / C2 | 8752 | 8398 | 4.04% | 4611 / 4349 | 14.63 / 15.24 |

TPOT baseline/batched:14.888889/14.888889,23.007937/23.126984 and
65.111111/63.888889ms. The expected prefill saving survives serving; kernel10%
does not become whole-engine10%. Both long-cell TTFT reductions are approximately
5.7%. C1 token vectors match across sides and repeats. C2 has up to45 changed
positions within each side and across sides, including warmup vectors. This is
the pre-existing unresolved C2 trajectory issue, not deterministic or quality
parity admission. Raw timing samples, request bodies and token vectors are retained.

Serving archive `abba179/measurement.tar.gz`, SHA-256
`238f13b518722dd4a902cdbccb372cdc11c19d05a779f5e04de817c919965226`;
download hash and all manifest entries verify. The parent comparison finishes
within its600-second deadline (364.7s), zero swap. A separate selected-symbol
trace is a dispatch check, not an unprofiled performance sample.

Fresh C1/C2 Nsight traces confirm the selected approximate entry point executes:
672/1400 calls, grid63×16×1, block128,234 registers/thread. The separate unchanged
strict wide-prefill module also executes224/392 calls; not every prefill call is
this changed path. The model descriptor pins fused bundle
`add504e337b0ad8867342d043ed96cb3a11212f74231984925dc7ca6ecb79724`;
the assembled kernel-root bundle has that hash and its embedded module6 is exactly
the new `7b42b353...` cubin. Its copied local hash verifies. This fused sidecar is
embedded in `kernel-release/kernel-root/reusable-fused-runtime-bundle.v3`, not a
standalone entry in the base execution manifest's `sha256/*.cubin` directory.
Worker hash remains `dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`.

Trace archive `trace179/measurement.tar.gz`, SHA-256
`66712b05fe50dc86e2b9bb3723d4f62f2294d44d858aa226d540679bb32807db`;
download hash and all manifest entries verify. Both GPUs are idle after the
completed campaigns. These measurements do not imply a production deployment or
a new benchmark against vLLM/SGLang.

The initial `abba` launcher stopped before inference because its basename
reused a transient bridge unit from an earlier campaign. The failed output is
preserved. The replacement `abba-page-batch` has a distinct unit namespace; no
production service or sealed artifact was modified to retry.
