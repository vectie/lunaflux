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

Implementation `158d80ed` is pushed. Full local warning-denied native tests pass
5,157/5,157 with existing migration exclusions unchanged; `moon info` reports
561 existing warnings and zero errors for the selected package closure.
Scoped administrator NCU capture of the R22 diverse fixture is retained and
downloaded as `paired.ncu-rep` in the probe root. It does not modify driver
permissions. Both selected kernels use grid512/block128 and identical operands:

| Counter | Separate staging | Shared left staging |
|---|---:|---:|
| Replayed kernel duration, ms | 1.857632 | 1.121632 |
| Warp instructions | 39,944,192 | 27,113,472 |
| Registers/thread | 71 | 56 |
| Actual allocated shared bytes/block (including overhead) | 14,848 | 12,544 |
| Active-warps percentage | 43.77% | 60.41% |
| Barrier stall percentage per active warp | 20.84% | 15.25% |
| Long-scoreboard stall percentage per active warp | 73.36% | 52.23% |
| Hardware shared-load bank-conflict counter | 524,793 | 263,318 |

These are same-fixture native kernel counters, not vLLM counters or independent
additive stall-time components. NCU replay timings differ from warmed complete
chain timings. The counters substantiate less supporting work and waiting;
the full-model measurement must establish the practical benefit. Bank counts
are not zero and are not treated as the objective. Requested DRAM-byte data
was not present in this exported report; no byte-traffic claim is inferred.

## Full-model decision and restored-workload interference

The pinned `158d80ed` candidate runs at
`/tmp/lunaflux-model-bench-glm-paired-20261014-v1` and
`/tmp/lunaflux-model-bench-glm-paired-repeat-20261014-v1` selected the new AOT
on both ranks (42 paired expert sources/rank, 34 fused KDA sources/rank).
Both native ranks closed their contexts and exited zero; all generated IDs
remain identical to the retained native baseline. The first cohort completed
before the external workload reappeared. The repeat is **not an uncontaminated
performance sample** (see timestamps below).

| Prompt | Input tokens | Native / vLLM output tokens | Retained native repeat s | Paired first s | Paired repeat s, contaminated | Fresh vLLM s |
|---|---:|---:|---:|---:|---:|---:|
| Fact | 22 | 4 / 4 | .608 | .612 | .584 | .540517 |
| Arithmetic | 23 | 3 / 4 | .534 | .538 | .522 | .392749 |
| Chinese | 22 | 8 / 8 | .950 | .963 | .931 | .652727 |
| Logic | 23 | 8 / 8 | .973 | .990 | .957 | .669396 |
| Code | 24 | 8 / 8 | .983 | .999 | .962 | .672864 |
| Retrieval | 36 | 3 / 3 | .652 | .665 | .648 | .376539 |
| Total (unequal output work) | | 34 / 35 | 4.700 | 4.767 | 4.604 | 3.304792 |

First-run TTFT seconds: .333/.346/.338/.359/.368/.471. Repeat TTFT:
.324/.337/.326/.346/.356/.460, retained with the same interference caveat.
One warmup was excluded from each cohort. Retained-native repetitions total
4.679/4.700 s, versus the preceding shape-aware repeat 4.865 s: about 3.4–3.8%
less completion time, not performance parity. With equal output counts only,
the retained five-prompt sum is 4.152/4.166 s versus vLLM 2.912043 s:
**42.6–43.1% longer**. Native CLI and reference buffered HTTP timing boundaries
and BF16 versus FP8 KV remain different. This is a short-prompt latency cohort,
not decode-only throughput, long-context coverage or quality certification.

The first paired run is 1.43% slower than the retained 4.700 s repeat. Therefore
shared FP4 staging remains an explicit `ExpertGroupingPlan` schedule option,
**not the default**. Compiler legality does not depend on the reference or
benchmark; measured scheduling decisions stay outside the numerical contract.
The ordinary live-grid lowering is retained. Full native tests after this
selection decision pass 5,157/5,157; the alternative and mismatched-capacity
rejection are regression tested. Component success is not silently promoted.

The original deployment UID `691da90e-1724-4272-8910-93327a105a53` was restored
to one replica by an external actor/controller before this task restored it.
Its new pod `qwen3-llm-db96595dc-l7tgq` (UID
`3ee090b1-332d-4925-8a0e-2bed255f251f`) was created at 2026-10-14 06:36:11 UTC,
during the second native cohort. It restarted three times; the previous
container exited 1 at 06:38:30 and the current container started 06:38:54.
The second native cohort retired at 06:39:03. Thus the faster 4.604 s result
cannot establish an exclusive-host gain, even though its host reserve stayed
above 22.51 GiB and native task swap remained zero. The first paired cohort
retired at 06:34:32, before the new pod existed.

The final candidate trace root
`/tmp/lunaflux-model-bench-glm-paired-trace-20261014-v1` failed its prelaunch
GPU-owner check: PID 3601274 (`VLLM::EngineCore`, started 06:39:22) belongs to
that Kubernetes pod's cgroup. **No candidate trace was launched**, and there
is no full-model selected-kernel timing attribution for this last schedule.
Its unused local watcher was terminated. The workload was left at the user's
requested one replica, not stopped again. At the 06:42:41 UTC check it was
Running but not Ready (startup health port not yet listening); this is not a
service-readiness success claim. No deployment resources, manifests, payloads
or unrelated files were changed during that restoration check.

The user requested restoration of `qwen3-llm` to one replica after the final
benchmark/profiling owner has retired. Do not restore concurrently with a
benchmark; record actual restoration and readiness separately.
