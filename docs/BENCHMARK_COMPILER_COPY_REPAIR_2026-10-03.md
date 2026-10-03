# Compiler operand-copy repairs

This change targets the unchanged projection and one-query attention work
identified by the [mixed-decode trace](BENCHMARK_MIXED_PARTITIONED_DECODE_2026-10-03.md).
It repairs generated operand transport, not model semantics, scheduler policy,
or the number of compiler IR layers. It does not close every framework gap.

## Implementation

Projection bootstrap now uses the physical plan's retained transfer effects for
both initial and future input tiles. Previously the generator forced the initial
input through synchronous global-load/shared-store instructions even when the
physical program already supplied asynchronous wait/publication effects.
Completed-prefetch plans still emit synchronous transport; overlapped-transfer
plans emit asynchronous transport. This removes a generator-only bootstrap
exception without changing the ordered MMA fold.

Decode now has an immutable `PagedTileEpochs` physical ownership plan. For a
page-aligned tile, one subgroup owner loads each page identity once. Consumers
reuse that narrow identity and reconstruct their affine row offsets locally,
instead of redistributing a 64-bit address and validity after distributing the
page identity. That removes three redundant shuffles per vector. Unaligned
partitions, tiles that do not divide into complete pages, and insufficient
owner counts keep the general addressing path.

The portable ownership plan expresses page/alignment legality without CUDA
names. CUDA subgroup width, shuffles and asynchronous instructions remain in
the CUDA lowering. These are compile-time pure transformations; no page scan,
tuning, filesystem work, new allocation or cryptography enters token steps.
The ordered softmax law and arithmetic are unchanged.

## Isolated ordinary GPU timings

Hardware: DGX Spark GB10, sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. CUDA compiler 13.0.88;
MoonBit 0.1.20260920. Timings are medians of five alternating old/new samples,
not Nsight replay durations. No confidence interval is available.

Projection uses 32 request rows. The old/new artifacts have identical register
counts and shared-memory requirements. Every tested result is bitwise equal;
the probe also checks selected scalar dot products and segment boundaries.

| Projection | Tokens | Before µs | After µs | Less time |
| --- | ---: | ---: | ---: | ---: |
| Output | 129 | 20.920 | 20.517 | 1.93% |
| Output | 2048 | 157.810 | 156.494 | 0.83% |
| QKV | 32 | 13.171 | 12.326 | 6.41% |
| QKV | 2048 | 327.008 | 320.966 | 1.85% |
| Vocabulary head | 129 | 1417.952 | 1446.525 | −2.02% |
| Vocabulary head | 2048 | 1416.803 | 1428.867 | −0.85% |

The complete projection vector is 32/129/512/2048 tokens for output, QKV and
head. Several cells are neutral. **No vocabulary-head speedup is demonstrated.**
The projection change has not received a fresh end-to-end serving A/B here.
The QKV row is the generic projection probe, not the full fused ingress chain:
the full ingress renderer already issues asynchronous bootstrap copies. Its
serving performance cannot be inferred from the standalone QKV improvement.

The final decode experiment uses the partitioned c452/p8 schedule and fragmented
physical pages; only page/address transport changes. Old/new outputs are
bitwise equal and independently checked against the probe's scalar attention
reference. History denotes preceding tokens, not padded launch capacity.

| Request rows | History | Before µs | After µs | Less time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4095 | 92.934 | 84.263 | 9.33% |
| 16 | 4095 | 1206.928 | 1174.639 | 2.68% |
| 1 | 4096 | 95.179 | 86.300 | 9.33% |
| 16 | 4096 | 1213.375 | 1173.922 | 3.25% |
| 1 | 8191 | 179.383 | 167.430 | 6.66% |
| 16 | 8191 | 2338.822 | 2342.248 | −0.15% |

The 8191-history C16 result is neutral; these gains are not universal. Compute
Sanitizer memcheck, racecheck and synccheck passed for this final c452 experiment.

A subsequent wider qualification covers c452/p8 and c454/p8, request rows
1/8/16, and histories 126/127/4095/4096/8191: **30 cells**, all bitwise equal,
with six successful sanitizer runs. It reproduces long C1 gains, but c452 C8
at histories 4095/4096 is 1.53–1.56% slower. C16/history4096 improves 1.49%
in this repeat, versus 3.25% in the first six-cell run. The raw vectors are
retained; neither slower cells nor between-run variation are suppressed.

## Rejected experiment

Issuing future K/V immediately after QK, before the current softmax work,
passed numerical and sanitizer checks but slowed the already-page-retained
C16 path by approximately 1.05–1.43% at histories 4095/4096/8191. It was removed
from production lowering. A pure effect plan is necessary for correctness,
but earlier issue is not automatically a faster hardware schedule.

## Serving and counters

The final decode module has been exported through the real candidate frontend,
compiled twice to identical AOT binaries, installed into a new disposable
serving bundle, and given fresh module-bound route measurements. The existing
partitioned mixed runtime and all other AOT modules remain fixed. Both serving
arms use the same diagnostic forced mixed dispatch; no production route-7
measurements are fabricated.

Ordinary serving uses baseline/treatment/treatment/baseline order, two fresh
starts per arm, with one measured trial per cell after warmup. All request
input token IDs and output sequences match. Runtime stderr is empty.

| Input / output / concurrency | Before samples ms | After samples ms | Before median ms | After median ms | Less time | Output tok/s, before → after |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 4096 / 64 / 16 | 4752, 4779 | 4740, 4733 | 4765.5 | 4736.5 | 0.61% | 214.88 → 216.19 |
| 4096 / 256 / 16 | 13091, 13150 | 13007, 12996 | 13120.5 | 13001.5 | 0.91% | 312.18 → 315.04 |

This is a small observed whole-serving effect, not proof of a universal gain
or statistical significance. Minimum sampled MemAvailable was 104,720,048 KiB
(99.87 GiB). No fresh vLLM/SGLang serving measurement was run in this change.
Do not combine medians from earlier machine sessions into an additive speedup.

The changed decode cubin is
`f78ebd69a7a85b8e56d665d2a7ffadd4ccb99190c2c7a78233a13db55e8b89dd`;
the previous module is
`bf663e1055588f47ff62c733714df00baae8e9999b281ef052a243c25f5bbd5e`.
The serving frontend consumes the newly materialized bundle, not an old
diagnostic executable or a relabeled calibration record.
Its materialized bundle embeds the new module bytes. Modules 0–7 retain
their previous digests, and the prepared worker executable is identical
(`0b6dacc451f0b2694b32c517947bf41d174b5bff23895af2a5967bc1e7ee3ca7`).

A fresh Nsight Systems serving trace confirms execution of
`lunaflux_paged_attention_bf16_decode_split_partial_v3_ep_3903` and its
`ep_3904` merge companion: 4,592 calls each across warmup and measurement.
No ordinary decode fallback appears in the observed function inventory.
The trace uses the same measured worker and AOT modules; only a disposable
launcher preserves the profiler environment. It drains successfully with
`child_exit_code=0`, `child_closed=1` and empty runtime stderr. This verifies
selected execution, not an exclusive wall-time attribution or a new timing
benchmark. The aggregate call counts include warmup and must not be compared
to the earlier report's measured-only activity windows.

Final c452/p8 C16/history4096 cold-cache Nsight Compute capture:

| Metric | Before | After |
| --- | ---: | ---: |
| Executed warp instructions | 61,417,344 | 58,128,128 |
| Registers/thread | 80 | 80 |
| Local-memory sectors | 0 | 0 |
| Source-correlated excessive shared wavefronts | 0 | 0 |
| Long-scoreboard stalls per active issue | 3.16 | 3.01 |
| Barrier stalls per active issue | 0.53 | 0.75 |

Instructions fall 5.36%, while barriers remain costly. Stall ratios have a
different denominator from wall time and are not an additive causal budget.
The profiled invocation took 1.255 → 1.232 ms; cache flushing and replay make
this distinct from ordinary replay and serving time. The result supports
less supporting instruction work, not an eliminated memory-pipeline gap.

Earlier page-identity-only counters reduced warp instructions by 3.51%, with
unchanged registers. They precede the final address-shuffle removal and must
not be attributed to that final variant. Cold-cache profiler durations also
must not replace the ordinary timing results.

## Validation and reproducibility

Projection source tests: 101/101. Attention source tests: 62/62. Physical IR
tests: 21/21. Warning-denied native check and formatting passed. A full-suite
run had one unrelated online-TCP timeout; its focused rerun passed 54/54.
The fresh full-suite rerun passed **4,274/4,274**. The timeout was preserved,
not hidden by modifying the unrelated test.

Projection campaign:
`/home/wlc004s/lunaflux-bootstrap-all-20261003.SX8aOFI0`.
Final isolated decode campaign:
`/home/wlc004s/lunaflux-address-local-v2-20261003.QgXJ8rlN`.
Serving campaign:
`/home/wlc004s/lunaflux-page-serving-20261003.Vi2DPJ0s`.
Final counters:
`/home/wlc004s/lunaflux-address-counters-20261003.ypAF22Cr`.
Wider decode qualification:
`/home/wlc004s/lunaflux-address-full-20261003.eqlAdO1y`.
Selected serving trace:
`/home/wlc004s/lunaflux-copy-trace-20261003.pkKwjbIY`.

Downloaded archive: `/private/tmp/lunaflux-page-epochs-20261003.U9JRP6Se/gap-repairs.tar.gz`,
SHA-256 `517eb80e5e0d7992a0ee6acb353a3568db465d4d5118a7bfe669de94f102d146`.
Remote archive: `/home/wlc004s/lunaflux-copy-archive-20261003.FTuqSXxy/sealed`.
The local hash matches. The archive includes all ordinary campaigns, counters,
sanitizer logs, numerical checks and failed/rejected preliminary attempts.
Disposable source/build and serving materialization copies are excluded where
the archive inventory specifies; original remote runs remain preserved.
The separately downloaded `selected-serving-trace.tar.gz` in the same local
directory has matching SHA-256
`7b9c6fc28410e60bb16d8ff6cbdab680d4ad4a872f9be6c997f287d71b536fcd`.

## Remaining performance work

The observed win is small because it reduces supporting transport work, not
the entire QK/PV fold, projection chain or serving step. The final capture
still has material long-scoreboard and barrier waits. Early prefetch alone
was measured and rejected; blindly moving it earlier is not a solution.
The head remains unaccelerated, the c452 C8 regressions require workload-aware
evaluation, and projection bootstrap has not been propagated into a fresh
serving comparison. No claim is made that every kernel is optimal or that
LunaFlux now matches vLLM/SGLang. Those are remaining measured engineering
targets, not reasons to add IR layers without an executable optimization.

Reproduction tools under `benchmarks/gpu_pipeline` are MoonBit `.mbtx` programs:
`measure_projection_bootstrap`, `summarize_bootstrap_pair`, `measure_page_epochs`,
`prepare_decode_epoch_serving`, `run_mixed_decode_ab --compiler-module-ab`, and
`profile_bounded_pair`. Diagnostic containers are bounded to 8 GiB without swap;
serving uses the existing 64 GiB bound. GPU jobs are serialized with a 32 GiB
MemAvailable reserve.
