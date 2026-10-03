# QKV operand binding and immutable KV interval repairs

The selected QKV ingress kernel is faster after replacing opaque fragments with
packed operands and binding their product fan-out jointly. Attention performs
fewer supporting instructions after specializing complete immutable KV
intervals, but mixed-history latency remains nearly flat. Fresh-start serving
completion time falls only **0.57–1.50%** in this small sample set. These repairs
do **not** establish closure of the previously measured projection/post-op or
attention gaps against vLLM/SGLang.

Both changed modules reach actual serving launches. This is not an exporter-only
change. No fresh reference-framework benchmark or production deployment was
performed; old reference timings cannot establish a current cross-framework
ratio. The baseline is the repaired dual-score serving bundle described in
[the preceding report](BENCHMARK_DUAL_SCORE_DOMAIN_REPAIR_2026-10-03.md).

## Source changes and functional compiler boundary

The existing immutable `FragmentProgram` still supplies fragment geometry,
lookahead actions and ordered accumulation. CUDA lowering now loads packed
operands with `ldmatrix` and binds the complete row/column/half product in one
PTX instruction region. This makes shared operands and carried accumulators
visible together to the register allocator. The epilogue consumes the explicit
accumulator ownership mapping rather than an opaque WMMA fragment representation.
Reduction order, geometry, staging count and numerical law are unchanged.

The final ingress binding defines inactive B column owners as zero. This fixes
an uninitialized-operand hazard introduced by evaluating a joint product whose
column owners include an unstored tail. Independent fixtures now execute the
small-dimension/tail cases rather than merely compiling them.

Attention refines its existing immutable read-map union into complete current,
complete page-aligned historical, and mixed/tail intervals. A complete affine
current tile uses direct current-row addresses without per-vector page/position
selection. Historical lanes sharing a page use one owner lookup and retain V
addresses. Invalid-page publication, zero filling and ordinary tail behavior
remain intact. The page-owner width saturates before multiplication, including
very large legal power-of-two page sizes; the overflow regression does not
change generated CUDA for the measured page-size-eight ABI.

These are pure product/interval specialization followed by private CUDA
lowering. There is no Qwen-specific shape branch, arithmetic reassociation,
runtime JIT, new IR layer or added token-step validation. Physical resource
ownership and runtime side effects remain unchanged.

Implementation commits, pushed to `origin/parallel`:

- `92208812`: joint packed ingress binding and executable tail qualification.
- `354fbe4c`: complete immutable KV interval specialization.
- `906042f4`: overflow-safe page-owner width and its boundary regression.

Principal sources are
[source_fragment_head.mbt](../kernels/luna_cuda_projection_aot/source_fragment_head.mbt),
[source_ingress_projection.mbt](../kernels/luna_cuda_fused_parallel_aot/source_ingress_projection.mbt),
and [source_query_stage.mbt](../kernels/luna_cuda_attention_tile_source/source_query_stage.mbt).

## Hardware and comparison controls

Hardware is DGX Spark GB10, sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Compilation uses CUDA 13.0.88 and the
existing pinned MoonBit toolchain. GPU workloads run serially, with a 32 GiB
host-memory reserve. Counter containers have an 8 GiB no-swap limit; the serving
harness retains its 64 GiB limit. Minimum sampled serving MemAvailable is
104,347,272 KiB, approximately 99.51 GiB.

The two serving arms use the identical worker and launcher binaries, identical
launch geometry, and identical seven other AOT modules. Only full ingress and
readonly prefill modules change. Module-bound route measurements are regenerated
for each arm. Both retain the same diagnostic forced mixed dispatch; this is a
package/selection comparison, not a fixed-route production promotion.

The frozen attention cubin is sm120, while the new exporter binds sm121. That is
a real package difference, not a source-only intervention. A separate campaign
therefore recompiles the actual frozen source with the exact new sm121 flags.
Both comparisons are preserved below. Neither may be silently substituted for
the other. Geometry, mathematical inputs and ordered numerical law match.

## Ordinary isolated timing

Times are medians of five alternating ordinary CUDA-event samples, not profiler
replay times. Raw sample vectors are preserved. All pairs are bitwise equal;
attention additionally passes the independent scalar oracle. History denotes
preceding tokens, not launch capacity. The probe uses fragmented page IDs.

First, the actual frozen serving cubins versus the new modules:

| Family | Query tokens | Request rows | History | Mixed | Before µs | After µs | Less time |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| QKV ingress | 2048 | 1 | 0 | No | 482.771 | 426.452 | 11.67% |
| QKV ingress | 2048 | 1 | 2048 | No | 486.234 | 426.267 | 12.33% |
| QKV ingress | 2048 | 16 | 4096 | Yes | 518.109 | 471.158 | 9.06% |
| QKV ingress | 129 | 16 | 4096 | Yes | 73.221 | 69.652 | 4.87% |
| QKV ingress | 2048 | 16 | 0 | No | 521.988 | 487.523 | 6.60% |
| QKV ingress | 1 | 1 | 4096 | No | 26.556 | 26.513 | 0.16% |
| Prefill attention | 2048 | 1 | 0 | No | 316.614 | 285.403 | 9.86% |
| Prefill attention | 2048 | 1 | 2048 | No | 854.445 | 812.915 | 4.86% |
| Prefill attention | 2048 | 16 | 4096 | Yes | 1847.230 | 1841.330 | 0.32% |
| Prefill attention | 129 | 16 | 4096 | Yes | 1206.046 | 1182.077 | 1.99% |
| Prefill attention | 2048 | 16 | 0 | No | 78.871 | 78.523 | 0.44% |
| Prefill attention | 1 | 1 | 4096 | No | 151.413 | 161.440 | **−6.62%** |

The single-query prefill regression is retained, not hidden. Serving selects
the distinct decode family for pure decode; this table does not justify
selecting matrix prefill universally.

Source-only attention comparison, both compiled with identical sm121 flags:

| Query tokens | Request rows | History | Mixed | Before µs | After µs | Less time |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 2048 | 1 | 0 | No | 410.293 | 286.580 | 30.15% |
| 2048 | 1 | 2048 | No | 959.524 | 820.691 | 14.47% |
| 2048 | 16 | 4096 | Yes | 1861.372 | 1834.692 | 1.43% |
| 129 | 16 | 4096 | Yes | 1200.163 | 1185.648 | 1.21% |
| 2048 | 16 | 0 | No | 82.575 | 79.353 | 3.90% |
| 1 | 1 | 4096 | No | 153.305 | 143.351 | 6.49% |

The large complete-current benefit is not the benefit of replacing the actual
frozen serving package. Mixed-history sample ranges overlap substantially.
These results support reduced overhead in selected subdomains, not universal
latency improvement or a confidence-bounded whole-framework claim.

## Final selected-kernel counters

Nsight Compute profiles one old and one new invocation for query2048,
rows16/history4096, mixed. Counters use cold-cache replay and are not serving
wall time. Attention below uses the same-target pair; ingress uses the actual
frozen module. Mathematical instruction counts are unchanged.

| Metric | Ingress before | Ingress after | Attention before | Attention after |
| --- | ---: | ---: | ---: | ---: |
| Executed warp instructions | 69,568,704 | 62,970,048 | 209,757,504 | 191,119,808 |
| Registers/thread | 118 | 116 | 230 | 232 |
| Tensor activity, sustained elapsed peak | 24.61% | 27.75% | 30.72% | 32.68% |
| Long-scoreboard stalls/active issue | 3.35 | 2.76 | 2.72 | 2.86 |
| Barrier stalls/active issue | 0.38 | 0.51 | 1.53 | 1.58 |
| Short-scoreboard stalls/active issue | 1.54 | 2.43 | 0.42 | 0.44 |
| Source-correlated excessive shared wavefronts | 24,903,680 | 24,903,680 | 57,344 | 57,344 |
| Profiled invocation µs | 577.696 | 515.296 | 2006.656 | 1883.872 |

Ingress removes 6,598,656 supporting instructions, including 6,291,456 generic
`LD` instructions, while adding 1,572,864 `LDSM`. It retains 4,194,304 `HMMA`
and 143,360 `BAR`. Joint binding reduces register-copy cost relative to the
first experimental per-MMA binding, but register-source `MOV` remains higher
than the original: 4,930,272 → 8,862,432. It is not a zero-copy result.

Same-target attention removes 18,637,696 instructions (8.89%), including
4,639,440 register-source `MOV`, 1,608,128 `LDS`, and substantial predicates,
branches and reconvergence. `HMMA`, `LDSM`, async-copy and barrier instruction
counts remain unchanged. `SEL` increases by 3,014,656. The remaining load and
publication dependencies are not repaired merely by deleting address work.

Stall ratios have changing active-issue denominators. They must not be added
as wall-time percentages or read as absolute wait-duration changes. DRAM-byte
measurements were not included in this pair, so it is not a bandwidth roofline.
The same-target attention ptxas logs report zero stack/spill bytes; this is not
a substitute for arbitrary-shape runtime spill counters. Ingress runtime
`LDL`/`STL` instruction counts are zero in the selected capture.

## Fresh-start end-to-end serving

ABBA order supplies two fresh starts per arm, one measured trial per cell after
warmup. Literal token inputs, output count and output sequences match. Workers
drain successfully with empty runtime stderr. These small samples have no
confidence interval; the long-output C16 sample ranges overlap.

| Input / output / concurrency | Before samples ms | After samples ms | Before median ms | After median ms | Less time | Output tok/s before → after |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 4096 / 64 / 8 | 2559, 2580 | 2554, 2548 | 2569.5 | 2551.0 | 0.72% | 199.26 → 200.71 |
| 4096 / 256 / 8 | 7298, 7295 | 7305, 7203 | 7296.5 | 7254.0 | 0.58% | 280.68 → 282.33 |
| 4096 / 64 / 16 | 4739, 4714 | 4666, 4645 | 4726.5 | 4655.5 | 1.50% | 216.65 → 219.95 |
| 4096 / 256 / 16 | 12997, 12917 | 12916, 12850 | 12957.0 | 12883.0 | 0.57% | 316.12 → 317.94 |

Worker SHA-256 is
`e52e695b8320d4d29d8fce43611e1e6ac3932e31e530dd8cf11c3dbd629a8675`.
Changed ingress module SHA-256 is
`84756ff213e9066f219543650705271efb9b1073fbd92147ebcc5c32f734f829`;
changed readonly-prefill module is
`bfb8461b89becd441bfb9b8226ecc18a44fd6a5622b1ee905c5fa50209127647`.
The dual-score decode module remains
`462ebdcff2374a5ccbbb90a6d349601f8b1bce4ba28239319d283305e995aa79`.

## What actually ran, and why the gain is small

A separate diagnostic launcher enables Nsight injection while preserving the
exact worker and modules. The measured-only 4096/64/C16 trace window is
4629.512 ms. All 21,908 observed kernel calls are graph nodes. The repaired
ingress launches with 116 registers and the repaired c322 prefill with 232,
matching qualified binaries; full ingress row64 ownership and the original
launch geometries are retained. The wide and partitioned attention alternatives
are unchanged, not silently described as new implementations.

The additive activity budget, with overlaps retained, is:

| Exclusive activity in measured window | ms |
| --- | ---: |
| Attention, all phases including merge | 2650.969 |
| Decoder projections and post-ops | 1641.480 |
| Head/sampling | 118.061 |
| Norm/embedding | 101.129 |
| No observed GPU activity | 116.542 |
| Transfer | 0.337 |
| Cross-chain overlap | 0.993 |

Within non-additive kernel sums, partitioned decode partial alone is
2016.653 ms; readonly prefill is 567.922 ms; ingress is 518.408 ms. Long-row
ingress is 412.377 ms of that ingress total, while decode-size ingress is
103.939 ms. Thus the faster large-row ingress is only about nine percent of
this profiled window, and tiny-row ingress is effectively unchanged. The
prefill repair has its smallest gain when many historical request rows are
present. The dominant partitioned decode fold is outside these two source
repairs.

This is an activity accounting, not a dependency critical path or an exact
unprofiled attribution of the wall-time gain. It nevertheless rules out
"the new kernels were never selected" in this capture and explains why a
double-digit isolated ingress gain need not produce a double-digit serving gain.

The remaining priority is operand availability and publication cost in the
selected historical attention/decode fold, followed by the rest of the
projection/post-op chain. These data do not support claiming that one more
binding edit will close the historical 20–27%/26% reference gaps.

## Rejected work and validation

The first raw per-MMA ingress binding retained too much accumulator-copy work;
joint binding supersedes it. A joint attention PV consumer increased executed
instructions by 3,392,544 and regressed ordinary mixed timing. Its extra
address/selection/NOP cost exceeded the proposed fragment-sharing benefit.
That implementation and unused helper were removed before the source commits;
the failed/regressed campaign remains preserved.

Affected-package tests pass, including complete versus mixed/tail source
generation and the large-page-width boundary. The native warning-denied check
uses the existing toolchain migration suppression list
`-79-20-29-25-92-14`; it is not an all-warnings-enabled claim. The full working-tree
suite passed 4,281/4,281 before the additional overflow regression. The final
full-suite rerun passes **4,282/4,282**. `moon info`, whole-tree formatting,
and diff whitespace checks pass. Each new standalone helper passes a
warning-denied native check; the timing summarizer's regression checks median
purity and rejects zero, negative, nonfinite, missing and duplicate timings.

Both selected kernels compile deterministically in two independent invocations.
All twelve paired cells pass bitwise checks. Independent ingress qualification
passes 164 numeric/KV/tail cases. Memcheck with explicit leak checking,
racecheck and synccheck pass for both selected families. Benchmark automation
is first-party MoonBit `.mbtx`, not production Python or runtime JIT.

Primary final root:
`/home/wlc004s/lunaflux-chain-final-20261003.ozExcK9G`.
It includes final selected sources/cubins, paired raw timing, ordinary ABBA
serving, same-target controls, final NCU reports, independent fixtures, and the
measured-window Nsight trace. Earlier experimental roots remain preserved.
The compact four-campaign archive contains 11,907 files; its explicit exclusions
include source/build caches and deployment/model copies, not silently missing
profiler logs. The full original remote roots remain in place. Archive SHA-256,
verified after downloading without overwrite, is
`5605c248e41a3720f8cab72643901b8ac4b04cc52f561016dcb9c01f2d852498`.

## Follow-up: load readiness remains unresolved

The terminal instruction repairs above did not fix attention's asynchronous
copy completion and workgroup-publication dependencies. The latest mixed
prefill result remains almost flat; the selected partitioned decode was not
changed by those repairs. It is incorrect to call the load/barrier bottleneck
solved on the strength of fewer address or register-copy instructions.

An independent K/V single-slot experiment tested a concrete alternative,
without modifying production source or selecting new serving modules. It
retains candidate c454's dual-score numerical law, KV32, two query heads per
KV head, eight partitions, capacity grid 32×8×8, ordered fold and merge. Both
arms are compiled for sm121 with identical strict arithmetic flags and a
128-register cap. Current K acquisition publishes the previous V readers,
then issues current V; current V acquisition publishes completed K readers,
then issues next K. K copies overlap PV; V copies overlap QK. The two
publications per tile and terminal reader handoff remain necessary.

| Resource | Paired stages | Independent single slots |
| --- | ---: | ---: |
| Dynamic shared bytes | 33,040 | 16,656 |
| Registers/thread | 92 | 114 |
| Maximum resident blocks/SM | 2 | 5 |
| Compiler-reported local/spill bytes | 0 | 0 |

Ordinary CUDA-event medians of five alternating samples include the identical
partition merge. These are synthetic equal-history vectors using the traced
capacity grid, **not a fresh serving or reference-framework benchmark**.

| Rows | History | Before µs | After µs | Latency change |
| --- | ---: | ---: | ---: | ---: |
| 1 | 127 | 14.374 | 12.261 | −14.70% |
| 8 | 127 | 20.860 | 15.869 | −23.93% |
| 16 | 127 | 30.762 | 20.518 | −33.30% |
| 1 | 4096 | 57.035 | 62.624 | +9.80% |
| 8 | 4096 | 597.285 | 609.799 | +2.10% |
| 16 | 4096 | 1177.818 | 1192.216 | +1.22% |
| 1 | 8191 | 144.898 | 157.187 | +8.48% |
| 8 | 8191 | 1177.239 | 1207.367 | +2.56% |
| 16 | 8191 | 2320.761 | 2345.339 | +1.06% |

All nine vectors are bitwise equal to the same-law control and pass the
independent scalar oracle (maximum absolute error about 0.000244). Memcheck
with full leak checking, racecheck and synccheck pass. A short-history win is
not a long-context fix: this alternative is **not promoted into production**.

Fresh cold-cache Nsight Compute capture of one partial invocation, C16/4096:

| Metric | Before | After |
| --- | ---: | ---: |
| Executed warp instructions | 46,967,552 | 48,340,224 |
| Active warps, percentage of sustained active peak | 8.22% | 20.12% |
| Eligible warps/scheduler/active cycle | 0.09 | 0.10 |
| Average warp latency/issued instruction, cycles | 12.12 | 28.24 |
| Long-scoreboard stalls/active issue | 6.16 | 13.41 |
| Barrier stalls/active issue | 1.68 | 3.27 |
| Short-scoreboard stalls/active issue | 2.08 | 5.89 |
| MIO-throttle stalls/active issue | 0.14 | 2.49 |
| Source-correlated excessive shared wavefronts | 0 | 0 |
| Profiled partial invocation µs | 1216.608 | 1216.800 |

These ratios are not percentages of wall time. Increased residency creates
more stalled resident warps without materially increasing eligible warps;
it does not prove that absolute memory latency doubled. In the original,
the hottest sampled PC is `BAR.SYNC.DEFER_BLOCKING` immediately after
`DEPBAR.LE SB0, 0x1`. In the alternative, the hottest two sampled PCs are
adjacent to `DEPBAR.LE SB0, 0x0` and their subsequent publications. The sampled
`UMOV` is not itself a global load; disassembly identifies the preceding
copy-completion dependency. Both arms still stop on operand readiness.

This falsifies the proposed shortcut that halving staging storage and raising
residency alone closes the long-history gap. Future changes must demonstrate
reduced copy/address dependency latency or more useful work issued while
transfers are outstanding. Independent producer/consumer issue, future-page
offset reuse and alternative key-parallel ownership remain experiments, not
completed fixes or promised speedups. The same applies separately to matrix
prefill, which this decode-only experiment does not modify.

Runner: `benchmarks/gpu_pipeline/measure_split_operand_lifetime.mbtx`, with
`--summarize` and `--source-stalls` for bounded result inspection. Three native
runner regressions and the local affected source package's 67 tests pass.
GPU jobs were serialized, bounded to 8 GiB without swap and admitted with a
32 GiB available-memory reserve. Preliminary attempts (stale frozen digest
test, unused-variable compile error and an experimental validity-offset
error) are preserved and contribute no timing result. No production source,
module, model or container was changed.

Completed ordinary experiment:
`/home/wlc004s/lunaflux-independent-kv-v4-20261003.q3jdnqeK`.
Counters:
`/home/wlc004s/lunaflux-independent-kv-counters-20261003.YpDcIVtv`.
Downloaded archive (119 files, including preliminary failures):
`/private/tmp/lunaflux-independent-kv-20261003.mW7sZNMC/gap-repairs.tar.gz`.
Local SHA-256 matches the remote archive:
`fcf7f1c883991b8dbe91a1d1507f4b69a78c7422e23b100433c6038954735c18`.
Local download:
`/private/tmp/lunaflux-chain-results-20261003.WDsaMhp9/gap-repairs.tar.gz`.
