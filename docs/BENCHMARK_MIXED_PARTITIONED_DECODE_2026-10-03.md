# Mixed-graph partitioned decode

Mixed attention now supports an ordered partition/merge decode companion.
On Spark, the controlled serving A/B reduced completion time by 2.95% for
4096/64/C16 and 0.81% for 4096/256/C16. The selected mixed decode chain became
82.2% faster, but this change does not close the whole-serving framework gap.

Runtime implementation: `bb3ad7be` (`feat(runtime): support measured partitioned
decode in mixed graphs`). This is a general execution-planning change, not a
Qwen-specific kernel or a change to model semantics.

## Implementation and selection

The pure startup transformation narrows prefill to its row domain, then inserts
either the ordinary decode writer or the ordered split partial/merge sequence.
Upstream KV writes and downstream consumers remain once in the original order.
Decode scratch and existing AOT functions are reused; no new KV arena is added.

Measured route 7 identifies the complete mixed graph. It is legal only for mixed
buckets, requires the companions, and consumes the existing capture-memory
budget. Missing companions or unsuccessful capture cannot silently substitute
an ordinary graph for the measured winner. Compatible buckets reuse a prepared
owner. Token steps use retained scalar owner indices: no tuning, filesystem
work, graph construction, or new host allocation was introduced in that path.

Unmeasured production buckets retain their previous policy. Route 7 requires
real per-bucket **whole-graph** measurements; isolated decode timings are not
those records. The A/B forces the treatment only in a disposable diagnostic
source copy, without manufacturing measurements or deploying it to production.
Both arms rebuild the runtime with the new implementation; AOT artifacts,
frozen model, workload and capacity configuration are held fixed.

## Ordinary end-to-end A/B

Hardware: DGX Spark GB10, sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Model: Qwen3-0.6B BF16.
CUDA compiler: 13.0.88. Toolchain: MoonBit 0.1.20260920.

Each route has two fresh server starts, one measured trial per cell after warmup,
in baseline/treatment/treatment/baseline order. Times below are client completion
times, not profiler replay times. All measured output token sequences match.

| Input / output / concurrency | Baseline samples ms | Partitioned samples ms | Baseline median ms | Partitioned median ms | Less time | Output tok/s, before → after |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 4096 / 64 / 16 | 4888, 4955 | 4767, 4786 | 4921.5 | 4776.5 | 2.95% | 208.07 → 214.38 |
| 4096 / 256 / 16 | 13311, 13256 | 13157, 13195 | 13283.5 | 13176.0 | 0.81% | 308.35 → 310.87 |

Two observations per route do not provide a confidence interval. These results
support a positive effect for the tested cells, not a universal speedup.
Minimum sampled MemAvailable was 104,952,672 KiB (100.09 GiB), above the 32 GiB
reserve. Runs were serialized and bounded by the existing 64 GiB service limit.

## The new path actually executed

The treatment serving trace contains these functions, not the ordinary decode
function:

| Selected function | Grid | Block | Registers/thread | Shared bytes/block |
| --- | --- | --- | ---: | ---: |
| `lunaflux_paged_attention_bf16_decode_split_partial_v3_ep_3903` | 32,8,8 | 64 | 80 | 33040 |
| `lunaflux_paged_attention_bf16_decode_split_merge_v3_ep_3904` | 32,16,1 | 64 | 36 | 48 |
| Unchanged compiler prefill | 63,16,1 | 128 | 229 | 49168 |

For the single decode row in the mixed vector q2047/p0 + q1/p4096, partitioning
exposes 8 KV heads × 8 partitions = 64 useful CTAs instead of 8. The launch
envelope remains capacity-sized; this is not a claim that all launched CTAs
perform useful work.

Across its 28-layer attention chain:

| Component | Previous capture ms | Partitioned capture ms |
| --- | ---: | ---: |
| Prefill | 9.226240 | 9.295872 |
| Decode, including merge | 18.112544 | 3.228192 |
| Complete attention | 27.338784 | 12.524064 |
| Projection and postops | 40.637056 | 40.836256 |

The new decode consists of 3.140000 ms partial activity and 0.088192 ms merge.
Its activity falls 82.2%; the complete attention chain falls 54.2%. For the
second mixed vector q2047/p2047 + q1/p4097, complete attention activity falls
42.463424 → 29.149280 ms.

These are sums of CUDA activity in separate profiled sessions, not an exclusive
wall-time attribution or a new instruction-level counter diagnosis. Nsight
Systems confirms selected execution and timing; no fresh Nsight Compute stall
counter comparison is claimed here.

## Why the whole-serving gain is smaller

The previous and new measured windows match **all 96 actual query/history
vectors and their occurrences**, not just padded launch dimensions:

- 66,544 query tokens: 65,536 in multi-query rows, 1,008 in one-query rows.
- 138,411,520 causal query-key pairs per query head.
- 44 pure C16 steps; zero orphan or unmapped kernel calls in the treatment.

Only 19 steps contain both one-query and multi-query rows. Fourteen are
multi-query-only and 63 are one-query-only. Query length alone does not prove
that a row is semantically decode.

| CUDA activity budget | Previous ms | Partitioned ms |
| --- | ---: | ---: |
| Mixed-step attention | 709.108 | 551.141 |
| One-query-only attention | 1858.315 | 1858.021 |
| Multi-query-only attention | 338.981 | 343.242 |
| All projection and postops | 1679.756 | 1677.459 |

The mixed attention activity savings are about 158 ms, consistent in scale with
the ordinary 64-output median improvement of 145 ms. This is not an additive
causal decomposition of client wall time. Most attention activity and projection
activity are unchanged. Dilution by unchanged work is a plausible explanation
for the smaller 256-output gain; that longer cell did not receive this detailed
activity capture.

## Reference scope and remaining work

No fresh vLLM or SGLang end-to-end campaign was run in this change. Against the
retained exact-work vLLM trace, first mixed attention is now 12.524 ms versus
10.450 ms, while projection/postops remain 40.836 ms versus 32.423 ms. The second
mixed attention vector is 29.149 ms versus 22.967 ms. These retained references
show remaining optimization targets; they are not a fresh three-way serving
speed claim. See the [equal-work report](BENCHMARK_EQUAL_WORK_PROOF_2026-10-03.md)
for reference work and attribution limits.

The next performance work is in the unchanged one-query attention and projection
chains, plus real whole-graph mixed-bucket calibration. More IR layers alone
would not reduce those execution times.

## Validation and reproducibility

- Warning-denied native check and affected formatting checks passed.
- Full local native suite: 4,270/4,270.
- Spark focused package suite: 204/204 on the frozen campaign source.
- Physical eager/captured tests: 2/2, covering row ownership, sequence order,
  measured-slot reuse, incomplete companion rejection, budget failure and cleanup.
- Compute Sanitizer memcheck, racecheck and synccheck passed with zero errors.
- Measured model output sequences match between the two arms. The synthetic
  physical ownership fixture is not independent model numerical qualification.

Campaign directory:
`/home/wlc004s/lunaflux-mixed-decode-20261003.O8Li8ugm`.
Local summaries: `/private/tmp/lunaflux-mixed-decode-20261003.Pt3viIng`.
Archive: `evidence.tar.gz`, SHA-256
`1a9a20e70e1e627f2a4cfcbc8cc517f6d9ab679d88ff77317477957641aaba9b`.
Remote archive directory:
`/home/wlc004s/lunaflux-mixed-decode-archive-20261003.Xt2TtJHn`.
The archive retains ordinary timing, request tokens, memory samples, trace,
qualification logs, diagnostic overlays and failed preliminary attempts;
disposable source/build copies are excluded. Failed attempts remain failures.

Reproduction tools are MoonBit `.mbtx` programs under `benchmarks/gpu_pipeline`:
`prepare_mixed_decode_serving`, `run_mixed_decode_ab`, `summarize_mixed_decode_ab`,
`qualify_mixed_decode`, `trace_all_new_runtime`, `compare_equal_work --luna-ab`,
and `summarize_mixed_activity`. Their output destinations are non-overwriting.
