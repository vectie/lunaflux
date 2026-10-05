# Dual-Spark long-history schedule experiment, 2026-10-05

## Decision

Keep the frozen Q64/K64 approximate-exp2 baseline. Neither existing alternative
won on either Spark. No serving selection, numeric default, compiler pass or
production kernel was changed. These results concern one attention invocation,
not end-to-end token throughput or a new vLLM/SGLang comparison.

Both machines were used concurrently. Artifacts were exported/compiled once
on `.179`, then the identical cubins/probe were transferred to `.178`. Timed
GPU workloads were serialized per device. Afterwards `.179` ran counters while
`.178` ran sanitizers. This avoids duplicating compilation and overlapping
instrumentation with timing on the same GPU.

## Fixed experiment

- `.178`: GB10 sm121, UUID `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`.
- `.179`: GB10 sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
- Frozen exporter/source: `/home/wlc004s/lunaflux-ako-query-chunks-20261005.PCwRF4kW`.
- Baseline candidate `30322`: Q64/K64, split single-slot async pipeline, no
  retained query fragments; `approx-base2-f32-v1` explicitly enabled.
- `kv128`, candidate `32004`: Q64/K128, otherwise the same pipeline/law.
- `retain128`, candidate `70645`: Q64/K64, entire 128-dimensional query fragment
  retained across KV epochs; ordinary, not joint-right-consumer, lowering.
- Compiler: CUDA 13.0.88, sm121, O3, explicit no-FMA/precise-div/sqrt flags.
  Each candidate compiled twice to identical hashes, with no spills.
- Budget: two alternatives × three workloads × two machines, five alternating
  timing pairs per cell. All cells completed. Robust selection requires every
  paired gain to be at least 3%; no candidate approached this threshold.
- Q=2048 total active query tokens distributed across R=1 or 2 requests;
  H=historical tokens per request. Runtime bucket geometry is preserved:
  bucket tokens 2048, bucket rows 32, grid 63×16×1, block 128.
- User systemd timing limits: 16 GiB, swap disabled, 600 seconds, 64 tasks;
  32 GiB MemAvailable reserve. Recorded timing reserve never fell below
  121,798,128 KiB on `.179` or 122,294,844 KiB on `.178`.

## Unprofiled results

Times are median microseconds. The increase column is the median of paired
`candidate/baseline - 1`, not a ratio of separately aggregated medians.

| Host | Alternative | R / H | Baseline µs | Candidate µs | Paired time increase |
| --- | --- | --- | ---: | ---: | ---: |
| .178 | KV128 | 1 / 28672 | 7208.07 | 10136.28 | 40.21% |
| .178 | KV128 | 2 / 28672 | 7186.98 | 10531.21 | 46.76% |
| .178 | KV128 | 2 / 8192 | 2099.38 | 3015.01 | 43.01% |
| .179 | KV128 | 1 / 28672 | 7475.06 | 10345.29 | 38.40% |
| .179 | KV128 | 2 / 28672 | 7372.70 | 10757.48 | 46.40% |
| .179 | KV128 | 2 / 8192 | 2159.12 | 3050.32 | 40.48% |
| .178 | Retain query128 | 1 / 28672 | 7301.36 | 7560.72 | 3.55% |
| .178 | Retain query128 | 2 / 28672 | 7206.52 | 7426.89 | 3.84% |
| .178 | Retain query128 | 2 / 8192 | 2116.51 | 2167.41 | 2.63% |
| .179 | Retain query128 | 1 / 28672 | 7550.02 | 7851.59 | 2.81% |
| .179 | Retain query128 | 2 / 28672 | 7459.98 | 7696.36 | 3.36% |
| .179 | Retain query128 | 2 / 8192 | 2186.18 | 2233.24 | 2.15% |

## Counter diagnosis

Two matched baseline/candidate launches for Q2048/R2/H28672 were captured for
each alternative on `.179`. Profile timings are instrumented, not the timing
table above; stall percentages have changing denominators and are not an
additive decomposition of the end-to-end gap.

| Metric | Baseline | KV128 | Retain query128 |
| --- | ---: | ---: | ---: |
| Registers/thread | 234 | 255 | 254 |
| Dynamic shared bytes/CTA | 49,168 | 81,936 | 49,168 |
| Resident CTAs/SM (probe) | 2 | 1 | 2 |
| Active warp percent | ~15.88% | 8.33% | 15.88% |
| Executed warp instructions | 936,583,040 | 935,401,411 | 965,599,936 |
| `BAR.SYNC.DEFER_BLOCKING` | 2,808,832 | 1,408,000 | 2,808,832 |
| Non-transposed LDSM | 37,396,480 | 33,693,696 | 29,933,568 |
| HMMA | 119,668,736 | 119,799,808 | 119,668,736 |
| NOP | 25,242,624 | 32,757,760 | 41,136,128 |

KV128 **does** reduce loop/publication work, but increases shared storage past
the two-CTA residency boundary. Total instructions remain approximately equal,
active warps nearly halve and measured issue-active falls from 27.05% to
18.15%. This is a strong resource/latency-hiding explanation for its regression,
not a claim that larger tiles are intrinsically bad. The current storage plan
must change before this larger tile is a useful alternative on this GPU.

Query retention **does** reach the executable: non-transposed LDSM falls by
7,462,912 while HMMA, exp2 and async-copy counts remain unchanged. However,
register count rises by 20 and total warp instructions rise by 29,016,896
(3.10%). NOP increases by 15,893,504, LOP3 by 8,347,648 and IADD3 by 5,582,976.
Residency stays at two CTAs, so an occupancy drop cannot explain this case.
Long-lived operands change backend allocation/scheduling and supporting
instruction cost; the precise causal contribution of each increase needs a
bounded lifetime/consumer-window ablation, rather than another blanket hoist.

These are alternatives already expressed by the general immutable schedule
and operand-lifetime plans, not Qwen-specific implementation branches. The
missing win is in executable resource/scheduling economics, not another IR
layer. Next hypothesis: partial query retention/consumer windows may preserve
load reuse without the full lifetime cost. It has **not** been measured here.

## Correctness and retained measurements

All timed cells passed full-output comparison under the fixed 0.003 ceiling
and sampled FP64 oracle. Query retention was bitwise equal; KV128 was not,
as expected for changed softmax tile/fold boundaries. Long-history comparison
maxabs was 0.000488281; sampled oracle maxabs 0.000377474. This is not a
whole-model quality/parity result.

Both alternatives passed memcheck, racecheck and synccheck on irregular
Q129/R2/H128 on `.178`, with zero errors/hazards. No spills or local storage
were reported. Terminal checks confirmed both devices released their work.

Local retained root: `/tmp/lunaflux-ako-long-schedule-20261005.nEX7wb8a`.
Downloaded archives matched the remote hashes; all 79 `.178` and 2,028 `.179`
manifest entries verified locally.

- `.178`: `/home/wlc003s/lunaflux-ako-long-schedule-20261005.4B3ImvXz/experiment`;
  `measurement178.tar.gz` SHA-256
  `64f5a4708e0e461bf4d091922ef29fd6c3ca9beb479dbb41c553afb5eff68493`.
- `.179`: `/home/wlc004s/lunaflux-ako-long-schedule-20261005.3NFKcdZ8/experiment`;
  `measurement179.tar.gz` SHA-256
  `34ee7344622617c7f58e10c36430d8a156b93767cfa976a5431195821e92956e`.
- Baseline cubin: `66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`.
- KV128 cubin: `586d941e5dda2d15a4e0af106a82bcaa5f2c57c65cb6782841b36cc30330474b`.
- Retained-query cubin: `656ac792db1766e1352fe322a0510a7915f6930c1a00622103a0904304cc560b`.

Automation: `benchmarks/gpu_pipeline/ako_long_schedule.mbtx` and
`ako_long_schedule_report.mbtx`. Each passed warning-denied native check and
its regression test. No public package/API changed. Production selection is
unchanged; rejected kernel candidates were not sent through a serving rollout.
