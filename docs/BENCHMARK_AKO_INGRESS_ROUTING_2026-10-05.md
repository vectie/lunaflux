# AKO bounded ingress routing — 2026-10-05

## Result and decision

The small-query QKV alternative now propagates from the AOT bundle into the
existing startup execution buckets. It is an **explicit optional alternative**,
not a global replacement or a default schedule change. The larger prefill
module is retained because the preceding kernel experiment measured large
regressions when replacing it everywhere.

Fresh matched serving measurements show a modest benefit, not the earlier
15.6% kernel-only gain translated to whole-engine throughput. The clearest
cell is **512 input / 64 output / C8: 721 → 695 ms**, with identical complete
output vectors in all nine matched samples. Concurrent short cases with token
vector instability remain diagnostic; this is not a new model-quality or
production-promotion claim. No fresh vLLM or SGLang measurement was performed
in this bounded follow-up.

The `lunaflux-ako` skill guided the finite alternating-order measurement loop,
separate dispatch profiling, retention of losses and explicit stopping point.
The MoonBit guides kept control and automation in MoonBit; no request-path JIT,
benchmark dependency, filesystem check or profiler was added.

## Hypothesis and architecture

The [preceding cache-sensitive QKV experiment](BENCHMARK_AKO_SMALL_PROJECTION_2026-10-05.md)
found that a 16-row, transfer-width-64 schedule beat the frozen 64-row,
transfer-width-32 schedule on distinct-layer small batches, but lost on larger
prefill. Test one coherent change: preserve both immutable AOT modules and
use the existing bounded projection-variant mechanism at bucket preparation.

The chain is:

```text
existing compiler-generated numeric-equivalent AOT alternative
  → immutable bundle row-domain metadata (schema v13)
  → startup resource/ownership preparation
  → existing projection-variant launch resolution
  → existing execution/graph buckets
  → unchanged device submission path
```

Schema v13 carries one bounded alternative, its module identity, query tile
and launch dimensions. It does **not** implicitly permit approximate softmax:
strict bundles serialize `allow_approximate_prefill_exponential=0`. Variant
bytes participate in aggregate module limits and the route identity. Existing
schemas retain their behavior. The contract is currently closed to the cached
rotary ingress ABI; it is not advertised as supporting every fused family.

The runtime stores the alternative in existing prepared resources, with
deterministic release through existing module ownership. No new selector runs
in the token loop. Shape bounds are supplied at export, not model-name
branches: this experiment chooses maximum rows 64 and query tile 16.
Regression tests cover 0/1/2/8/16/17/32/64/65/128/129 rows and all 28 ingress
spans. C1 deliberately retains the baseline; rows 2–64 select the alternative;
larger prefill retains the baseline.

## Exact experiment

GB10 DGX Spark, `sm_121`, CUDA 13.0.88, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Supplied toolchain-policy digest
(not the nvcc executable SHA-256):
`c3e8741c84713f825ef038de8f80529fbeb55abe9e85066ec89cc7f9aa862857`.
Model: Qwen3-0.6B BF16, the frozen model/tokenizer and capacity contract.

Frozen baseline:
`/home/wlc004s/lunaflux-domain-serving-v4-20261004.QmE9WZ7s`.
Experiment:
`/home/wlc004s/lunaflux-ako-ingress-route-20261005.2Pa43nNl`.
Both arms use the same newly built worker:
`c28e9020df8f03b0b7430afba916c1aa86438e47ebdde05bcea34f898793f40c`.

The experiment source is the frozen baseline plus a scoped overlay, not the
entire unrelated dirty checkout. Final 20-file overlay SHA-256:
`363fc2a9c0580be4c6950dc93821fec01fb3524595a5d40417a21f06a68cabc0`.
The subsequently repaired kernel-assembly script is archived separately:
`e42ac686644a2cd59046c636d8c61690da2bf7338a277d4e57b3b0906ed62851`.

Both modules expose the same cached-rotary production symbol:
`lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3`.

| Property | Baseline | Small-query alternative |
| --- | --- | --- |
| Query tile rows | 64 | 16 |
| Transfer width | 32 | 64 |
| Pipeline stages | 2 | 2 |
| Threads / head ownership | 128 / one head | 128 / one head |
| Registers/thread | 116 | 75 |
| Source SHA-256 | `82dc995df98f2c1100a09e564e0c91dbfef0df2a00a8e71f6aa2d473291a3019` | `c8d0b7718d7bcbfedff43c20a1e145fe5eb42cfbafc71e38431cc16f86740c69` |

Alternative cubin SHA-256:
`8edd350599c39f91abe8863a51b8614b84a7bda9df33212b4dc1d6ee8f4ff0ca`.
Fresh attention route tables were generated for both bundle identities:
**all 203 winning bucket choices agree**, avoiding an attention-selection
change as an explanation of the A/B delta.

## Serving measurement

Six fresh serving starts: control, candidate, candidate, control, control,
candidate. Each start runs 128/32, 512/64 and 4096/64 at C1/C8/C16. Each cell
contains one excluded warmup and three measured request groups, giving nine
samples per arm. Fixed per-request input/token vectors, generation lengths,
worker bytes, attention routes and timing boundaries are matched. Profiling
and sanitizer runs are separate and excluded from these numbers.

Completion is wall time for the whole concurrent request group. Throughput is
requested output tokens divided by median group completion time, not decode
kernel throughput. Positive completion reduction means faster.

| Input/output | C | Control ms | Candidate ms | Completion reduction | Control → candidate output tok/s | Equal paired output vectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128/32 | 1 | 242 | 241 | +0.41% | 132.23 → 132.78 | 9/9 |
| 128/32 | 8 | 304 | 294 | +3.29% | 842.11 → 870.75 | 8/9 |
| 128/32 | 16 | 354 | 345 | +2.54% | 1,446.33 → 1,484.06 | 3/9 |
| 512/64 | 1 | 488 | 483 | +1.02% | 131.15 → 132.51 | 9/9 |
| 512/64 | 8 | 721 | 695 | +3.61% | 710.12 → 736.69 | 9/9 |
| 512/64 | 16 | 972 | 964 | +0.82% | 1,053.50 → 1,062.24 | 8/9 |
| 4096/64 | 1 | 693 | 696 | **−0.43%** | 92.35 → 91.95 | 9/9 |
| 4096/64 | 8 | 2,510 | 2,490 | +0.80% | 203.98 → 205.62 | 9/9 |
| 4096/64 | 16 | 4,577 | 4,552 | +0.55% | 223.73 → 224.96 | 9/9 |

| Input/output / C | Median per-request TTFT ms, control → candidate | Median inter-token interval ms |
| --- | ---: | ---: |
| 128/32 / 1 | 16 → 15 | 7 → 7 |
| 128/32 / 8 | 43 → 43 | 8 → 8 |
| 128/32 / 16 | 53 → 53 | 9 → 9 |
| 512/64 / 1 | 21 → 22 | 7 → 7 |
| 512/64 / 8 | 88.5 → 87.5 | 10 → 9 |
| 512/64 / 16 | 143.5 → 143.5 | 12 → 12 |
| 4096/64 / 1 | 122 → 121 | 9 → 9 |
| 4096/64 / 8 | 630.5 → 635 | 24 → 24 |
| 4096/64 / 16 | 1,169.5 → 1,174 | 41 → 41 |

Raw JSON retains each request's complete token IDs and token-time vector,
including all samples and startup ordering. The client uses millisecond timing;
small median shifts are not precise per-kernel effects or statistical proof.

### Numerical qualification

All requests returned the full requested output count. The unchanged control
is already unstable at 128/32 C8 and C16: respectively 7/9 and 2/9 complete
group vectors match its first sample. The candidate has 8/9 and 6/9. At
512/64 C16, control is 9/9 while candidate is 8/9. Every other cell is 9/9
within each arm. Consequently, concurrency can change output trajectories;
the paired mismatch is **not** evidence of bitwise serving equivalence or a
new independent quality oracle. Baseline-relative kernel equality from the
preceding experiment remains a separate, narrower result.

The deterministic 512/64 C8 cell has separated observed ranges: control
714–725 ms; candidate 688–705 ms. Single-request and long cells overlap or
move very little. No aggregate percentage is used to hide those differences.

## Propagation bugs, validation and physical checks

Real materialization found the assembler's supported-schema list lagged the
new bundle schema. The v13 acceptance and v14 rejection regression fixes that
seam; materializer and bootstrap classifiers also recognize v13. Earlier
failed preparation logs and the initial no-CUDA trace remain preserved, not
relabeled as successful qualification.

Local full native tests: **4,405/4,405**. Affected-package tests:
**292/292** locally and **290/290** on the frozen-overlay Linux source.
The native check uses the repository's existing migration warning exclusions
`-79-29-25-20-92-14`; this is not a claim that default warning-denied checks
are clean. Targeted formatting, generated-interface refresh and whitespace
checks passed. Assembler and serving-helper `.mbtx` regressions passed.

Production deliberately removes
profiler injection at worker exec: the initial trace captured no kernels.
The retry rebuilds **only an isolated parent launcher** retaining injection;
deployed worker and all AOT cubins stay unchanged. This launcher is never
deployed or used for timing.

The successful retry recorded **8,736 launches** of the small-query module
at grid 1 × 32, block 128, 75 registers/thread and **2,072 launches** of the
baseline module at grid 32 × 32, block 128, 116 registers/thread. Those
resource/geometry identities distinguish modules sharing the same symbol.
The trace covers C16 at all three input/output vectors and confirms actual
serving propagation. Its profiled aggregate times are not the A/B timing data.

**Serving memcheck/leak qualification remains blocked.** The diagnostic
sanitizer invocation terminated before its first instrumented CUDA API:
supervisor status 4, child exit `-6` (SIGABRT), no acknowledged drain. It
produced neither a zero-error summary nor a leak summary. The cause of this
sanitizer/launch integration failure is unresolved; it must not be called a
passed check, a GPU-kernel fault, or proof of production memory safety. Normal
serving and the separate dispatch trace drained successfully. The earlier
kernel-boundary memcheck/racecheck/synccheck results do not substitute for the
missing serving lifecycle gate. The route therefore stays explicit opt-in.

Implementation commit: `22fdbcdb` on `parallel`. Only the 21 scoped contract,
preparation, exporter, materializer and regression files were committed;
unrelated dirty work remained untouched.

## Reproduction and evidence

Owned offline drivers:
`benchmarks/gpu_pipeline/ako_ingress_route{,_serving,_report,_finish}.mbtx`.
The prepare/refresh/stage/routes/materialize phases preserve source overlays,
route tables and artifacts. Timing, trace and memcheck serialize GPU use.
User-systemd limits serving to 64 GiB, bridge to 2 GiB, no swap, bounded
runtime and a monitored 32-GiB MemAvailable reserve.

Completed timing and dispatch runs were sealed together with the diagnosed
sanitizer failure, explicitly labeled
`outcome=completed-with-validation-blocker`. The minimum sampled MemAvailable
over completed timing/dispatch runs was **103,110,628 KiB (98.3 GiB)**; the GPU
was idle at closure. No measured optimum or framework-wide parity is asserted.

Remote archive: experiment root `/measurement.tar.gz` (12 MiB).
SHA-256: `bcc9006f711f7d21a109ba66d9f7e346ac280770867ac06c42cd8b6949c3288f`.
Downloaded without overwrite to
`benchmarks/qwen3_comparison/results/ako-ingress-route-20261005.GdTFpgBg/measurement.tar.gz`;
its local hash matches. **All 7,795 inventoried files verified locally.** The
original inventory had remote absolute paths after its first line; a derived
`FILES.portable.sha256` preserves every digest and rewrites only that prefix.
The offline writer was fixed with a multiline regression. The original
inventory and archive remain unchanged.

`serving-comparison.json` retains every timing/token vector;
`trace-1-candidate/kernels.stdout` records dispatch;
`memcheck-0-candidate` preserves the unsuccessful sanitizer invocation.
The companion driver supports `--verify-downloaded LOCAL_ROOT REMOTE_ROOT`
to perform the portable inventory verification and retain its logs.
