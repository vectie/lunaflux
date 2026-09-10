# Current-runtime source counters and Qwen comparison

## Scope and current state

This campaign uses commit `13369c5222f30a6903b7a4ce5b2b7131d0c3174c` and
Qwen3-0.6B BF16 on the RTX 5060 Ti. The finite source-counter matrix is complete:
**178 cases / 356 valid samples**, with no source-attributed shared excess.
The fresh three-engine timing campaign is also complete. Long-input C8 reaches
227.29 output tok/s versus vLLM's 446.51 and SGLang's 420.82. Short-input C8
regresses to 667.54 tok/s. **Zero source excess is not performance parity.**

The zero target is Nsight SourceCounters **shared excessive wavefronts**, split
between operand-copy instructions and other shared instructions. It is not the
aggregate hardware load/store arbitration counter. A launch with no executed
shared instructions is reported separately, not counted as positive evidence
for a conflict-free shared layout. Finite tested vectors do not prove every
shape, dtype, device, or unobserved dispatch path is conflict-free.

No production service or deployment is changed. GPU profiling and the three
servers run serially; normal serving timing is collected without the profiler.

## Full-runtime integration fixes

The clean committed runtime, rather than isolated exporter fragments, exposed
four defects corrected and pushed in separate commits:

| Commit | Fix |
| --- | --- |
| `437db46` | Delimit projection source fragments so a following `#include` cannot join a preceding closing brace. |
| `29cfd45` | Give single-token prefill its own startup owner; bucket-8 matrix grid sizing did not cover the scalar GEMV branch's complete output width. |
| `348f168` | Emit result-address maps only for lowering consumers that use them, removing unused helper declarations from strict gate/up compilation. |
| `13369c5` | Emit prefill warp-count and float-reload helpers from actual lowering demand, removing unused declarations from strict fused compilation. |

These are generic composition, startup planning, and CUDA lowering fixes, not
Qwen-name branches. The T1 owner uses the existing preallocated dispatch table;
no new token-step allocation or diagnostic work enters production execution.
The MoonBit agent/refactoring skills guided focused regressions and `.mbtx`
diagnostic orchestration.

Native interface generation, warning-denied check, and scoped formatting pass.
Affected projection and attention-source suites pass 47/47 and 28/28. The final
aggregate run passed 3,666/3,667: the unchanged zero-wait loopback read at
`service/online_tcp/control_server_wbtest.mbt:82` failed once. Its exact retry
passed 1/1 and the full TCP package passed 54/54 without a source change. This
does not relabel the original aggregate run as a full pass.

## Counter vectors

- Projection: QKV, output, gate/up, down, and vocabulary head at
  `(tokens, selected rows)` = `(1,1)`, `(7,1)`, `(8,1)`, `(8,8)`, `(17,8)`,
  `(32,32)`, `(65,8)`, `(504,8)`, `(1024,32)`; two samples each.
- Other selected families: residual/RMSNorm, RMSNorm, partial QKNorm/RoPE/KV
  write, segmented greedy partial/merge, and direct/split decode. Decode uses
  contexts 1/17/128/504/1024/1528, row counts 1/8/32, and mixed-row cases.
- Prefill: exact runtime CUBIN replay for base and two-way split partial/merge,
  including query tails and context 1528. These are forced topology coverage,
  not a claim that every topology is selected by the benchmark scheduler.
- Eight-way split prefill supplement: partial and merge at per-row query
  counts 1/7/32, eight rows, and context 8192. These replay the registered p8
  CUBIN with selection-compatible shapes and forced topology, not an observed
  automatic-dispatch trace. The timing matrix never selects this long-context
  branch, so this supplement does not require or imply a new timing run.

| Family | Completed cases | Valid samples | Positive shared cases | Shared-free cases |
| --- | ---: | ---: | ---: | ---: |
| Five projection roles | 45 | 90 | 38 | 7 |
| Pointwise, sampling, direct/split decode | 113 | 226 | 94 | 19 |
| Base and two-way split prefill | 14 | 28 | 11 | 3 |
| Eight-way split prefill supplement | 6 | 12 | 3 | 3 |
| Total | **178** | **356** | **146** | **32** |

All positive-shared samples have both operand-copy and other source excess
zero. The 32 shared-free cases are not added to positive layout coverage.
The base and supplemental logical summaries have no outstanding cases and no
measured nonzero source observation in the retained histories. The first selected campaign completed
58 cases before its parser rejected an omitted-column shared-free merge;
the corrected runner completed the remaining 55 unchanged cases. Original
reports and absent terminal records remain unchanged.

The selected-path inventory also finds embedding gather statically shared-free
in its installed SASS; it is not an extra physical counter case in the table.
All 56 residual-add/RMSNorm pairs are fused in this runtime. Stored standalone
residual, QKNorm, RoPE, original attention, full-ingress and compatibility
sampling artifacts are not additional selected paths in this configuration.
These observations limit the claim to this runtime and the listed vectors;
they do not certify every retained artifact or another model's dispatch.

Where Nsight omits both shared-wavefront columns for a statically shared-free
merge, the classifier additionally checks complete executed SASS, absence of
static shared/generic memory accesses, explicit non-shared address spaces,
zero N-way conflicts, and zero raw shared metrics. Missing columns for a
positive-shared kernel are still a failure, not a zero result.

Pointwise and decode probes include CPU reference checks. Prefill keeps its
sampled CPU oracle, all-output finite checks, and unchanged KV bytes. Projection
launch coverage uses finite/nonzero output checks and NaN sentinels; it is not
an independent full numerical oracle. Earlier arithmetic and sanitizer
campaigns remain separately scoped, not inflated by these counter samples.

The projection logical matrix is complete: **45 cases, 90 valid samples**,
all source excess zero. Seven cases are shared-free; 38 execute shared
instructions. The head's scalar branch depends on selected row count, not
token count: all three R1 head cases are shared-free. Earlier wrong classifier
expectations are preserved as diagnostic failures, not kernel failures or
original campaign passes.

The four unique standard projection CUBINs, compiled-set manifest, and execution
manifest are byte-identical between the r3 profiling runtime and final r4.
Their equality was checked before joining r3 non-head and r4 head results.
This reuse is limited to those checked artifacts; changed fused prefill kernels
require their own r4 replay.

The following current T1024/R32 samples illustrate the distinct hardware
metric. Both source categories are zero with positive shared execution, yet
the aggregate counters are nonzero. These are not controlled timing results.

| Kernel | Source copy / other excess | Hardware shared LD / ST totals, sample 0 |
| --- | ---: | ---: |
| QKV | 0 / 0 | 20,265 / 8,498 |
| Output | 0 / 0 | 4,811 / 2,489 |
| Gate/up | 0 / 0 | 69,713 / 542,662 |
| Down | 0 / 0 | 6,145 / 2,378 |
| Vocabulary head | 0 / 0 | 16,095 / 9,843 |

Labels are `projection-kK-t1024-r32-bound1024-s0`, K=0 through 4.
Sample 1 also has source excess zero and nonzero hardware totals. The earlier
[same-CUBIN residency experiment](SIBLING_RESIDENCY_2026-09-10.md) explains why
forcing the aggregate counter to zero can lose concurrency and worsen latency.

## Fresh matched timing results

Arithmetic mean of three measured trials, after one warmup at every coordinate.
Throughput is output tokens per second over the complete batch, including
prefill. All 324 measured requests have the required output counts. All kernels
and serving executables in the LunaFlux arm come from the clean committed
source; no legacy path was restored to improve the result.

| Input | Output | C | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | vLLM / LF | SGLang / LF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 | 256 | 1 | 247.58 | 277.06 | 265.41 | 1.12x | 1.07x |
| 59 | 256 | 8 | 667.54 | 1850.05 | 1769.10 | 2.77x | 2.65x |
| 128 | 128 | 1 | 240.45 | 269.29 | 259.14 | 1.12x | 1.08x |
| 128 | 128 | 8 | 652.51 | 1747.44 | 1648.96 | 2.68x | 2.53x |
| 512 | 64 | 1 | 208.47 | 248.72 | 239.71 | 1.19x | 1.15x |
| 512 | 64 | 8 | 509.28 | 1183.36 | 1133.58 | 2.32x | 2.23x |
| 1528 | 32 | 1 | 131.88 | 181.50 | 172.99 | 1.38x | 1.31x |
| 1528 | 32 | 8 | 227.29 | 446.51 | 420.82 | 1.96x | 1.85x |

Mean client first-token latency and request end-to-end latency, milliseconds:

| Input / output / C | LF TTFT | vLLM TTFT | SGLang TTFT | LF E2E | vLLM E2E | SGLang E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 / 256 / 1 | 15.33 | 21.67 | 30.00 | 1032.00 | 922.00 | 961.00 |
| 59 / 256 / 8 | 32.17 | 33.29 | 59.46 | 3058.21 | 1101.75 | 1149.75 |
| 128 / 128 / 1 | 18.33 | 25.33 | 26.67 | 531.00 | 474.00 | 492.33 |
| 128 / 128 / 8 | 47.63 | 49.04 | 63.00 | 1562.83 | 583.63 | 618.33 |
| 512 / 64 / 1 | 34.67 | 24.33 | 29.67 | 306.00 | 256.67 | 266.00 |
| 512 / 64 / 8 | 148.71 | 93.92 | 98.71 | 987.83 | 428.92 | 448.96 |
| 1528 / 32 / 1 | 93.67 | 54.00 | 59.33 | 242.00 | 176.00 | 184.33 |
| 1528 / 32 / 8 | 552.21 | 221.38 | 245.58 | 1105.63 | 558.13 | 606.00 |

The short-C8 first-token latency remains competitive while the post-first-token
spacing is 11.86 ms for LunaFlux versus 4.18/4.26 ms for vLLM/SGLang. This
places the observed short-C8 regression after first-token generation; it does
not by itself isolate a kernel or exclude host/graph overhead. Long-C8 spacing
is 17.79 versus 10.81/11.56 ms and may include interference from ongoing prefill.
These client millisecond timestamps are not exact per-kernel or ITL histograms.

**Sequence agreement is limited.** At 1528/32/C8, all 24 measured sequences per
engine match exactly. The other seven coordinates do not all have three-way
agreement, although some LunaFlux/baseline pairs match. Earliest differences
and every token ID are retained in `summary.json` and raw request records.
Thus the table is a descriptive fixed-token workload comparison, not a blanket
numerical-equivalence or model-quality claim. No statistically significant
ranking or saturation curve is established by three trials.

## Matched timing protocol

The fresh matrix is `(input tokens, output tokens)` = `(59,256)`,
`(128,128)`, `(512,64)`, `(1528,32)`, each at concurrency 1 and 8. Each engine
gets one warmup and three measured trials per coordinate. Report output
tokens/second, first-token latency, and end-to-end latency separately.

All engines use the same Qwen3-0.6B model files, BF16, greedy sampling, fixed
output counts with EOS ignored, and disabled prefix reuse. Baselines are the
installed vLLM 0.24.0 and SGLang 0.5.2 environments, not claims about the newest
upstream versions. Client SSE records and token IDs are retained. Differing
sequences make throughput observations descriptive fixed-workload timings,
not correctness-equivalent performance wins.

LunaFlux uses C32 capacity, a 1,024-token step/chunk ceiling, preplanned
prefill/decode owners and capture-with-eager-fallback policy. It is not forced
eager. The ingress configuration remains standalone QKV plus partial
QKNorm/RoPE/KV-write, matching the prior optimized comparison. Ordinary
configuration policy alone is not proof that every graph capture succeeds.

The native launcher recorded readiness and a 62,587-ms cold start, excluded from
timing. Its outer stderr is empty. Actual graph capture/replay/fallback counters
were not saved by this harness, so this report does not infer successful capture
from policy alone.

LunaFlux completed every planned trial with `measure.exit=0`. The orchestration
then exceeded its 60-second shutdown wait and stopped before starting vLLM.
A subsequent read-only check found the owned process group gone, both benchmark
listeners closed, and the GPU idle. This is retained as a harness cleanup
deadline failure, not an inference failure; the exact transient cause was not
captured. The baseline-only continuation preserved the completed LunaFlux
records and unchanged measurement client, completed both baselines, and left
the GPU idle. `RESUME.txt` and the final `RESULT.txt` retain the cleanup history.

## Small-batch regression: source-level findings

The complete new-path implementation is not uniformly faster. The freshly
measured LunaFlux short-C8 throughput is approximately 668 tok/s, versus the
historical approximately 1,423 tok/s in the prior matrix-pipeline campaign.
That historical comparison is not a same-run causal experiment.

A read-only comparison with the previous optimized runtime found unchanged
row-variant launch records and a byte-identical tuning record; variants were
not silently dropped and tuning was not ignored. The generated implementations
behind the same strategy labels changed:

- `rows8` MLP down formerly evaluated one 16-row matrix tile. It now evaluates
  four 16-row tiles before masking stores: four times the padded row-tile
  arithmetic, plus shared staging and synchronization.
- Gate/up's pipeline threshold changed from 256 to 2 tokens, bringing small
  batches onto the staged 64-row pipeline.
- The small-row vocabulary path formerly vector-staged input and weights with
  128-bit copies. Its generic matrix-map implementation now uses scalar BF16
  input staging and 32-bit global weight loads.

These are concrete work/transport changes and plausible regression mechanisms,
not a measured allocation of the serving-time gap. Old measured tuning
latencies cannot establish the performance of changed generated code. The
next optimization should specialize the generic schedule to the actual live
row extent, remove dead padded tile work, and recover vectorized transport;
reintroducing an unrelated legacy implementation is not required. Fresh
operation-level paired timing is needed before attributing the full regression
to any one of these changes.

## Artifacts

Final source archive SHA256:
`19e7944f45016eaabf9436a34d9b0c54dc2472fc9b079a4c7a8acf62281f3c9c`.

Remote final runtime root:
`/run/user/1000/lunaflux-13369c5-e2e-20260910-r4`.

Timing root:
`/run/user/1000/lunaflux-three-engine-13369c5-20260910-r1`.

The base combined archive contains 6,546 files (296,816,640 bytes), including raw
NCU reports, failed original classifier attempts, logical joins, probe sources,
current source archive/CUBINs, launch metadata, all three engines' streamed
records, and timing summaries. Model weights and build caches are excluded;
model identity is retained. Archive SHA256:
`02101dad887974472b083f53fd3f44cf4ad23b9dde3bd9ceea0a0392f98c6da6`.

Downloaded without replacing prior records to
`/private/tmp/lunaflux-source-counter-benchmark-20260910.HPNeO9/lunaflux-final-counter-benchmark-20260910-r1.tar`.
Local and remote archive size and SHA256 match.

The final p8 supplement contains 149 files (14,274,560 bytes), including its
six-case/twelve-sample result, raw reports, exact CUBINs, host probes and helpers.
The base archive and its 172-case summary remain unchanged; the independently
verified six-case summary raises total coverage to 178 cases / 356 samples.
Supplement SHA256:
`2cdc9b1758c7f6a5a929ea8120cc1b2bbc83a06e4282f7a52305a9425cb6d7ee`.

Downloaded without overwrite to
`/private/tmp/lunaflux-source-counter-benchmark-20260910.HPNeO9/lunaflux-p8-supplement-20260910-r1.tar`.
Local and remote size and SHA256 match. The test GPU is idle after the final
supplement; no benchmark server remains running.
