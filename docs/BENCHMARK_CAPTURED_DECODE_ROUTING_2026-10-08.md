# Captured unsplit/split decode routing

This follow-up repairs the serving propagation and measurement gaps in the
[joint decode binding experiment](BENCHMARK_JOINT_DECODE_BINDING_2026-10-08.md).
It uses the existing immutable workload-route selector, not another compiler
layer or request-time tuning mechanism.

## Source corrections

- The materializer, kernel-root assembler and worker-bootstrap schema dispatch
  stopped at bundle v14. They now recognize v15, which carries the selected
  partition count. A regression exports a real two-partition v15 bundle and
  verifies that bootstrap dispatch and admission preserve that count. Unknown
  future schemas remain rejected by admission.
- Route rebinding now preserves the runtime exporter's trailing vendor-projection
  option. Previously, appending the route option after it broke suffix parsing.
- Decode calibration compares the ordinary entry and the complete partial/merge
  chain **from the same composed module**, each captured in its own CUDA graph.
  The probe uses the runtime row envelope, including inactive rows for C2/C4.
  The merge launch is included in candidate timing.
- A startup-dispatch regression verifies that measured unsplit winners override
  a broad split heuristic, while a winning split bucket reaches its prepared
  owner. No measurements are parsed, kernels compiled, or identities checked in
  the token step.

The compiler still provides immutable alternatives; offline effects produce
observations; pure selection constructs startup route tables; prepared execution
performs the GPU effects. Neither model semantics nor scheduler policy acquires
NVIDIA-specific branches.

## Device and experiment boundary

Spark .179 / GB10 sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`,
PCI `0000000F:01:00.0`. CUDA 13.0.88 and compute-sanitizer hashes are pinned to
the preceding experiment. The source is `2566e452` plus the seven-file repair
overlay; the serving worker and exporter are rebuilt from that isolated source.
Unrelated dirty-worktree changes are not in this snapshot.

The same selected c468/KV32/two-stage module supplies both alternatives, with
two partitions for the split chain. Ordinary and split reductions need not be
bitwise equal: all physical controls retain the explicit 0.003 sampled FP64
maximum-absolute-error bound and KV-immutability checks. These are not an
exhaustive model-quality test.

Calibration covers 42 homogeneous row/history combinations: rows 1/2/4 up to
32K context, rows 8 up to 16K, and rows 16 up to 8K. Each context ladder starts
at 128 and doubles. Aggregate diagnostic KV capacity stays at 131,072 tokens.
The existing selector requires at least 1% median improvement to replace its
baseline. This is a representative-bucket calibration, not exhaustive ragged
history or mixed-phase tuning.

## Independent captured controls

Five alternating pairs per row. History excludes the current token. Percentages
are medians of paired latency improvements; positive is faster.

| Rows / history | Unsplit median µs | Split-chain median µs | Paired improvement |
| --- | ---: | ---: | ---: |
| 8 / 127 | 10.235 | 13.161 | −28.43% |
| 8 / 8,191 | 1163.008 | 1178.519 | −0.91% |
| 1 / 32,767 | 1311.914 | 653.036 | +50.22% |
| 2 / 32,767 | 1434.525 | 1153.469 | +19.58% |
| 4 / 32,767 | 2410.433 | 2320.844 | +3.72% |
| 16 / 8,191 | 2324.706 | 2308.605 | −0.01% |

The C16 median-of-medians ratio looks slightly favorable, but the median paired
change is essentially zero and crosses both directions. Do not promote that as
a reliable improvement. Short C8 strongly favors unsplit. C1/C2 long-context
improvements persist in every pair.

## Serving verification

The A/B experiment uses identical newly built workers and the same modules.
Control binds the real baseline-only calibration records; routed binds both
alternatives' records. It does not fabricate measurements or reuse a route
scope from another module. Prefill, ingress, normalization and projection
artifacts are held constant. The serving package includes the existing opt-in
reference prefill and vendor projection paths; this is not an all-default
compiler-kernel performance claim.

Four fresh service sessions run in control/routed/routed/control order, each
with one warmup and three measured waves per workload. Inputs are deterministic
varied token vectors; output is exactly 64 greedy tokens per request with EOS
ignored. The three cells are 512/C8, 8,192/C8 and 32,512/C1. Numerical bounds and
token counts do not alone establish cross-framework output-quality parity.

Six measured waves per arm; median completion time, with aggregate generated
tokens divided by that median:

| Input / output / concurrency | Unsplit ms | Routed ms | Unsplit tok/s | Routed tok/s | Completion change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 64 / 8 | 707.0 | 711.5 | 724.19 | 719.61 | 0.64% slower |
| 8,192 / 64 / 8 | 4726.5 | 4687.0 | 108.33 | 109.24 | 0.84% faster |
| 32,512 / 64 / 1 | 5222.5 | 4077.5 | 12.25 | 15.70 | **21.92% faster** |

C8 differences are below 1% and should not be presented as a convincing gain.
For C1, the two control-session medians are 5209/5232 ms; routed are 4086/4068
ms. Median request TTFT is essentially unchanged (2428 versus 2431 ms), while
post-first-token mean interval falls from 44.00 to 25.84 ms. The complete-service
gain is 28.08% in output throughput, not the isolated chain's 50% latency gain.

All C1 output vectors are exactly equal between the paired arm samples. C8
output vectors are not all equal; changing batching/tail routes and permitted
reduction ordering is not a bitwise guarantee. Repetition within the same arm
also differs: short-C8 control matches 40/40 row comparisons to its first wave,
routed 37/40; long-C8 control matches 30/40, routed 28/40. C1 matches 5/5 for
each arm. Thus C8 variation is not exclusive to the new route; this does not
establish its exact numerical cause or excuse a future quality regression.
This campaign verifies fixed
token work and the stated kernel-level numerical bound, **not model-quality
equivalence for those C8 outputs**.

This is causal comparison with the **same module forced unsplit**, not a claim
to beat every previously measured package. The earlier report's best C1 result
was 16.35 tok/s with a different module/route configuration; this increment does
not supersede that best result or establish a new cross-framework lead. A fixed
two-partition realization is not a proof of the optimal partition count at C1.
No new vLLM/SGLang benchmark or production rollout is claimed.

## Actual dispatch, not just startup capture

Nsight node traces are filtered to the second request wave's wall-clock window,
excluding startup/capture and warmup. The table counts attention layers, not
requests. Qwen has 28 layers and each 64-output request wave has 63 pure-decode
steps; 28 × 63 = 1,764 layer invocations.

| Request window | Control unsplit calls | Routed unsplit calls | Routed partial + merge pairs |
| --- | ---: | ---: | ---: |
| 512/C8 | 1764 | 1736 | 28 |
| 8,192/C8 | 1764 | 1316 | 448 |
| 32,512/C1 | 1764 | 0 | 1764 |

Every partial launch has `gridZ=2`. C1 uses `(1,8,2)` partial and `(1,16,1)`
merge grids. C8 partial launches use row envelopes 8 or 1 as the active batch
shrinks; a grid of 8 does not imply eight active requests. The route is therefore
not globally split, nor inferred from the presence of unused captured graphs.

In the C1 request window, ordinary attention sums to 2349.623 ms. Routed partials
sum to 1194.411 ms and merges 4.273 ms, a 1150.939 ms difference. That independently
matches the scale of the 1145 ms unprofiled completion reduction. Profiled sums
are not substituted for unprofiled throughput and do not prove a new instruction-
or memory-counter explanation; this experiment isolates execution decomposition.

The initial profiler attempt preserved the normal closed child environment and
collected no worker CUDA events. That failed trace is retained. The successful
trace uses a separate current-source parent with only profiler-environment
inheritance changed, following the existing diagnostic method. The deployed
worker, bundle and AOT modules are unchanged. This temporary parent is explicitly
non-deployable; no production environment policy is weakened.

## Validation and resource safety

- Clean Linux `moon info`, formatting check, native check and full native tests:
  **3,474/3,474**, using the existing migration exclusions
  `-79-20-29-25-92-14`. This is not an unrestricted warning-clean claim.
- Local affected native packages: **254/254**. Materializer and assembler
  automation: **1/1 each**; route-calibration automation: **8/8**. Host probe
  geometry passes a warning-denied C++ build/run.
- Captured memcheck/leak, racecheck and synccheck pass at the bounded maximum
  contexts for rows 1, 2 and 16; captured initcheck additionally passes C1/32K.
  The unchanged selected module's deterministic compilation and earlier complete
  sanitizer record remain in the preceding report.
- GPU workloads are serialized. Probe units have an 8 GiB limit; build units
  16 GiB; serving 64 GiB and bridge 2 GiB, with no swap. A live guard requires
  32 GiB MemAvailable reserve and stops the owned trial on a breach. Completion
  requires drain acknowledgement, child reap, empty runtime stderr and an idle
  GPU. All four unprofiled sessions satisfy those checks; their minimum observed
  MemAvailable is **99.09 GiB**. No production service is changed.

Remote campaign:
`/home/wlc004s/lunaflux-decode-routing-20261008.bIfeYtZt`.
The archive preserves source/overlay inputs, scripts, generated modules, serving
manifests/binaries, raw paired measurements, sanitizer logs, unprofiled token
vectors, both profiler attempts and the actual-dispatch report. Model weights,
dependency caches and build/source checkouts are excluded; source archives and
artifact identities are retained.

Archive SHA-256:
`331608f4e73bff4ca555fc3b8d4069d8a0c6a55b309084c598be0a527337eb93`.
Downloaded without overwrite to
`/tmp/lunaflux-captured-decode-evidence-20261008.2rz4C14j/evidence-v1.tar.gz`.
The local archive hash and extracted `FILES.sha256` are verified. The final GPU
process query is empty.
