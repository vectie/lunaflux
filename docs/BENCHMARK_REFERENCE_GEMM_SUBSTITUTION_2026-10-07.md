# Reference-GEMM causal substitution: a real but partial serving gain

Replacing only the selected output/down GEMMs in a copied LunaFlux worker
reduced median completion time by **2.01%**, from 4936.5 to 4837.5 ms.
Their combined captured GPU activity fell by about **32%**. This supports a
real contribution from those implementations, **not** a claim that they explain
the whole framework gap. No production runtime or compiler pass was changed.

## Controlled question

Following the [corrected equal-work attribution](BENCHMARK_EXACT_MIXED_WORK_2026-10-07.md),
test whether replacing the two projection families produces an actual serving
gain, rather than inferring it from counters or standalone kernel timing.

- Spark .179, GB10 sm121, GPU `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`,
  PCI `0000000F:01:00.0`.
- Qwen3-0.6B BF16, eight 8192-token inputs and 64 output tokens/request.
- Frozen model, attention, scheduler, routes, bridge and production parent.
- Copied worker has a diagnostic-only CUDA driver dispatch shim. It replaces
  selected GEMMs with cuBLAS BF16 input/output, F32 accumulation, with reduced-
  precision reduction disallowed. This is **not** the exact vLLM implementation.
- Control loads the same shim and initializes the same vendor handle/workspace,
  but substitutes no kernels. Graph capture/initialization precede measured runs.
- One GPU workload at a time; serving capped at 64 GiB, controller/compiler at
  8 GiB, swap disabled, sampled MemAvailable reserve at least 32 GiB.

Only these exact base-symbol launches are eligible:

| Operation | Grid / block | Static / dynamic shared bytes | GEMM M,N,K |
| --- | --- | --- | --- |
| Output | 512 / 256 | 32768 / 8192 | 2048,1024,2048 |
| Down | 256 / 128 | 49152 / 0 | 2048,1024,3072 |

The 128-CTA output route and `_rows8`/`_rows16` variants pass through unchanged.
Vendor GEMMs compute the full 2048-row envelope, including padding; only live
rows are consumed. This bounded diagnostic is not a general shape-complete
backend or a proposed model-specific production shortcut.

## Unprofiled serving results

Order: control → output → down → both → both → down → output → control.
Each fresh start has one discarded warmup and three measured waves: six
measured waves per arm. Every request body matches across arms; all requests
return 64 tokens. Raw token IDs and per-token timestamps are retained.

| Substitution | Median wall ms | Output tok/s | Median TTFT ms | Median TPOT ms | Wall reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| None: control | 4936.5 | 103.72 | 1506.5 | 50.45 | — |
| Output only | 4886.5 | 104.78 | 1476.0 | 50.14 | 1.01% |
| Down only | 4896.0 | 104.58 | 1474.0 | 50.21 | 0.82% |
| Both | 4837.5 | 105.84 | 1435.5 | 49.67 | **2.01%** |

Fresh-start wall medians, in chronological order within each arm:

- Control: 4912 / 4955 ms.
- Output: 4873 / 4902 ms.
- Down: 4883 / 4920 ms.
- Both: 4817 / 4849 ms.

Both-versus-control reductions are 1.93% and 2.14% on the two sides of the
alternating order. This is a consistent direction in this bounded experiment,
not a broad confidence interval or evidence for every shape. Neither vLLM nor
SGLang was rerun here; historical baseline rates are not a fresh paired claim.

## Actual dispatch and GPU timing

Two additional Nsight captures use the same measured workers, with the existing
offline profiler-environment parent. Those runs are **not** throughput samples.
They contain warmup and one measured request wave, plus startup operations.

- Control output base, 512 CTAs: 1842 calls, 330.386 ms.
- Control down base, 256 CTAs: 2066 calls, 430.911 ms.
- Combined control activity: **761.297 ms**.
- Both arm: those two selected originals disappear. The captured replacement
  is `nvjet_sm121_tst_mma_128x192x64_2_32x96x64_tmaAB_bz_TNNN`:
  3908 graph calls, **517.870 ms**. Its two vendor startup calls add 0.240 ms
  and are excluded from that graph-call number.
- Replacement launch: 88 CTAs, 64 threads, 82944 dynamic shared bytes,
  255 registers/thread. These resource counts describe selection; they do not
  individually prove the reason for speedup.

Thus the implementations really changed, and their aggregate GPU activity
fell by **31.98%**. Dividing that improvement into a serving claim without the
unprofiled ablation would still be wrong. The measured whole-request gain is
99 ms, not 32%, and attention plus all other work remains.

This experiment does not identify instruction-level reasons for the vendor
win, or attribute the entire remaining gap to attention. It also does not test
an attention replacement. It narrows the causal claim to these two operations.

## Correctness, limitations and failures

The separate shadow arm runs both implementations on actual serving activations
but feeds the original result onward. It compared **7,504,216,064 live values**:
zero violations of the predeclared `abs(delta) <= 0.01 + 0.02*abs(original)`
diagnostic gate; 485 values differ bitwise, maximum absolute difference 0.03125.
This is not a bitwise compiler rewrite or release-level model quality admission.

Standalone full-size output/down graph capture and three replays checked
4,194,304 constant-input results exactly. Compute Sanitizer reports **zero
memory errors and zero leaked bytes**. All serving runs drained and closed
their child successfully, with empty runtime stderr and explicit GPU release.

Output-token repeatability remains unresolved even in control: 4/40 subsequent
control request vectors differ from the first same-row vector. Counts are
6/40 output, 12/40 down and 5/40 both; versus the first control vector, candidate
differences are 8/48, 7/48 and 9/48. These are mismatch counts, not model-quality
scores. The shim is **not promoted** on the basis of speed or the shadow gate.

Setup failures are preserved and excluded:

1. Initial shim compilation accessed a private BF16 field; corrected to public
   float bit conversion.
2. The approved worker closes stderr, so early diagnostic output was invisible.
   A separate exclusively created startup/teardown log now carries counters.
3. A reused service-unit name blocked one launch; experiment-specific names fix it.
4. The first dispatch guard confused total shared memory with launch-time dynamic
   shared memory. Positive substitution counters rejected those no-op runs.
   Exact Nsight static/dynamic fields corrected the guard before any timed result.

The AKO workflow's actual-dispatch and paired-serving checks are why the no-op
runs were rejected and the 32% operation gain is not reported as engine speedup.

## Evidence and decision

Remote root: `/home/wlc004s/lunaflux-reference-swap-20261007.68XbpHuQ`.
The archive includes 1813 checksummed files: source/scripts, copied worker and
shims, all request vectors, counters, Nsight reports/SQLite, sanitizer results,
external identities and the excluded setup attempts. Build caches and model
weights are omitted; the frozen source-copy command and external paths remain.

Archive SHA-256:
`73e250c296ecc9a75082b67b8db693fb08dd63e2d1f1fbbf00657e0217a5454c`.

Downloaded to `/tmp/lunaflux-reference-swap-verified-20261007.wiwviWTX/measurement.tar.gz`;
the local archive hash and all 1813 extracted file hashes verified. The unpacked
duplicate was removed afterward to recover local disk space; archive, manifest,
comparison JSON and remote evidence remain. Diagnostic worker SHA-256:
`0f4a9f729468f827c09917244e2a76907c8696aeeff0504add38ccf6722774af`.

Decision: retain this as an offline causal experiment. No production cuBLAS
fallback, capability change, model-specific branch, or multi-layer IR change.
These GEMMs explain a measurable **partial** gain, not the entire persistent
framework gap. Any next implementation must target a separately demonstrated
remaining cost and be judged by the same serving boundary.
