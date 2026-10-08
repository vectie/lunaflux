# Reference attention: from diagnostic substitution to AOT serving

## Objective and boundary

Integrate the reference attention experiment through the normal compiler,
artifact and ordered graph path. Keep model semantics and KV ownership
independent of vendor lowering. Do not enable an interception shim in the
production runtime or turn a kernel timing into a serving claim.

The working change is based on `e164eff4`. The Linux experiments use an
isolated HEAD archive plus an explicit overlay, excluding concurrent local
tensor-parallel and model-family development. No production deployment or
global default is changed. Vendor output/down GEMM remains a separate,
unfinished ordered-execution integration.

## Architecture

1. A named numerical law distinguishes BF16 probability reference attention
   from the existing F32 exponential-only alternative.
2. An immutable AOT module declares its output ownership as
   `PrefillRowsOnly` or `AllActiveRows` at startup.
3. The pure mixed-phase list rewrite preserves upstream KV write and
   downstream consumers. An all-row writer has no decode companion;
   a prefill-only writer has a disjoint companion.
4. CUDA lowering emits the pinned reference specialization. Its exact-size
   CSR tail resolver is bounded; original dependency headers are untouched.
5. Pure decode stays independent. Fresh decode measurements bind the exact
   new bundle scope; replacing prefill does not justify relabeling old timing
   records.

No extra request-path validation, JIT, allocation or host/device diagnostic
round trip is introduced. The first specialization is head-128 BF16 with
page size divisible by eight, not arbitrary dtype/shape support.

## First integration: preserved regression

Root: `/home/wlc004s/lunaflux-reference-integration-20261008.tGddCOzs`.
Qwen3-0.6B BF16 on Spark .179, same release/model/other kernels, normal rebuilt
worker. Four fresh-start ABBA arms, one warmup and three measured waves per
cell per start (six samples per arm).

| Input / output / concurrency | Control ms | Prefill-only reference ms | Change |
| --- | ---: | ---: | ---: |
| 128 / 256 / 1 | 1703.5 | 1734.0 | +1.79% |
| 4096 / 64 / 16 | 4520.0 | 4569.5 | +1.10% |
| 8192 / 64 / 8 | 4912.5 | 4996.0 | +1.70% |
| 32512 / 64 / 2 | 7506.0 | 8309.0 | +10.70% |

This was not promoted. A separate 8K/C8 Nsight trace excluded the warmup and
matched 96 measured steps in each arm. Mixed attention took 1330.47 ms in the
control versus 850.74 ms reference prefill + 541.99 ms compiled decode in the
integration. Other mixed kernels were essentially unchanged (1181.33 versus
1180.11 ms). The earlier winning diagnostic executed *all* mixed rows in one
reference call; this first integration did not reproduce that ownership.

Replacing the whole attention bundle also invalidated its measured route
table. The fallback split-decode policy only covered batch one, losing the
previous long-context batch-two choice. These are concrete propagation and
selection differences, not evidence of an intrinsically slower reference
kernel.

The preceding packaging failure is preserved separately at
`/home/wlc004s/lunaflux-reference-integration-20261008.gOfWMvlo` (macOS `._`
metadata entered the source overlay; no GPU test ran there).

## Revised integration and bounded retest

Root: `/home/wlc004s/lunaflux-reference-integration-20261008.G89e6nwn`.
Uses explicit all-active mixed ownership and new decode-route measurements.
Local targeted native tests passed 294/294 (293/293 for the physically tested
overlay; the additional test preserves the independent FlashInfer page-4
capability, without changing the measured FlashAttention specialization).
Existing toolchain deprecation
warnings use the repository experiment exclusions `-79-20-29-25-92-14`;
this is not an unfiltered warning-denied claim.

The revised AOT kernel passed the independent FP64 oracle and CUDA memcheck,
racecheck, initcheck and synccheck. The oracle covers exact-size shuffled CSR,
ragged phases, long-history tails, determinism, unchanged input/KV and
inactive output preservation. These are component-level numerical checks,
not a broad model-quality or production qualification.

The corrected paired retest used the same four fresh starts and six samples
per arm/cell. Throughput is aggregate output tokens per second; completion
change compares median wall time, with positive meaning slower.

| Input / output / concurrency | Control ms | Reference ms | Control tok/s | Reference tok/s | Completion change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 256 / 1 | 1702.5 | 1712.5 | 150.37 | 149.49 | +0.59% |
| 4096 / 64 / 16 | 4534.0 | 4491.5 | 225.85 | 227.99 | -0.94% |
| 8192 / 64 / 8 | 4910.5 | 4818.0 | 104.27 | 106.27 | -1.88% |
| 32512 / 64 / 2 | 7516.5 | 7661.5 | 17.03 | 16.71 | +1.93% |

The deployment integration reproduces a modest 8K gain, not the combined
attention-plus-vendor-projection diagnostic's entire gain. It is not a
universal win: short input is effectively close, and 32K remains worse in this
sample. Keep explicit opt-in, not a global default or framework-parity claim.
Fresh baseline/ordinary/split-decode calibration is supported. Automatic
workload-specific selection between this reference implementation and the
compiler implementation still needs a complete-chain comparison and route
integration; do not simply transfer the old prefill table.

The separate 8K trace verifies propagation:

| Measured GPU work (warmup excluded) | Control | Revised reference |
| --- | ---: | ---: |
| Prefill + mixed attention | 1447.61 ms / 2548 calls | 1325.72 ms / 924 calls |
| Other prefill + mixed kernels | 1344.41 ms | 1360.31 ms |
| Pure decode attention | 1602.36 ms / 2212 calls | 1598.74 ms / 2212 calls |
| Other pure decode kernels | 430.38 ms | 428.92 ms |

Both traces exclude 96 warmup graph launches and retain 96 measured launches.
The all-row reference symbol does not distinguish pure prefill from mixed
requests; its report therefore labels that group `prefill_or_mixed` instead
of inventing a per-phase split from a kernel name. The earlier compiled
decode companion is absent on those steps. The independent pure-decode
path and call count are retained. This is a systems timeline, not a fresh
Nsight Compute hardware-counter comparison.

A final 32K/C2 diagnostic traces both warmup and measured waves (190 graph
launches in each arm; not presented as warmup-excluded steady-state timing).
Prefill/mixed attention increases from 7525.94 to 7886.23 ms, while other
prefill/mixed kernels remain 2578.05 versus 2578.45 ms. Pure decode attention
is 3601.02 versus 3608.12 ms with 7056 calls in both arms. The remaining
32K loss is in the selected reference prefill/mixed attention schedule, not
another lost decode route or additional projection work. This timeline does
not yet identify its instruction/stall-level cause.

The explicit memcheck leak check reports zero leaked bytes/allocations.
Failed harness attempts remain preserved: calibration initially indexed the
shared-memory argument instead of the module path; timing initially reused
stale systemd unit names. Corrected controllers and timing live under separate
names (`reference_serving_integration_v2/v3.mbtx`, `timing-retry`). Neither
failure is counted as a timing sample.

The first remote `moon info` failed because the frozen tools lost executable
permission on `mooninfo`/`moonfmt`. Validation uses a disposable byte-identical
toolchain copy with only those permission bits repaired, not a changed
benchmark compiler. Interface generation, formatting check and native check
passed; the isolated Linux full native suite passed **3442/3442**.

## Commits and preserved evidence

The implementation is committed as `de57c072` (`feat(attention): integrate
AOT reference mixed-row ownership`). The physical source archive uses the
base and explicit overlay described above; unrelated working-tree changes
are excluded.

Both evidence archives were downloaded without overwrite to
`/tmp/lunaflux-reference-aot-verified-20261008.f4ESQGNf`.
Their local SHA-256 hashes match the remote archives:

| Archive | SHA-256 | Verified internal files |
| --- | --- | ---: |
| `prefill-only.tar.gz` | `b764736cd276266c8414f8408fefbda88d993e3315837a23d33dc91e58ce0262` | 2866 |
| `mixed-row.tar.gz` | `c70ed7a92c1e4c221d909cf172e579748b3021375bf7ec3c7b8d0d34765df6c1` | 3234 |

All **6100** internal `FILES.sha256` entries passed verification after
extraction into separate directories. The first archive is the compact
evidence copy: unchanged model payloads are excluded, with its parent archive
hash and correction note retained. The original remote archive is preserved.
Timing comparisons, raw Nsight traces, oracle/sanitizer logs, source identity
and failed integration attempts remain available; failed attempts are not
included in benchmark samples.

All physical runs are serialized under user-systemd runtime/memory limits,
with swap disabled and at least 32 GiB available-memory reserve. No fresh
vLLM/SGLang measurements are part of this integration comparison.
