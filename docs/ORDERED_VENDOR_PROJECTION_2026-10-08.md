# Ordered vendor projection experiment — 2026-10-08

## Question

The earlier launch-interception diagnostic combined reference attention and
vendor output/down GEMMs for a 4.48% completion-time reduction at 8192/64/C8.
The subsequent real serving integration included only attention (1.88%).
Those were different interventions, not evidence of a compiler ceiling.

This experiment integrates the missing projection alternative and measures
four arms with one rebuilt worker: existing AOT, reference attention only,
vendor projections only, and both. The existing attention numerical law and
frozen model, host runtime, projection artifacts and decode routes are retained.
No shim or launch-symbol interception participates in serving.

## Architecture and scope

The reusable bundle v14 carries an explicit `vendor_projection_exact_rows`
opt-in. Older bundle versions select zero (disabled). This is independent of
`supports_cublas_lt`, which remains false: the terminal implementation uses
cuBLAS GEMM, not the old cuBLASLt interface.

At startup the BF16 executor derives an immutable operand plan from the
admitted operation kinds and operand regions. Output projection and the down
companion of split gated MLP are eligible; QKV, gate/up, and vocabulary head
are not replaced. No model names, GPU geometries, or kernel symbol strings
decide this plan. Widths and argument indices come from the admitted bindings.

The normal graph planner makes captured companions for matching prefill
buckets, including mixed-attention and output-free prefix owners. After
attention/output-demand selection, an allocation-free check requires positive
prefill work and the exact live token count. A rounded capacity is not proof
of that count. Pure decode, partial buckets and unsupported companion routes
retain their original AOT executor. The experiment admits 2048 live rows only;
it is not an all-shape replacement or a new default.

The measured mixed split-decode owner (candidate 7) and deep-prefill-split
owner do not yet propagate this exact-row projection companion. Those owners
retain their AOT projections; this is a coverage limit, not a vendor-kernel
loss. Reference all-row attention uses a different eligible owner.

The device boundary receives typed BF16/F32 projection records. CUDA details
remain in the private native terminal lowering, which:

- Revalidates operand sizes, alignment and output non-aliasing before capture.
- Owns one private library handle and a fixed 32 MiB workspace per companion.
- Primes algorithms on bounded private zero operands, not model or KV state.
- Uses the existing stream, graph capture/replay and completion event.
- Requires successful capture; vendor calls never execute in the token loop.
- Destroys graphs before releasing vendor storage and existing operand leases.

This is an alternate backend implementation of a pure projection plan, not
another model-specific optimization pass. BF16 inputs/outputs use F32
accumulation with reduced-precision reductions disabled; bitwise equivalence
to the hand-lowered AOT kernel is not asserted.

## Validation and measurement contract

GPU: Spark `.179`, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`,
PCI `0000000F:01:00.0`. GPU workloads are serialized. Qualification uses a
32 GiB process cap; serving retains its 64 GiB cap, swap disabled and a
monitored 32 GiB MemAvailable reserve. No larger model/context envelope is
introduced.

The physical test covers 17x33x65, 128x128x256, 2048x1024x2048 and
2048x1024x3072 (M,N,K), three captured replays, an FP64 numerical oracle,
untouched output-tail bytes, and explicit release. The two large geometries
use deterministic strided oracle samples; this is not full-model quality
qualification. Memcheck, initcheck, racecheck and synccheck are required.

Timing order is control → attention → GEMM → both → both → GEMM → attention
→ control. Each cell has one warmup and three measured waves per start.
Cells are 128/256/C1, 4096/64/C16, 8192/64/C8 and 32512/64/C2. Raw request
vectors, TTFT, TPOT and completion time are retained. A separate Nsight run
checks actual vendor dispatch and preservation of decode. Instrumented timing
does not replace unprofiled completion measurements.

## State

Implementation and local targeted tests are complete. The isolated Linux
native suite passed **3450/3450**, including the actual exported-v14 bootstrap
regression. The component FP64 oracle, guarded tail, three replays, memcheck,
initcheck, racecheck and synccheck pass; memcheck reports **zero leaked device
bytes/allocations**. The existing C ownership fixture also passes GCC
ASan/UBSan on Linux (the stock gate requires absent system clang) and its
local sanitizer run. Warning-denied checks use the existing experiment
exclusions `-79-20-29-25-92-14`, not an unfiltered warning-clean claim.

The bounded four-arm experiment is complete. At 8192/64/C8 the combined
integration reduces median completion time by **4.81%**, versus **1.97%** for
attention alone. This reproduces the approximate diagnostic-scale gain in
the real captured serving path, not just in launch interception. It does
not establish all-shape improvement or production/model-quality admission.

The first archive attempt retained macOS AppleDouble sidecars and failed
before compilation. The first physical controller used a filename as a test
name filter and selected zero tests; sanitizer rejected that no-work run.
Both attempts are preserved and are **not** qualification evidence. The
corrected controller requires the physical success marker before proceeding.

Additional preserved harness failures: instrumenting `moon test` followed
compiler subprocesses after a toolchain cache invalidation; qualification now
instruments the already-built exact test executable. CUDA 13.0 synccheck's
automatic vendor-barrier tracking overflowed; a fresh run with
`--num-cuda-barriers 256` passes without suppressing synchronization checks.
The additional GEMM-only profile controller also had a parse error before
GPU admission; its corrected, checked copy ran under a new user unit. Both
controller copies and unit journals are preserved; no failed launch is timed.

End-to-end preparation found two stale metadata helpers and the worker's
bootstrap schema classifier still stopping at v13. All three now recognize
v14, with positive and future-version-negative packaging regressions plus
an actual exported-bundle bootstrap regression. The four final variants use
one rebuilt worker; earlier materializations are preserved, not timed. The
updated packaging scripts do not bypass inventory/digest verification.

The first unprofiled four-arm attempt stopped before the GEMM-only arm could
accept requests. A diagnostic-only worker recorded successful capture
reservations through owner 85, then `nodes=202 reserved=0 owner=84`.
Optional graph construction had consumed the structural capture allowance;
this was not a request-time kernel failure or evidence of a performance loss.
The failed attempt is excluded from timing summaries and retained.

The repair is a pure startup budget partition. It reserves complete requested
exact-row families (full projection, original/vendor effect-prefix and mixed
companions) before optional variants. Both partitions come from the same
64-graph/32768-node allowance; no capacity or asserted byte ceiling is raised.
Unspent conservative reservations are not reused after a capture failure.
Two regressions cover optional-graph starvation and fail-closed graph/node
limits. Diagnostic logging is absent from the rebuilt serving worker.

## Unprofiled end-to-end results

Qwen3-0.6B, BF16, Spark `.179`; deterministic synthetic varied token inputs,
prefix reuse disabled, fixed output lengths. Each entry is the median of
six measured waves across two fresh starts. Completion is the entire wave;
throughput is aggregate **output** tokens per second, including prefill time.

| Input/output/concurrency | Control ms | Attention ms | GEMM ms | Both ms | Both completion change | Control → both output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128/256/C1 | 1695.0 | 1711.0 | 1695.5 | 1709.0 | +0.83% | 151.03 → 149.80 |
| 4096/64/C16 | 4520.5 | 4467.0 | 4429.5 | 4309.0 | -4.68% | 226.52 → 237.64 |
| 8192/64/C8 | 4904.0 | 4807.5 | 4902.0 | 4668.0 | -4.81% | 104.40 → 109.68 |
| 32512/64/C2 | 7498.5 | 7669.0 | 7441.5 | 7521.0 | +0.30% | 17.07 → 17.02 |

Negative completion change is better. At 8K, a 4.81% time reduction corresponds
to a **5.06% output-throughput increase**; these are different denominators.
The 8K control range was 4871–4917 ms and combined range 4661–4679 ms.
Only two restarts per arm were measured: do not treat six waves as six
independent startup experiments or claim a population confidence interval.

| Input/output/concurrency | Control → both TTFT ms | Control → both TPOT ms |
| --- | ---: | ---: |
| 128/256/C1 | 15.50 → 16.00 | 6.50 → 6.56 |
| 4096/64/C16 | 1242.19 → 1134.16 | 47.31 → 45.71 |
| 8192/64/C8 | 1531.75 → 1421.31 | 48.87 → 47.05 |
| 32512/64/C2 | 3837.25 → 3845.75 | 54.94 → 55.11 |

TTFT/TPOT here are medians of per-wave request means, not medians of pooled
requests. The raw per-request token/time vectors are retained. Lower TPOT
in a mixed workload is not evidence that pure-decode kernels changed.

All arms use byte-identical request bodies and identical input/output-length
vectors. Output **sequences are not identical**: differing request sequences
versus the first control wave were control/attention/GEMM/both = 5/53/14/56
of 96 at 4K, 7/16/7/14 of 48 at 8K, and 0/12/0/12 of 12 at 32K. Short
sequences all matched. Even repeated controls differ, so this count is not
itself a numerical error measure. Component correctness/sanitizer tests pass;
end-to-end quality equivalence has not been established by this synthetic
performance test. There is no fresh vLLM/SGLang measurement in this campaign.

Decision: retain the explicit opt-in. Do not promote combined attention/GEMM
globally: short and 32K cells do not win. GEMM-only at 8K is effectively flat,
so the four arms cannot be described as two independent additive gains.

## Selected-path check (final worker, separate Nsight capture)

The control and combined captures each contain 192 graph launches; analysis
excludes 96 warmup launches and retains 96 measured steps. The combined path
executes **1786 vendor GEMM graph calls**, retaining 28 output and 28 down
calls on the non-eligible tail. The control has 921 output plus 921 down
prefill/mixed calls: the replacement preserves that 1842-call total.
Pure decode contains **zero vendor calls** and preserves 1764 output, 1764
down and 2212 attention calls. The newly exported source has actually reached
the captured serving path, without a launch-interception shim.

| Measured-wave GPU category | Control ms | Combined ms | Saved ms |
| --- | ---: | ---: | ---: |
| Prefill/mixed output + down | 375.208 | 237.029 | 138.179 |
| Prefill/mixed attention | 1425.850 | 1320.363 | 105.487 |
| Other prefill/mixed kernels | 951.655 | 981.717 | -30.062 |
| All pure-decode kernels | 1995.254 | 1993.977 | 1.277 |
| Summed kernel durations | 4747.967 | 4533.086 | 214.882 |

This is instrumented kernel activity, not unprofiled completion time or an
attribution of the unchanged-kernel slowdown to a particular hardware cause.
The latter still requires clocks/cache/counter controls if pursued. It does
show the missing projection saving is real and that the diagnostic's
oversized pure-decode GEMM substitution has not been carried into serving.
The earlier pre-budget-repair capture is also retained as `trace-comparison.json`;
the table above uses `trace-comparison-retry.json` from the final worker.

### Why GEMM-only at 8K is flat

The additional `profile-gemm` capture resolves the route interaction:

| 8K measured-wave projection calls | Control | GEMM-only | Combined |
| --- | ---: | ---: | ---: |
| Vendor output/down | 0 | 218 | 1786 |
| AOT prefill/mixed output/down | 1842 | 1624 | 56 |
| Pure-decode vendor calls | 0 | 0 | 0 |

GEMM-only replaces the four pure-prefill steps, but the 29 mixed steps retain
812 output and 812 down AOT calls. The measured split-decode construction in
`engine/device_step/mixed_decode_split_prepare.mbt` does not pass
`projection_query_bound` into `prepare_mixed_attention_variant`, so it cannot
select a projection companion. The reference all-row attention route in the
combined arm does not have this exclusion. This is an explicitly retained
coverage limit of the experiment, not an inferred cache or math interaction.

The 218 replaced calls save approximately **17.10 ms** of pure-prefill kernel
time in the trace (44.819 → 27.721 ms), less than 0.4% of the unprofiled 8K
wave. This is consistent with an almost-flat GEMM-only end-to-end result;
it must not be presented as a full-coverage test of vendor GEMM. In the
combined arm almost all prefill/mixed projections are replaced, producing
the much larger measured saving. To support this projection alternative
independently of attention choice, the measured split/deep-prefill owners
would need explicit companion construction and budget/dispatch tests.

Thus the 2% versus approximately 5% question is answered at two levels:
the previous real integration omitted projection replacement entirely;
within this integration, selected execution-owner coverage determines how
many eligible projections actually use it. Neither result proves a ceiling
caused by functional programming or the number of IR layers.

## Identity and reproducibility

Implementation commit: `95c4af21` (local branch `parallel`). Linux source was
HEAD `070547ab` plus preserved `overlay.tar`, `bootstrap-repair.tar` and
`budget-repair.tar`; the commit additionally cleans up a comment in
`attention_phase_executor.mbt`. Do not describe this as a byte-identical
checkout build of the final commit. The exact tested source archives and
worker binaries are retained independently.

- All four final workers: `1c5cfa3cb2538d70b5cea0fc59d93e2b8b3f0f94a9cc76a32e24e63e64854c87`.
- Exporter: `5bdea2fd7c9298725b256f0abc33e03fefc753a6964dc043b2cc23057678e0b7`.
- Implementation commit tar: `c87396b6b6a5ec982d6c8626d3fdfec2acce92363d14febc5a35be8e8c151577`.
- Remote root: `/home/wlc004s/lunaflux-vendor-integration-20261008.c0FiMcnC`.
- Final timing: `timing-retry/`; original failed `timing/` is not aggregated.
- Final control/combined profile: `profile-retry/`; GEMM coverage check: `profile-gemm/`.
- Downloaded evidence: `/tmp/lunaflux-vendor-verified-20261008.uuHITeCW/evidence`.
- Archive: `/tmp/lunaflux-vendor-verified-20261008.uuHITeCW/measurement.tar.gz`.
- Archive SHA-256: `fb004bb4c262e33b414f713ed2a6bea08223339270188ce9f8ff01823a2e473d`.

The downloaded archive SHA-256 matches the remote receipt; all **5967**
internal file hashes verify locally. The local verifier independently checks
all **648** measured request token-id and token-timestamp vector lengths
against their fixed output budgets. `VERIFIED.txt` and
`local-manifest-check.txt` sit beside the archive. Model weights, expanded
build/dependency directories and derived SQLite are excluded; serialized
AOT bundles, raw request vectors, Nsight reports, source archives and failed
attempt receipts are retained. All campaign GPU processes were stopped.

The report, trace analyzer, controller, startup diagnostic, archive sealer and
local verifier are MoonBit `.mbtx` tools under `benchmarks/gpu_pipeline/`.
No runtime Python/JIT/profile dependency or model-specific strategy branch
was introduced. Existing unrelated working-tree changes were not committed.
