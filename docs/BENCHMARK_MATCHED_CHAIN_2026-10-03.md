# Fresh matched workload benchmark on Spark

The remaining gap is reproducible with fresh reference measurements. At
4096 input tokens, 256 output tokens and C16, LunaFlux takes 13.356 seconds,
versus 11.843 for vLLM and 12.024 for SGLang. Completion time is therefore
12.8% and 11.1% higher. This is a benchmark tooling and diagnosis update,
not a new production kernel optimization or a claim that the gap is fixed.

## Runtime and experimental scope

The GPU is GB10, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
The model is the same Qwen3 0.6B BF16 model and explicit input token IDs.
Prefix caching is disabled in both references. Generation is greedy, ignores
EOS and completes the requested output count. Client tokenization is excluded.

The LunaFlux runtime is the immutable October 1 qualified serving overlay,
including the terminal reader handoff correction. It is **not** a deployment
of the unrelated dirty working tree. Its worker SHA-256 is
`97d6f5d884e48eab93abcb3ea7a3844be3ca7c7798987a908048ae17a4f86aed`;
launch SHA-256 is
`9f72414d12f3dd48dda201e1379c3b7353e0483146d3ce57bcd3e28cba1a0ece`.
The selected split partial entry is compiler c452, not the old per-key decoder.

The reference image identities are:

- vLLM: `sha256:73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`.
- SGLang: `sha256:3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`.

The ordinary timing campaign rotates engine order across three fresh starts:
vLLM/LunaFlux/SGLang, SGLang/vLLM/LunaFlux, LunaFlux/SGLang/vLLM.
Each cell has one warm-up and one measured trial per start. GPU workloads are
serialized. Reference memory fractions remain 0.5; counter-replay fractions
are not substituted into throughput results. Memory cgroups disallow additional
swap, and a 32 GiB host reserve is checked every 500 ms. Minimum MemAvailable
was 56,105,416 KiB for timing and 53,097,128 KiB for tracing, both above reserve.

Only C16 is freshly measured here. The token vector is 128/32, 4096/64 and
4096/256; no fresh C1 or C8 result is implied. Three trials report repeatability
and ranges, not statistical significance across machines or realistic prompts.

## Ordinary timing results

Values are medians of the three unprofiled trials. Throughput counts output
tokens only. Parentheses give minimum and maximum completion times in ms.

| Input and output | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | LunaFlux ms | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 and 32 | 1323.0 | 1561.0 | 1565.7 | 387 (384–391) | 328 (326–335) | 327 (326–328) |
| 4096 and 64 | 205.1 | 243.9 | 242.3 | 4993 (4967–5003) | 4198 (4188–4206) | 4226 (4215–4233) |
| 4096 and 256 | 306.7 | 345.9 | 340.7 | 13356 (13244–13393) | 11843 (11795–11883) | 12024 (12014–12056) |

| Input and output | LunaFlux TTFT ms | vLLM TTFT ms | SGLang TTFT ms | Extra completion time vs vLLM | Extra completion time vs SGLang |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 and 32 | 54 | 70 | 57 | 18.0% | 18.3% |
| 4096 and 64 | 1408 | 1069 | 966 | 18.9% | 18.1% |
| 4096 and 256 | 1416 | 1070 | 968 | 12.8% | 11.1% |

TTFT is the median over the 48 measured requests per engine and cell. It is
client-observed streaming latency, not an isolated prefill kernel duration.

## Output agreement

All 192 long-input paired sequence comparisons agree exactly. The fresh
profiled long-input sequences also agree for all 32 paired comparisons.

Seven of the 96 short-input comparisons differ. Each first divergence is at
zero-based token index 2, token 624 versus 382. SGLang differs on rows 0 and
12 in all three rounds; vLLM differs on row 0 in round zero and agrees in
the next two rounds. vLLM therefore also changes that sequence across fresh
starts. This does not establish which numerical implementation is wrong.
Logit margins and matched-operation accuracy tests are required before
assigning the difference to rounding, scheduling or a runtime defect. The
report does not turn this short-case difference into an accuracy pass.

## Additive activity accounting

Fresh Nsight Systems captures cover the same 4096/256 C16 request vector.
The diagnostic LunaFlux launcher retains the profiler environment; its worker
and selected cubins remain unchanged. The launcher is not deployable.
Profiler durations below are not substituted for ordinary timing above.

The interval sweep clips every kernel, memcpy and memset to the client window.
A segment with multiple different chains active is counted once as overlap.
Unknown names remain unmapped. Dense projections and their post-ops are
combined: a shared reference GEMM symbol does not identify output versus down.
Attention includes prefill, decode, mixed steps and merges. These are coarse
name-based whole-chain groups, not a dependency critical path or proof of
equal per-launch logical work.

| Exclusive activity or remainder | LunaFlux ms | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: |
| All attention phases and merges | 9595.829 | 8872.188 | 8943.994 |
| Decoder projections and post-ops | 2754.608 | 2203.123 | 2316.089 |
| Norm and embedding | 174.923 | 109.041 | 145.630 |
| Vocabulary head and sampling | 393.437 | 393.533 | 361.314 |
| Transfer | 1.248 | 2.304 | 1.836 |
| Unmapped kernels | 0 | 76.763 | 100.378 |
| Cross-chain overlap | 1.999 | 1.432 | 69.591 |
| No observed GPU activity | 406.401 | 216.664 | 41.173 |
| Entire client window | 13328.444 | 11875.047 | 11980.005 |

Unrounded budgets sum exactly to each window. GPU-inactive time includes window
edges and is not automatically CPU scheduling overhead. Host/API correlations
are needed to distinguish queueing, submission, completion and transport.

All 65,456 LunaFlux kernel calls have graph node identities. This capture does
not support graph fallback as the explanation for the gap. vLLM has 112,650
kernel calls and SGLang 109,073; fewer launches alone do not make us faster.

## What the fresh comparison establishes

Against vLLM, the exclusive differences are approximately 724 ms in attention,
551 ms in projections/post-ops and 190 ms without observed GPU activity.
Against SGLang they are 652, 439 and 365 ms. Unknown and overlapping reference
activity remains explicit; these differences must not be summed as causal
speedup predictions or applied directly to the unprofiled timing trial.

The selected LunaFlux split partial contributes 8441.141 ms over 7140 calls;
its merge contributes 43.139 ms. The selected ordinary blockwise decoder adds
451.753 ms over 532 calls, while matrix-prefill configurations add 662.110 ms.
SGLang's main decode kernel contributes 8621.264 ms over 7168 calls, with
364.789 ms in its ragged-prefill kernels and 24.169 ms in its state merge.
These invocation totals have different scheduling and mixed-work composition.
They are **not** matched single-operation microbenchmarks. Nevertheless, the
faster SGLang service cannot be explained solely by its main decode kernel
being faster: its aggregate main-decode duration is higher in this capture.

Thus the remaining gap is not proven to be one pipeline defect. The next
causal experiments must cover mixed/prefill attention, the complete projection
chains, and host gaps, alongside the selected split decoder. The head is not
the dominant remaining difference in this long-C16 capture.

## Implemented tooling and remaining causal work

Five offline MoonBit tools implement repeated matched timing, token and input
checks, generated-source preflight, serial trace collection, overlap-safe
accounting and trace comparison. Their pure accounting and validation functions
have regression tests for overlap, clipping, invalid times, warm-up rejection,
divergent outputs and unequal work vectors. All five warning-denied native
checks and all three native self-test entry points pass. The production runtime,
compiler semantics and token hot path receive no diagnostic code in this change.
The full dirty-tree production test suite was not rerun for these tooling-only
changes.

Still unresolved: live per-launch operand/history equivalence, instruction
dependency counter capture for those matched launches, and controlled kernel
ablations demonstrating an end-to-end benefit. The older synthetic selected
cubin counters do not become fresh live-serving counters through this report.
No bandwidth roofline, exact hardware stall cause, or performance closure is
claimed. This replaces the stale-baseline and partial-chain ambiguity with
fresh whole-service measurements; it does not replace causal testing.

## Archived results

Timing root: `/home/wlc004s/lunaflux-matched-chains-20261003-r2.I7I4ln0I`.
Trace root: `/home/wlc004s/lunaflux-matched-traces-r3-20261003.mWbdq63D`.
The latter references the completed LunaFlux capture under
`lunaflux-matched-traces-v2-20261003.1d3oHwLB/luna-full` without modifying it.
Earlier bridge-readiness and container-name harness failures are preserved
separately; they are not production failures or measured timing trials.

Archive: `/home/wlc004s/lunaflux-matched-archive-20261003.n0qj0eTj/matched-timing-traces.tar.gz`.
SHA-256: `6320de51a8a4fa11cbca1365814fecbe6296656ae5f3f953f53273659138c545`.
The archive includes raw requests, summaries, traces and command records;
duplicate source/build caches, deployment copies and toolchain installations
are excluded, not deleted. The prior qualified runtime archive remains intact.
The download at `/private/tmp/lunaflux-matched-20261003.uEpVBQHS/matched-timing-traces.tar.gz`
has the same verified SHA-256. The GPU returned idle and both completed campaign
units exited zero.
