# GLM cold/warm collective progress experiment

## Fixed question and scope

The exact-row candidate in `GLM_NCCL_SUBMISSION_2026-10-20.md` still has a
roughly 3 ms collective span after both ranks arrive. Measure cold and warm
replay before assigning this interval to network, proxy starvation or graph
preparation. This campaign allows at most two evidence-backed implementation
changes; it is not an overall benchmark.

Use the source-pinned `211791a1` executable, qualified scales/bulk/local-MoE
artifacts and exact 64-row collective graph. Two identical 7,973-input,
one-output requests run sequentially in one resident process. Require zero
cached prompt tokens for both. Startup and the short decode warmup are outside
the capture. Record both ranks; do not assume synchronized host clocks.

## External observation

The installed Nsight Systems does not directly expose NCCL proxy states.
An offline NCCL v5 profiler-interface observer records collective, proxy
operation and proxy-step events from installed NCCL 2.31.2. Its fixed event
pool, phase-gated callbacks and post-retirement TSV export have no production
dependency. Original ABI declarations come from NVIDIA NCCL v2.31.2-1.
CPU ASan/UBSan tests cover gating, parentage, concurrent state publication,
pool exhaustion and release on both Sparks. This is not yet a GPU timing
qualification or proof of negligible observer overhead.

NVTX point markers associate observer monotonic time with each local timeline.
Read NIC/RDMA counters before and after each request. Counter deltas cover
the whole diagnostic request, not individual 3 ms transfers. Proxy states
describe software progress; kernel spans and overlapping waits must not be
added as independent wire time. Inspect dropped events and actual loaded
profiler before attributing any result.

All comparison and tracing remain outside compiler legality. Any demonstrated
execution change belongs in immutable parallel/submission IR with narrow
device lowering. No payload hashing, altered numerical contracts or reference
substitution. Do not promote the opt-in graph policy from a cold-only result.

## Resource and service boundary

Stop only the owned GLM pair for GPU diagnostics. Keep 100 GiB per GPU and
16 GiB host reserve; diagnostic early stop is 17 GiB. Owned units have zero
swap. Export reports only after diagnostic GPU retirement. Restore the
qualified `eb152338` service and verify health and an actual answer afterward.
Prepared evidence roots: `/tmp/lunaflux-nccl-progress-20261020-v1` and
`/tmp/lunaflux-glm-http-nccl-coldwarm-20261020-v1`.

## First paired observation and observer limitation

Both literal requests completed with 7,973 prompt tokens, zero cached tokens,
one completion token and terminal `[DONE]`. The post-trace 17*19 request
returned 323 with `stop` (26 input/six output tokens).

The retained suspect ordinal 2 is 3.274912/3.183040 ms cold and
0.190336/0.153696 ms warm, on ranks 0/1 respectively. First-collective spans
are 9.823232/1.291680 ms cold and 6.474976/0.201376 ms warm; cold/warm does
not remove all rank-arrival cost. These are diagnostic GPU spans, not a
request throughput measurement or a new production improvement.

Rank 1's cold ordinal 2 has no overlapping compute kernel or profiler
allocation span. Its software GPU-data and receive-completion waiting states
overlap the whole 3.183040 ms kernel. The maximum individual send-completion
overlap is 0.152611 ms. Thus receive waiting cannot be interpreted as wire
transfer time. Local NVTX/monotonic marker-offset spread is 5,488 ns.
Observed per-request NIC error/retry/congestion counters did not increase.

The first observer stopped retention AFTER report flushing, exhausting its
65,536 records during cold phase (453,126 dropped on rank 0). It retained no
warm proxy records. Rank 0's offline exec bridge also omitted the NVTX
injection variable; its CUDA trace is present but local NVTX clock markers
are absent. These failures are retained, not treated as full proxy coverage.
The v2 observer stops retention before flushing, limits each phase to the
first 64 collective callbacks plus their children, adds thread-CPU timestamps,
and forwards the installed NVTX injection variable through the external bridge.

NCCL source distinguishes progress from queued-operation ingestion:
`src/proxy.cc::ncclProxyGetPostedOps` batches posted operations and
`allocateArgs` grows a pool of `NCCL_MAX_OPS` (2,048) entries on the progress
thread. A 64-row capture uses 32 channels while the short warmup does not
exercise this work distribution. This is a **candidate explanation**, not yet
proof that pool growth causes the observed stall. Observe ProxyCtrl append
intervals before attributing the missing progress to that code.

## Nondefault execution experiment

Prepare `CaptureWindowedCollectives(rows, maximum_sites)` in parallel IR:
one-row decode remains a single graph, exact prefill is partitioned at
existing joined collective boundaries, and other live counts use unchanged
interval replay. This bounds each graph's pending host proxy work without
padding a tensor or altering a reduction. The explicit horizon is supplied
by resource selection; no model name, observed time or NCCL pool constant
enters IR legality. Device graphs remain serialized on the owned stream,
with final completion retiring every window and retained effect/operand lease.
No intermediate host wait, request-path allocation, or new numerical kernel.

This is a falsifiable candidate, not yet a performance fix: it may reduce a
cold queue-ingestion burst, but extra graph boundaries may cost warm execution
time. First test a 16-site horizon, preserving the whole-frame alternative.
Measure cold and warm with the same observer before considering promotion.

## Proxy-control finding (unchanged candidate, v3)

Both phases now retain records without capacity overflow. ProxyCtrl observes
the actual `ncclProxyGetPostedOps` ingestion interval, not a guessed label
from GPU duration. Rank 1 cold has a 6.075124 ms append interval (1,088 posted
operations; 6.076400 ms thread CPU) overlapping 5.601138 ms of ordinal 1's
5.850144 ms GPU span. Another append takes 3.496624 ms. Rank 0 records
3.245221 and 4.406759 ms append intervals during its first collective.
The progress loop calls this ingestion synchronously on the same thread as
send/receive progress: it cannot progress those operations during this call.
The location of the long GPU span shifts with the timing of queue ingestion.

This demonstrates a cold proxy-ingestion burst, not OS descheduling or
millisecond wire transfer. Pool growth is a source-level contributor candidate
inside that call; individual allocator calls were not instrumented, so do not
assign the entire append duration to `calloc`. The CPU-timestamp observer
itself adds overhead. Compare the bounded candidate using the same observer.
The window alternative is still nondefault and unqualified for promotion.

## Windowed source and interrupted first trial

Source `6c10bf37` implements the explicit horizon through parallel IR,
prepared-window ownership, capture and HTTP policy. The committed ARM build
is `/tmp/lunaflux-glm-service-arm-windows-20261020-v1`; qualified CUBINs are
unchanged. Alternate exact/tail/decode execution, auxiliary forks, cancellation
and zero-allocation replay pass affected tests. The empty-cut test initially
expected construction to succeed, contrary to the existing nonempty contract;
`8c9bc575` corrects that test without changing executable behavior.

The first 16-site run, `/tmp/lunaflux-glm-http-nccl-window16-20261020-v1`,
retains both GPU/proxy traces, but a local disk-full error lost the warm HTTP
response and final status. It is **not a completed correctness qualification**.
Both exact diagnostic owners were stopped and their files collected; no
memory-limit waiver was made. Only regenerable local release/script build
caches were removed. Source, models, artifacts and evidence were preserved.

Its preserved cold trace is encouraging: rank 1's first four collectives
are 1.164864, 0.254208, 0.246816 and 0.227648 ms; maximum overlapping proxy
append is 0.438017 ms rather than 6.075124 ms. Four launch correlations in
the first 64 collectives prove actual 16-site partitioning. This is not yet
an accepted fix: repeat the complete fixed pair after the disk failure.
The full native campaign also encountered 30 disk-space fixture failures
plus the old test expectation (5,306/5,337); a fresh run is required, not a
tolerance waiver or a claim of a passing suite.
