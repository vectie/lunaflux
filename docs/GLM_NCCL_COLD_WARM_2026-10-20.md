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

## Completed same-observer repeat

`/tmp/lunaflux-glm-http-nccl-window16-20261020-v2` completes both literal
requests: 7,973 input, zero cached, one output token and `[DONE]` each.
The subsequent request returns `323`, `stop`, 26 input/six output tokens.
The long requests exercise exact 64-row frames and the 37-row tail; the
smoke checks subsequent request retirement. This is not broad model quality.

| First-chunk measurement, ms | Whole graph cold R0/R1 | Window16 cold R0/R1 | Whole graph warm R0/R1 | Window16 warm R0/R1 |
| --- | --- | --- | --- | --- |
| First dense FFN reduction (ordinal 1) | 5.458784 / 5.850144 | 0.270880 / 0.200896 | 0.262720 / 0.244224 | 0.261824 / 0.227968 |
| Previously suspect ordinal 2 | 0.236256 / 0.282208 | 0.253536 / 0.227328 | 0.237920 / 0.224864 | 0.250272 / 0.229504 |
| First 64 collective-span sum | 31.330304 / 25.561184 | 25.633120 / 15.756512 | 21.203648 / 49.834496 | 23.896864 / 16.068512 |
| First-to-64th collective wall span | 211.950848 / 205.091776 | 205.588064 / 197.663040 | 236.593632 / 230.391200 | 204.496192 / 198.265952 |

These are diagnostic windows, **not request latency or throughput**. In
particular warm rank 0's collective sum increases; the shorter warm wall
span must not all be credited to communication. Its compute sum also changes
from 291.969184 to 252.631263 ms. Compute sums exceed wall span because shared
and routed work overlap. No sum across ranks is a wall-time metric.

Maximum cold append interval overlapping the observed NCCL nodes falls from
4.406759/6.075124 ms to 0.480193/0.264441 ms. Both repeats retain all first
64 collective records without event overflow. Their actual four graph-launch
correlations replace one for this prefix. The largest post-first collective
is now 1.193760/0.856000 ms cold and 0.924064/0.927936 ms warm; not every
collective has become a constant-time operation.

The remaining first collective is a **different cause**: rank 0 takes
8.178816 ms cold, with 7.780512--8.096512 ms of early arrival bounded from
both-rank causality. Rank 1 takes 0.235520 ms. This is not a surviving
8 ms network/proxy transfer. Offset-bound width is 0.316 ms; local NVTX
marker spreads are reported separately, never used to synchronize hosts.

Source and execution relationship:

```
Before: one whole-frame graph -> bulk publication of future proxy operations
        proxy thread: progress -> append burst (up to 6.08 ms) -> progress
        GPU ranks:    NCCL waiting during append -> consumer

After:  pure ReplayPolicy(maximum_sites=16)
        -> joined collective cuts -> owned graph windows -> ordered replay
        graph 0: sites 0..15 -> graph 1: sites 16..31 -> ...
        one terminal request fence; auxiliary forks/join and tail route retained
```

No NIC error, retry or congestion counter increments were observed in either
completed phase. These counter intervals cover the complete diagnostic
request, not individual collective wire time. On rank 1, cold RX/TX data
counters increase 2,113,183,493 / 2,113,269,069 four-byte units. There is real
RoCE traffic; absence of errors alone would not prove absence of queuing.

GPU maxima are 94.2422/94.2705 GiB; host minima 17.3842/17.7628 GiB.
All available owned-unit swap readings are zero. Three startup/retirement
readings are `[not set]`, not nonzero swap: retain the initial overly broad
boolean summary and use `envelope-network-summary-v2.json` plus raw logs.
Both GPU owners retire. After clearing regenerable caches, the full native
suite passes **5,337/5,337** with unchanged migration-warning exclusions.
Both ARM observer builds pass ASan/UBSan/leak tests, now including ProxyCtrl
append/count/CPU-timestamp assertions. The native graph capture/lease/cancel
ASan+UBSan gate and explicit ownership balances pass; CUDA kernels/ABI did
not change. GPU execution is separately evidenced by these HTTP traces.

The source-pinned 16-site option remains explicit/nondefault. A final trace
without the proxy plugin checks whether instrumentation created the apparent
remedy; no overall benchmark or new production policy is implied here.

## No-proxy-plugin control: measured stall removed

`/tmp/lunaflux-glm-http-nccl-window16-light-20261020-v1` runs the same
source-pinned option and fixed pair, with no `NCCL_PROFILER_PLUGIN` or custom
proxy callbacks. Both NCCL logs report that no default profiler plugin was
found. Nsight CUDA tracing remains: this is **not an unprofiled performance
campaign**. Both 7,973-token/zero-cache/one-output requests finish, followed
by `323` with `stop`. All diagnostic owners retire.

Only **59** collective nodes are common to all four traces; do not compare
their sums directly to the previous 64-node table.

| Same sites, GPU ms | Prior exact whole graph, cold R0/R1 (no proxy plugin) | Window16 cold R0/R1 | Window16 warm R0/R1 |
| --- | --- | --- | --- |
| First dense FFN, ordinal 1 | See prior retained graph | 0.208096 / 0.150624 | 0.212096 / 0.161472 |
| Original residual-stall site, ordinal 2 | 3.194368 / 3.148288 | **0.165408 / 0.173856** | **0.176608 / 0.176448** |
| Second dense FFN, ordinal 3 | See prior retained graph | 0.177984 / 0.161504 | 0.178048 / 0.178688 |
| First 59 collective-span sum | 23.928608 / 18.978048 | 20.843232 / 11.472768 | 19.904224 / 11.253248 |

The maximum after ordinal 0 is 0.725568/0.841056 ms cold and
0.680864/0.790816 ms warm. The old multi-ms post-arrival spike did not merely
move to another site in this observed prefix. This control corroborates the
remedy without custom proxy instrumentation; it does not measure individual
allocator costs or claim every possible context/concurrency is qualified.

The first collective still lasts 8.119488/0.174176 ms cold and
6.323040/0.171328 ms warm. Causal alignment bounds rank 0 early arrival at
7.808352--8.062432 ms cold and 6.010432--6.267424 ms warm. That separate
rank-entry skew remains; fixing proxy ingestion is not fixing every TP wait.
Four actual graph-launch correlations cover this 59-node prefix. Preserve
the first node in all sums; excluding it would conceal the remaining cost.

The retained vLLM rank-1 first two dense-FFN reductions are
0.239808/0.174592 ms, versus native 0.150624/0.161504 cold here and
0.161472/0.178688 warm. This is a **retained operator-level reference**, not
a fresh vLLM run or framework-wide victory. Its NCCL version and output-token
limit differ as documented in the preceding report. No unsafe new full-model
vLLM load or overall throughput benchmark was attempted.

The no-plugin control reaches GPU maxima 94.2422/94.2705 GiB and host minima
17.4593/18.1598 GiB. All available owned swap readings are zero; three
startup/retirement readings are unavailable. NIC error/retry/congestion
counters remain unchanged. Evidence includes `paired-{cold,warm}-59.json`,
`prefix-summary.json`, `envelope-network-summary-v2.json`, both SQLite
timelines, complete HTTP responses and raw unit/memory/NCCL logs.

Decision: retain the successful **explicit** windowed replay capability and
the whole-frame alternative. Do not automatically replace the qualified
production policy from this one trace workload: the earlier eager-serving
rank-0 59-node sum was 17.301760 ms, and a whole-service or broad-workload
promotion campaign was explicitly outside this task. Restore the qualified
serving configuration, not a silently selected experimental policy.

## Restoration and handoff

The qualified `eb152338` LunaFlux GLM configuration is restored through the
existing LunaNexa lifecycle control. Both user units are active, `/health`
returns `healthy`, and a fresh backend chat returns **323**, `stop`, 26 input
and six output tokens. Production remains decode-only collective capture;
the new exact-row/window policy is opt-in. No route/model/kernel promotion
or customer-key requalification is claimed.

Startup-through-answer observation in
`/tmp/lunaflux-glm-nccl-restoration-20261020-v2` records GPU maxima
**94.1904/94.2188 GiB**, host minima **17.8354/19.5084 GiB**, and zero in all
available owned swap readings. Startup unavailability is retained separately;
final active-unit swap peaks are both zero. The original response is
`/tmp/lunaflux-glm-nccl-restored-answer-20261020-v2.json.reply`.

Completed: the measured cold proxy-ingestion stall has a demonstrated,
generic IR remedy, with cold/warm and no-plugin corroboration. Remaining:
separate rank-entry skew, broader policy-promotion evidence, and a fresh
vLLM load that fits the unchanged host reserve. No hashing, memory-limit
waiver, overall throughput campaign or unproven production-default change.
