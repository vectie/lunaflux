# NCCL submission, measurement overhead and exact-row replay

Scope: the latest scales + bulk + locally combined MoE + BF16 projection-TP
HTTP path, both Spark ranks, one literal 7,973-input/one-output-token request.
This is an execution trace, **not an overall benchmark**. No payload hashing.
Limits remain 100 GiB per Spark and 16 GiB host reserve. GLM is temporarily
stopped for the experiments and must be restored after GPU owners retire.

## Why the old 3.4 ms is not established wire time

The retained paired capture is
`/tmp/lunaflux-glm-http-dense-pair-20261020-v6`.
At ordinal 8 (recurrent layer 4 output), NCCL spans 3.442528/3.379264 ms.
Both launches were submitted approximately 17 ms earlier; causal rank-start
bounds exclude more than 0.19 ms of arrival skew. But rank 0 also has a
**5.538768 ms CUDA profiling memory allocation** at 212730640--218269408 ns,
overlapping 3.160704 ms of that NCCL span. Across the capture it records 90
profiler memory allocations (178.271 ms total) and 90 TLS allocations.
These overlapping CPU measurements are not additive GPU costs.

Native `internal/nccl/communicator.c` explicitly sets nonblocking submission.
NCCL 2.31.2 `src/group.cc:1019--1049` launches a fresh async host thread per
nonblocking group. The retained trace has 64 distinct runtime launch threads
for its first 64 NCCL calls. Installed vLLM's `pynccl.py` uses caller-thread
`ncclCommInitRank`/`ncclAllReduce`; fresh startup reports NCCL 2.30.7 versus
native 2.31.2. These are separate submission/version differences, not proof
that either library is inherently faster.

The unchanged lighter paired control
`/tmp/lunaflux-glm-http-nccl-light-20261020-v1` omits captured compute interiors
but retains eager collectives. It completes the literal request. Ordinal 8
is now 0.282272/0.152416 ms, while ordinal 5 is 6.641952/6.629696 ms. Rank 0
still records 91 profiler allocations totaling 152.366480 ms. The movable
outlier is not a stable cost of recurrent layer 4. The control alone does not
prove that profiler allocations cause every outlier.

Startup INFO confirms IB/RoCE on the dedicated link, not TCP fallback. INFO
must be file-directed through the offline worker bridge: an initial attempt
put diagnostics on the worker's framed stdout and failed startup. That failed
root is retained. A hardware-trace attempt at
`/tmp/lunaflux-glm-http-nccl-hw-20261020-v2` stopped at the 17 GiB early guard
before reaching the 16 GiB reserve; it is not a completed trace.

## Compiler candidate (explicit, not a promoted default)

`ReplayPolicy::CaptureExactCollectives(rows)` is a pure parallel-IR schedule:

```
live == 1 or live == selected rows -> captured compute + collective effect frame
all other positive row counts     -> compute intervals + exact-count collectives
both routes                       -> terminal ownership/cancellation fence
```

Unlike the previous padded-capacity graph, a partial prefill never reduces
inactive rows. The planner accounts for two complete frames plus interval
fallback storage before device binding. Existing private lowering captures
the same kernels, branches and ordered collectives once at startup, then
replays without per-collective NCCL submission calls. No NCCL blocking-mode
override, arithmetic change, comparison logic, runtime capture, hashing or
steady-state allocation is added. The HTTP manifest opts in with
`collective_replay: "exact-capacity"`; omitted configuration remains decode-only.

Local gates: 32 parallel-IR tests, 15 HTTP command tests and 136 packed execution
tests pass with existing migration-warning exclusions. Alternating exact and
partial rows, cancellation at all three row counts, BF16 storage, branch
ownership and zero steady-state allocation are covered. The fake graph counts
primary descriptors only; actual auxiliary-node execution requires GPU proof.
Existing native capture lease/replay/cancellation ASan+UBSan and explicit
ownership-balance gate passes. No new CUDA arithmetic or NCCL ABI is introduced.
Actual-source physical trace selection and its limitations are recorded below;
the candidate has not passed a net performance-promotion gate.

## Fresh reference safety boundary

`/tmp/lunaflux-reference-bench-glm-nccl-light-20261020-v1` reduces the startup
envelope to one sequence and capture size 1, while retaining 8192 context,
64-token chunks, 256 MiB KV and the same TP2 checkpoint. The new 250 ms host
watch stops at 18 GiB, before the user floor. It stopped during weight loading
at 17.95 GiB, before readiness or any request: **no fresh vLLM trace** from this
attempt. Do not count the failed attempt as a benchmark or silently change
checkpoint precision to make the reference fit.

## Actual exact-row result: submission fixed, latency claim rejected

Source commit `211791a1` was built on ARM in
`/tmp/lunaflux-glm-service-arm-exact-20261020-v1` using the qualified Spark
toolchain, 2 GiB CPU-only/no-swap ownership. It uses the same qualified kernel
artifacts as the retained serving path. The explicit candidate capture is
`/tmp/lunaflux-glm-http-nccl-exact-20261020-v1`.

Both ranks completed the literal 7,973-token prompt, with one output token,
zero cached prompt tokens and a terminal stream marker. This also exercises
64-row exact frames and a 37-row tail (7,973 = 124 * 64 + 37); no padded
collective operands are introduced for that tail. The post-capture request
then completed with **323** for **17 * 19**, finish reason `stop`, 26 prompt
and six completion tokens. This is a request-retirement/arithmetic smoke,
not broad model-quality qualification.

The capture contains 60/59 collective nodes on ranks 0/1; only the common
first 59 are paired. Their graph IDs match in order. All 59 correlate with
**one `cuGraphLaunch` per rank**, not 59 separately submitted NCCL calls.
The previous eager control used one runtime launch thread per collective.
The same NCCL geometry is retained: 32 CTAs, 640 threads, 96 registers/thread,
61,760 dynamic shared bytes. This demonstrates actual IR-to-device selection,
not just a manifest setting. It does not by itself establish unchanged NCCL
transport/protocol internals.

Auxiliary work is also present. For layer 3 on rank 0, shared gate/up runs on
stream 663 at 253515776--255551712 ns, while routed gate/up runs on stream 665
at 253621664--255944608 ns: **1.930048 ms overlap**. Routed down completes at
257050624, routed combine at 257078272, local MoE combine at 257084224, then
the sole combined MoE reduction runs at 257092192--257264736. The graph has
not dropped the auxiliary branch or restored separate routed/shared TP.

| Selected NCCL site | Previous eager paired R0/R1, ms | Unchanged lighter control R0/R1, ms | Exact-row graph R0/R1, ms |
| --- | --- | --- | --- |
| Recurrent layer 4 output, ordinal 8 | 3.442528 / 3.379264 | 0.282272 / 0.152416 | 0.147424 / 0.180640 |
| Light-control outlier, ordinal 5 | See retained paired report | 6.641952 / 6.629696 | 0.170976 / 0.159680 |
| New graph outlier, recurrent layer 1 output, ordinal 2 | See retained paired report | See retained paired report | **3.194368 / 3.148288** |

There is **no profiler-allocation overlap** on the new graph's collective
spans. Rank 0 has no profiler memory allocation inside the selected window;
rank 1 has one 0.016896 ms allocation between collectives. This removes the
measured short-lived-thread/profiler-allocation amplification, but **does not
prove that amplification caused every old long GPU span**: an unchanged
lighter control already moved the old outlier, and a new multi-ms outlier
survives without such allocation overlap.

The new ordinal 2 still has at most 0.170816/0.089824 ms of early arrival on
ranks 0/1; therefore approximately **3 ms remains after both kernels start**.
It cannot be explained as millisecond rank-start skew. There is no proxy/NIC
progress or device-stall trace in this capture to identify that remaining
cause. Do not label it wire transfer, a confirmed proxy starvation bug, or
profiler overhead without that evidence.

The first graph collective is different: rank 0 lasts 7.553760 ms, with a
**6.702848--6.963488 ms demonstrated early-arrival interval**. The first whole
graph launch itself takes 6.605024/5.522864 ms on the host. Warmup used a short
prompt, so it did not execute the new 64-row whole-effect graph before this
capture. First-use graph/communication preparation and recording disturbance
remain separate hypotheses, not a confirmed explanation for ordinal 2.
The external analyzer now reports a null *per-collective* launch lead for
captured nodes: a shared graph-launch correlation is not an individual NCCL
submission timestamp.

Crucially, the **sum of the same first 59 recorded collective spans increased**
from 17.301760/17.020832 ms in the retained eager trace to
**23.928608/18.978048 ms** in the graph trace. These are selected GPU spans,
not request latency or throughput; neither rank is a net latency win in this
cold capture. No subtraction of CPU overhead, exclusion of the first three
collectives, or overall speedup claim is justified. **Keep the candidate
explicit and nondefault; do not promote it as a completed NCCL performance
fix.** The executable submission change is real, but the remaining multi-ms
progress delay is unresolved.

The retained vLLM first two dense-FFN reductions are 0.239808/0.174592 ms on
rank 1; the new candidate's matching sites are 2.010144/0.181120 ms. That
reference remains the older successful capture, not the failed fresh attempt
above. Its NCCL is 2.30.7, native is 2.31.2, and its maximum output count differs
(two versus one); only matching first-chunk operator sites are compared.

Next discriminating measurement, **not run in this campaign**: paired first
versus warmed replay of this same exact-row graph, with matched explicit
row-count evidence and NCCL proxy/NIC progress observation. Separate (a)
rank arrival, (b) graph preparation, (c) proxy/transport progress and (d)
device execution. A startup graph-upload or polling-policy change should not
be made just because the remaining trace resembles one of these hypotheses.
Fresh full-model vLLM profiling must first fit the existing memory floor;
do not change checkpoint precision or waive the reserve to obtain a trace.

Evidence (all external to compiler legality):

- `/tmp/lunaflux-glm-nccl-exact-paired-20261020-v2.json`:
  59 paired collectives, 0.260640 ms offset-bound width.
- `/tmp/lunaflux-glm-nccl-exact-overhead0-20261020-v2.json` and rank 1 counterpart:
  distinct runtime calls/threads, profiler overlaps and graph API timestamps.
- `rank0.sqlite` / `rank1.sqlite` in the exact candidate root: actual node
  identities, geometry, auxiliary overlap and GPU spans.
- `correctness.response`, `memory-watch.txt`, `bounded-memory.txt` in that
  root: request answer, owned memory and early-reserve observations.

Candidate memory observations: maximum GPU usage **94.2432/94.2695 GiB**;
minimum host availability across the two watchers **17.2982/18.3428 GiB**;
owned swap zero. Both diagnostic GPU owners retired before restoration.
The full native suite completed **5,333/5,333**, using the same existing
migration-warning exclusions; targeted phase checks and the native ownership
sanitizer gate above also passed. The final analyzer changes pass 2+1 tests.

## Restoration

The previously qualified `eb152338` GLM service was restored on both Sparks,
not replaced by this unproven performance candidate. `/health` returned
`{"status":"healthy"}`. A backend HTTP chat request returned **323** for
**17 * 19**, finish reason `stop`, 26 prompt/six completion tokens. Both user
units are active; owned swap remains zero. Maximum observed GPU use during
restoration is **94.1904/94.2188 GiB**; minimum observed host availability is
**17.9160/19.6070 GiB**. Observation covers startup through the answer check.
Evidence: `/tmp/lunaflux-glm-nccl-restoration-20261020-v1`, including the
unmodified answer, per-rank unit state/journals and sampled memory. The
backend check is not a new customer-key/public-route qualification.

Implemented and tested: exact-row graph ownership/fallback and removal of
per-collective host submissions in that explicit path. **Unfinished:** a
causal explanation/fix for the remaining approximately 3 ms simultaneous
progress stall, and a fresh matched vLLM trace within the host reserve.
The finite campaign stops with that negative result visible, production
restored, and no new kernel/transport policy guessed into production.
