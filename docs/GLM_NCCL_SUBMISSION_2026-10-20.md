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
Physical qualification and actual-source trace selection are still required.

## Fresh reference safety boundary

`/tmp/lunaflux-reference-bench-glm-nccl-light-20261020-v1` reduces the startup
envelope to one sequence and capture size 1, while retaining 8192 context,
64-token chunks, 256 MiB KV and the same TP2 checkpoint. The new 250 ms host
watch stops at 18 GiB, before the user floor. It stopped during weight loading
at 17.95 GiB, before readiness or any request: **no fresh vLLM trace** from this
attempt. Do not count the failed attempt as a benchmark or silently change
checkpoint precision to make the reference fit.

Further results and final restoration are recorded below when completed.
