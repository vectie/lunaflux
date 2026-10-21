# Deferred two-rank trace collection

Scope: fixed cold/warm execution windows of the latest fragment projection
path, not an overall benchmark. The model geometry, numerical laws and existing
100 GiB/Spark, 16 GiB host reserve and earlier 17 GiB diagnostic guard remain
unchanged. No payload hashing.

The previous capture stopped at 16.874 GiB host availability before crossing
the user floor. Its collector generated reports with the model still resident.
Report conversion is therefore moved after GPU-owner retirement; whether this
removes the measured peak must be tested, not assumed. The installed Nsight
2025.3.2 does not offer `stop --defer-report`; a separately extracted CLI is
qualified before full-model use. Drivers and the installed profiler are not
replaced. At most two capture configurations are attempted in this iteration.

The native window controller also incorrectly treated an issued stop RPC as
a completed stop. A competing timer and request completion could start another
capture or retire workers while the first stop was still draining. Stop callers
now wait for completion. This is offline orchestration, not compiler policy.

Before this iteration, the public rank 1 had already aborted at 03:12:28 CST,
after starting at 00:50:03. Rank 0 was subsequently stopped. Systemd records
SIGABRT, 60.9 GiB cgroup peak and zero owned swap; this is not evidence of OOM.
Both GPUs were free at inspection. That failure is retained separately from
projection qualification; the candidate was not serving when it happened.

Results and deployment state will be appended after physical validation.

## First configuration: stopped before capture

The separate Nsight2026.5.1 CLI captures the actual approved-spawn child GPU
operation successfully (`/tmp/lunaflux-nsys-worker-deferred-20261021-v1`).
No driver changes. The complete native attempt at
`/tmp/lunaflux-glm-http-fragments-deferred-20261021-v2` nevertheless stops during
loading at host availability17,796,520KiB (16.972GiB), before readiness or a
capture start. Both owners retire. Deferred conversion cannot explain or fix
that loading-phase overhead; no kernel timing is inferred from this failure.
The preceding v1 preparation was rejected before launch for missing the
required16-site graph-window policy; its outputs remain retained.

The new CLI rejects the legacy `cuda-hw` flag before launching the tiny worker
(`...nsys-worker-hardware-new-20261021-v1`). Its `cuda` mode already selects
hardware tracing where supported. The second full-model configuration instead
uses explicit `cuda-sw` and `node:host-only`, after a separate child-worker check.
It preserves the same workload, AOTs, numerical laws, capacity and17GiB early
guard. Device-launched graphs are not part of this runtime; individual nodes of
host-launched graphs remain required. Capture coverage must still be inspected
before assigning kernel times. The unused hardware-named prepared service root
never launched a model.

## Second configuration: still does not fit

The explicit software child-worker capture passes at
`/tmp/lunaflux-nsys-worker-software-new-20261021-v1`. The full pair at
`/tmp/lunaflux-glm-http-fragments-software-deferred-20261021-v1` stops during
loading at17,542,348KiB (16.730GiB), before capture/readiness. Both GPUs release
and owned swap remains zero. The two-configuration capture budget is exhausted;
there is no new complete timeline or measured decoder gain from this iteration.
No reserve is lowered and no opaque graph envelope is relabelled as node data.

A loading snapshot records target RSS about2.0GiB on each rank, another693MiB
for the rank0 HTTP process, and roughly83MiB of visible Nsight controller RSS
per node. These are overlapping-process RSS observations, not a causal or
additive accounting of unified GPU allocations. This does not justify claiming
that report conversion alone caused the shortage. Detailed snapshot and owned
memory/terminal observations remain in the failed root.

During the unprofiled acceptance load, the rank0 GPU worker is998,244KiB RSS
and its HTTP parent784,836KiB, versus2,094,512/709,552KiB in the profiled load.
The snapshots are at different loading points, so this is not an exact peak
ablation. They nevertheless locate roughly1GiB of additional target-process
resident memory before report conversion; controller RSS alone cannot explain
the observed loading shortage. No claim is made about which internal Nsight
allocation owns that difference.

Unprofiled acceptance is separate from tracing: the prepared serving candidate
retains16 admission lanes,8 resident slots,8192 context and the selected latest
projection/16-site graph path. The external acceptance runner checks literal
7973-token retrieval and complete answers, JSON/SSE/Responses, distinct routing,
sampling, cancellation and controls; it performs no overall timing benchmark.
Promotion requires its actual results, not the earlier component speedup.

## Acceptance and deferred-conversion validation

Acceptance at `/tmp/lunaflux-glm-http-fragments-acceptance-20261021-v1`
completes. The actual7973-token request has0cached tokens and returns
`orchid-70` with `finish_reason=stop`. JSON/SSE arithmetic returns323;
Responses returnsParis; four concurrent distinct identifiers are correctly
routed. Three same-seed sampling requests are identical. Cancellation/reuse
and request controls pass. Reviewed merge code handles empty lists and is
O(n+m); the probability answer is3/10; the transaction explanation is coherent.
The floating-point explanation is substantively correct but ignores its
two-sentence instruction. This is not a universal output-quality claim.

The unprofiled accepted artifact is
`/tmp/lunaflux-glm-http-fragments-serving-20261021-v1`, with the source-pinned
`b1788d1d` executable and qualified projection AOT. All230 paired memory
observations stay within the requested envelope: minimum host18,012,700KiB
(17.178GiB), maximum GPU96,530MiB (94.268GiB), owned swap0. Owners retire
after acceptance. No latency or throughput aggregate is produced.

Deferred conversion is independently exercised with a CPU-only target at
`/tmp/lunaflux-nsys-deferred-cpu-proof-20261021-v1`. `stop --defer-report`
retains the collection without creating the report; after the target exits,
`finalize --session` generates the72KiB report under a2GiB/no-swap owner.
This proves the collector lifecycle, **not** a GPU timeline or reduced
full-model peak. New collector preflight rejects a profiler lacking the flag
before loading a model. The installed2025.3.2 profiler remains unchanged.

Targeted external-runner tests pass13/13. The changed helpers check with
warnings denied, retaining only the launcher's existing deprecated-string
warning exclusion20. No runtime/compiler or native ABI code changed in this
iteration; no new full-suite or sanitizer result is claimed. The kernel's
earlier BF16 correctness/sanitizer gates remain in the fragment-pipeline report.

## Deployment

Both stopped LunaNexa units have been relocated to the qualified opaque
`glm53-lunaflux-b1788d1d` release, retaining their previous release and unit
backups. Checkpoint files are hard-linked, not scanned or hashed. The full
16-lane manifest is carried unchanged except deployment paths, existing
identity labels and backend address. Paired startup and post-restore checks
are pending; this section does not yet claim the endpoint is healthy.

The latest two-rank node retrace remains incomplete. The historical vLLM
October19 trace is not replaced by these failed attempts. No new execution
graph allocation, TP-wait comparison or full-decoder speedup is asserted.
