# Step handoff: measured bounds, source ownership and pending attribution

This extends `BENCHMARK_FULL_STEP_ATTRIBUTION_2026-10-04.md`. It does not
claim a new throughput result or identify all GPU-empty time as CPU scheduler
cost. No new GPU workload was run in this investigation.

## Exact existing-trace decomposition

Workload: Qwen3-0.6B, 4096 input / 256 output tokens, concurrency 16,
measured trial `i4096-o256-c16-t1`. The same-thread CUDA API boundaries are
joined to actual graph-node GPU envelopes. There are 287 inter-step pairs.

| Segment | Sum, ms |
| --- | ---: |
| Last GPU node end → completion-event API return | 2.430 |
| Event return → greedy readback start | 1.441 |
| Compact greedy D2H API span | 2.983 |
| Feedback end → next descriptor upload start | **314.502** |
| Descriptor H2D span | 17.400 |
| Last descriptor upload → graph launch API start | 1.492 |
| Graph launch API span | **47.604** |
| Launch API return → first GPU node start, signed | -0.211 |
| Total | **387.641** |

The launch tail is signed: GPU work can begin before `cuGraphLaunch` returns.
Upload API time (16.131 ms) is a subset of upload span, not another additive
component. Event synchronization duration while the preceding graph is busy
is not counted again as a gap.

The mandatory completion wake-up and compact sampled-result readback occupy
only 6.854 ms of these gaps. Removing deterministic completion ownership is
therefore neither justified nor likely to remove the dominant envelope.

274 producing handoffs account for 306.862 ms of the host-feedback envelope
(1.120 ms per pair). The other 13 account for 7.639 ms (0.588 ms per pair).
This distinction is useful for attribution but is not a controlled experiment:
the shapes and request phases differ. It does not prove that publication alone
costs the difference.

Artifact:
`benchmarks/results/full-step-20261004.yctDvljQ/step-handoff.json`, SHA-256
`223088f25ceb5423bf1f0c833a7ffa7a90dd0d295228f9c4cb1c9a6c1b45eb76`.
Downloaded copy matches the remote hash. The analyzer is
`benchmarks/gpu_pipeline/analyze_step_handoff.mbtx`.

## Actual runtime/effect chain

The comparison endpoint uses `lunaflux_qwen3_token_id_bridge`, whose upstream
is **native-framed**, not the OpenAI connection pool. The selected kernels are
graph nodes; an eager-fallback explanation does not fit this capture.

The host-feedback envelope contains all of the following, not just scheduling:

1. Child sampling results, descriptor retirement and completion serialization.
2. Child IPC write and parent nonblocking completion read.
3. Parent completion validation and scheduler commit.
4. Per-request stop/terminal checks, semantic publication, framed encoding,
   immutable output transfer, acknowledgement and socket output.
5. Next plan construction, including the current page table, binary checksum,
   reservation and parent IPC write.
6. Child IPC read, frame validation, completion-writer validation and device
   descriptor preflight/staging before the first H2D API.

Ownership boundaries remain necessary: a decoded stop/cancellation must take
effect before its request enters the next plan; the old descriptor cannot be
overwritten before its graph retires; request/publication generation must
remain exact. A fix must distinguish these dependencies from the physical
copy, polling and rendering mechanisms.

Read source paths include:

- `engine/device_worker/execute.mbt`
- `engine/device_step/paged_executor_run.mbt`
- `engine/device_step/paged_executor_greedy_sampling.mbt`
- `engine/device_step/stage.mbt`
- `engine/device_worker_child/run.mbt`
- `engine/worker_process/root_bound_exchange.mbt`
- `engine/worker_service/online_lease.mbt`
- `service/online_session/online_multi_progress.mbt`
- `service/online_tcp/framed_pool_progress.mbt`
- `ops/runtime_instance/control_owner.mbt`

Two tempting explanations were rejected by source/route verification:

- Native-framed listener/input polls already use zero timeout while a worker
  is waiting. Its service-advanced turn returns before idle admission polling.
- Operational HTTP control polling is already clamped to zero in both the
  workspace and the immutable serving source. The owner's control turn must
  not be described as a measured 1 ms timer.

## Diagnostic instrumentation prepared, not yet GPU-captured

`install_host_handoff_trace.mbtx` applies only to a disposable source copy.
Markers include monotonic nanoseconds, PID and exact worker sequence. They run
only at transition boundaries, not on every pending poll:

| Kind | Boundary |
| --- | --- |
| 19 | Child owner entered, before writer/ready-executor validation |
| 20 | Child before descriptor staging |
| 21 | Child after staging, before graph execution |
| 22 | Child graph execution complete, before sampled-result readback |
| 23 | Child after sampling, before descriptor retirement |
| 24 | Child before completion serialization |
| 30 | Parent before next plan encoding/reservation |
| 31 | Parent encoded/reserved plan |
| 32 | Parent finished writing plan IPC |
| 33 | Parent finished reading completion IPC |
| 34 | Parent completed completion-frame validation |

Together with CUDA API timestamps these separate child completion, IPC,
parent commit/publication/next-plan construction and child validation/staging.
`trace_step_host_waits.mbtx` wraps the existing bounded workflow and enables
CUDA plus OS-runtime tracing, while retaining every AOT cubin unchanged.
Marker stderr is explicitly validated. The wrapper retains 64 GiB/no-swap
systemd caps, the 32 GiB available-memory reserve and normal drain checks.

After the serialized GPU slot is granted, use a new empty output directory:

```text
moon run trace_step_host_waits.mbtx trace_all_new_runtime.mbtx IMMUTABLE_SERVING NEW_ROOT --selected --long-c16-256 --work-rows install_host_handoff_trace.mbtx
```

The marker capture is a diagnostic timeline, not an unprofiled performance
benchmark. Any repair must subsequently be measured without instrumentation,
with identical model, workload, source/kernel selection and output checks.

## Separately scoped OpenAI source repair

A genuine but **non-executed in this benchmark** defect was found in the
OpenAI pool: service-advanced turns still reached an idle accept timeout, and
coordinator `Waiting` was indistinguishable from runnable progress.

The local candidate repair uses a private value-type effect policy:

- Runnable or terminal progress: skip idle listener waiting.
- Worker waiting: perform a nonblocking admission readiness probe.
- Genuinely idle service: preserve the configured bounded listener wait.

The waiting flag belongs to the pool owner. No global state, model-specific
condition, numerical change, CUDA ABI change or token-step filesystem/crypto
operation is added. Semantic credits, output-flight lifetime, cancellation,
drain and error handling are preserved. This is not evidence that the captured
native-framed gaps have been repaired, and it has not been promoted.

## Validation and completion limits

- All three new `.mbtx` helpers pass warning-denied native script checks.
- Disposable marker-instrumented child and launcher build successfully on
  macOS; no GPU execution was attempted.
- `service/online_tcp`: 57/57 native tests pass, including three new effect
  policy regressions and existing zero-wait socket/deadline checks.
- Targeted warning-denied check passes using the existing migration warning
  exclusions `-79-20-29-25-92-14`.
- Unqualified global warning-denied check is not clean with the current local
  toolchain: existing implicit-trait-promotion, deprecated APIs, unused imports
  and black-box-test qualification warnings remain outside this repair.
- `moon info --target native service/online_tcp` completes, with existing
  toolchain migration warnings.
- No commit, push, deployment or new throughput claim. Exact active-path CPU
  attribution and any corresponding native-framed repair remain pending the
  serialized diagnostic capture.
