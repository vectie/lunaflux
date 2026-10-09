# Parallel execution IR

Branch: `parallel`. Compiler lowering and opt-in CUDA/NCCL dispatch are
implemented; physical overlap performance and two-Spark qualification remain open.

## Contract

The functional core remains immutable model plan → rank plan → physical
execution plan. `compiler/parallel_ir` adds an immutable, rank-local memory
effect graph after physical binding. It does not replace LunaTile, the request
scheduler, kernel selection, or distributed worker ownership.

An input program declares arena capacities/initialization, bounded buffer
views, compute and collective steps, read/write effects, and explicit earlier
dependencies. Step order is a topological source order. Compilation infers
read-after-write, write-after-read and write-after-write dependencies from
physical byte overlap, including distinct views of the same allocation.
Temporary reads require prior initialization. Collective sequence numbers are
contiguous, and group composition checks matching collective kind, element
type and byte geometry across all ranks. Pure values confer no device or
artifact authority.

Compilation also records each buffer's uses and source-order lifetime. Source
indices alone do not authorize asynchronous reuse: reuse must follow all
users through the dependency graph. Aliased writes therefore depend on prior
readers, including communication that has not completed.

Lowering has two explicit policies:

- Ordered: one FIFO queue, preserving the current compute/collective order.
- Compute/communication overlap: one FIFO queue for each kind, with precomputed
  event record/wait commands on cross-queue dependencies. Only the latest
  required predecessor on the other FIFO queue needs an event. Both queue tails
  must complete before the invocation retires or its memory is reused.

The existing TP execution planner projects its admitted kernel operands and
collectives into this representation. Physical activation aliasing and scratch
reuse remain authoritative; lowering cannot manufacture additional overlap by
ignoring those constraints. Existing execution-plan identity continues to bind
the physical inputs from which the IR is deterministically derived.

The public startup APIs are `TensorParallelExecutionPlan.parallel_program()`
and `tensor_parallel_execution_plan.parallel_group(plans)`. Projection is
opt-in: existing admission does not build the additional graph. Call
`Program.lower(Ordered)` or `Program.lower(ComputeCommunicationOverlap)` to
obtain immutable `Execute`, `Record`, and `Wait` commands. `Execute.step`
indexes the program's compute-operation or collective action; event IDs index
`Schedule.event_producers()`. Backends must wait for every
`Schedule.completion_steps()` tail before retiring an invocation, and must not
reuse its events for another overlapping invocation.

The physical adapter treats persistent KV operands conservatively as
read/write until their contracts expose finer effects, treats scratch as a
write, and represents gather as a local-shard read followed atomically by a
full-result write. Read-only weight/descriptor arenas need no hazard history.
Buffer lifetime queries are startup analyses, not allocation-free runtime
operations. Reusing a named view requires all of its uses to precede the new
step and all earlier overlapping views to be ordered before that step.

All graph construction, initialization analysis, lifetime analysis and queue
lowering happen before execution. The representation owns copied immutable
arrays. It opens no files, sockets, GPU handles or communicators and introduces
no token-step validation, hashing or allocation.

## Scope and validation

### Executor integration

The CUDA/NCCL adapter is selected by an explicit startup policy. Ordered
execution remains the default. Overlap preparation owns a second stream and
one reusable CUDA event with timing disabled per lowered producer. MoonBit dispatches the immutable
queues using scalar cursors; graph analysis and resource creation stay outside
the invocation path. Cross-stream waits are submitted only after the matching
event record has been issued, avoiding CUDA's unrecorded-event no-op behavior.

The NCCL owner retains one outstanding collective. A nonblocking submission
must reach NCCL async success before recording its CUDA completion dependency;
device completion remains the sole collective sequence-commit point. Compute
can continue while submission or device completion is pending. A later
collective waits for the previous collective to retire. Both lanes must retire
before descriptor, memory, or event reuse. Fault cleanup aborts NCCL, drains
both streams, then releases events and existing resource leases.

These ordering rules follow NVIDIA's [NCCL nonblocking submission contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html#nonblocking-group-operation)
and [CUDA event semantics](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html).

For embedding, pass `parallel_policy=ComputeCommunicationOverlap` to
`tensor_parallel_device_worker.prepare`, `prepare_configured_rank_child`, or
`tensor_parallel_rank_child.run`. `worker.parallel_policy()` reports the
selection. Kernel graph telemetry continues to describe eager versus captured
kernel execution; the queue policy is separate.

For deployment, build `cmd/tensor_parallel_overlap_rank_child` with
`moon build --target native --release cmd/tensor_parallel_overlap_rank_child`
and select its digest-pinned executable using the existing rank-child launch
configuration. It uses the same inherited control channel
and startup admission as `cmd/tensor_parallel_rank_child`; it is not a standalone
listener. The original executable and default API policy remain ordered.

For the two-Spark TLS worker, append `--overlap` on both machines:
`two_spark_worker ABSOLUTE_DEPLOYMENT_ROOT CONFIG_SHA256 --overlap`.
The existing digest-pinned deployment and network admission remain required;
the flag changes only the local queue policy. Embedders can likewise pass
`parallel_policy=ComputeCommunicationOverlap` to `ops/two_spark_worker.run`.

Tests cover alias hazards, initialization, collective agreement, deterministic
queue lowering, independent compute/communication overlap, and reuse only after
completion dependencies. Integration tests use existing admitted TP plans.

The overlap worker binds the emitted commands to its admitted AOT executor,
communication stream and startup-owned events. Physical numerical, resource
and performance gates are still required before deployment qualification.
Pipeline/expert parallelism, automatic placement and cost-based scheduling
can build on this representation but are not implemented by this milestone.

Local validation (2026-09-26): 17 native IR tests pass with warnings denied,
including the executable package example, a pairwise alias-hazard reference,
and enumeration of valid queue interleavings for a compute/collective fixture.
The TP execution-plan, device-worker and worker-bootstrap packages pass all
25 native regression tests. Package formatting, interface generation and the
core architecture boundary check pass. Repository-wide warning-denied native
check/test commands stop on existing warning 79 (`implicit_impl_as_method`)
deprecations outside this package; this milestone does not suppress them.

Executor integration validation additionally runs the real MoonBit worker
against delayed fake CUDA/NCCL boundaries in both policies. Each policy warms
two ranks and measures 65 further cycles per rank with zero detected managed
or native allocation, zero blocking synchronization, and balanced explicit
resource cleanup. The fake boundary rejects premature communication event
records and unrecorded or same-stream waits, and observes waits before GPU
collective completion. CUDA and NCCL native ASan/UBSan probes cover event-wait
resource guards, asynchronous submission, completion and failure retention.
These are host-side software checks, not GPU numerical or speedup evidence.

Final executor checks: 70 affected native tests pass (67 compiler/device/worker
tests plus 3 two-Spark entry tests). Ordered rank child, overlap rank child and
two-Spark worker release builds pass. The worker boundary, NCCL ABI boundary,
CUDA and NCCL ASan/UBSan gates and full warmed allocation gate pass. The ABI
scan excludes generated `**/_build/**` copies. Cleanup evidence includes abort
with work in flight and retry after a dependency-event close failure. Strict
warning-denied worker check/test remains blocked by existing warning 79 in
dependencies; no warning suppression or broad migration is included.
