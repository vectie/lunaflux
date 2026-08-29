# Rank-group protocol

This package coordinates one scheduler-owned semantic worker plan across one
ordered local tensor-parallel group of 2..16 ranks. Single-rank execution stays
owned by `WorkerProcessSupervisor`. This package does not copy plan rows, select devices,
allocate KV memory, call NCCL, sample tokens, or expose backend handles.

The owner admits one or two exact reusable scheduler plan-buffer identities and
allocates its follower and drain bitsets once at startup. Every plan then
reuses those fixed arrays. Rank zero is the only completion producer;
other ranks return a payload-free scalar acknowledgement. The canonical
completion becomes visible only after all followers acknowledge the same group
generation, plan sequence, model-plan generation, rank, and world size.
`complete_leader` validates the exact live scheduler plan and completion once
and returns the only post-validation retirement capability. The coordinator
may retain that capability across scheduler commit and retire the group only
after the scheduler has consumed and reset its plan/completion owners; group
retirement deliberately does not re-read those now-stale scheduler values.
An outer supervisor need not retain either capability in heap state:
`reauthenticate_submitted` rederives the live submitted value only from the
exact scheduler plan, while `reauthenticate_completed` accepts only copied
post-validation epoch evidence and never reopens scheduler-owned storage.

A deadline, rank loss, collective failure, execution failure, duplicate ACK,
or scalar substitution permanently faults the group generation. Its bounded
failure is published once. Every rank not known lost must then acknowledge
drain before the owner can close. If another worker is proven lost during that
drain, the coordinator can exclude that exact rank without replacing the
original failure reason, so recovery never waits for an impossible ACK.
Recovery constructs a new owner with a new group generation and the
scheduler's prior plan sequence; a faulted owner is never reused.

The package is intentionally thread-confined. A service coordinator must
serialize calls and supply monotonic ticks explicitly; there is no global
clock, runtime context, worker registry, or configuration object here.
