# Two-rank serial decoder pipeline

One model-neutral coordinator joins a local ingress rank, plain remote terminal
rank control and explicit activation effects. Both ranks receive the same
canonical request plan. Metadata retirement precedes execution; residual DMA
retirement precedes remote execution; all stage results and the remote commit
receipt precede the coordinator's single model completion.

Only one contiguous request is active. Prefill continuation/decode use retained
rank-local history; request release must retire on both ranks before reuse.
Local or peer failure poisons the entire step. Close control, drain activation,
then close rank queues/ports. This implementation has two pipeline stages, not
tensor parallelism or an arbitrary topology scheduler.

`cancel_request()` cancels the active request without closing resident weights,
queues or transport. `progress()` drains already-issued work and the peer commit,
discards unpublished output and releases both request frontiers. Reuse is allowed
only after `request_cancelled()` is true. Cancellation cannot retract tokens
already published, and it cannot turn a real execution failure into success.
The verification coordinator uses the same rule, additionally draining any
already-issued restore without submitting another accepted-prefix replay.

GLM and DeepSeek generation adapters accept the explicit replay suffix
`--cancel-request INDEX --after-steps COUNT`. It is parsed once before CUDA;
COUNT counts submitted ordinary frames or verification blocks, not polling
iterations, generated tokens or internal restore/replay substeps. This optional
control is not a default serving limit. A cancelled request writes `cancelled.txt`
instead of its final `tokens.txt`; previously published completions remain.

When a verification coordinator shares the same rank/client owners, its
completed request release retires those owners but not this coordinator's
publication frame. Call `retire_publication()` after awaiting verification
release, before the next ordinary prefill. This local host-state transition
performs no duplicate device or network release. The resident-session regression
returns to a third distinct zero-position prefill to cover this handoff.

The native regression uses real loopback control/activation sockets and a
device test double. It proves ordering, transferred bytes, token publication,
reuse and failure ownership; it does not prove real-checkpoint GPU arithmetic,
GPUDirect, RDMA, whole-model throughput or three-model support.
