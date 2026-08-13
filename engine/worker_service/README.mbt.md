# Scheduler/worker service owner

This package is the single thread-confined join between one scheduler and one
root-bound worker process. The scheduler retains reusable A/B storage, but the
production service admits only one outstanding plan to match the serialized
device child. Scheduler plan owners, process buffers, native handles, and
completion owners remain in their owning packages.

Construction also requires an independent immutable `WorkerServiceBinding`
containing the expected bootstrap and bootstrap-source digests, process-visible
device ordinal, and inference-contract limits. The service verifies those values together with the
scheduler's model identity, model generation, predecessor sequence, and exact
retained worker-protocol limits before it owns a ready process. The same binding
is rechecked before every replacement startup. The retained immutable source is
sent again, and the process handshake requires the Ready response to reproduce
that exact contract byte-for-byte.
The child response is evidence, never the source of expected deployment
identity.

`submit_next` checks the one-plan process credit before creating a scheduler
obligation, then records a plan before attempting transport, so a failed write
never loses the scheduler retirement obligation. `complete_oldest`
receives strictly in order and keeps a validated process frame pinned while
scheduler output or terminal publication is backpressured.
`take_next_publication` forwards the scheduler's atomic globally ordered
dequeue, so service adapters never choose between physical token and terminal
rings.

Recovery first closes and reaps the old child. recover_oldest then commits a
pinned valid response normally or synthesizes a retryable bounded worker-failed
completion for an unreturned plan. The process submission is abandoned only
after scheduler retirement succeeds. A replacement can start only after the
single production obligation is gone and every surviving active request has
been terminally failed and stripped of host page/table identities, because the
new child owns a fresh device-KV arena. Waiting requests remain eligible because
they have no device state. The replacement uses the exact non-reusing
predecessor derived by the closed supervisor.

This owner implements deterministic failure orchestration, not retry policy,
backoff, health endpoints, model/device loading, or CUDA readiness. Those
instance-level policies remain above this package.
