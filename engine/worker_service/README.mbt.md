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

Production construction starts from an immutable `SchedulerBlueprint`, two
ordinary caller-owned approved roots, and the exact executable/startup/source
inputs. `prepare_owned` allocates the service and cleanup publication shells,
instantiates a fresh scheduler, and preflights deterministic worker buffers
before root acquisition. It then creates and claims the rooted process as the
last fallible join. Its one-shot opaque result exposes either the ready service
or the only retained cleanup owner, never both. A closed failure raises its
primary error, or a bounded compound error when cleanup also failed; only live
child/root authority yields `OwnedServiceCleanupRequired`.

`new_fixture` is compatibility-only: its caller retains aliases to both mutable
owners and the resulting service is permanently ineligible for future online
session admission. There is no production-facing alias-taking constructor.

An owned preparation makes one permanent ownership-family choice.
`take_raw_ready` irreversibly selects compatibility dispatch before exposing
the service. `take_online` instead transfers a preallocated opaque
epoch-authenticated `OnlineWorkerLease` without ever exposing the service. The
lease retains the monotonic clock, exact request generation, token position,
and global publication cursor. It exposes only sanitized value publications,
including explicit suppressed-token evidence after a cancellation or deadline
cut.

The lease is a single-session lower-level engine seam for the owned
`service/online_session` aggregate. Production online construction prepares an
exclusive scheduler admission and its monotonic read before rooted activation,
then commits that exact shell after startup without accepting a request through
the lease. Lower fixture entry points are boundary-restricted to tests; an
application cannot pair a raw `TokenizedRequest` with the aggregate. No decoder,
scheduler, process, request handle, request identity, generation, or raw
publication owner escapes.
The service and lease remain thread-confined; copying a current lease reference
does not create independent authority and is forbidden by the aggregate's
exclusive-owner discipline. The aggregate never releases or renews this owner:
it terminally shuts down a healthy request, closes after recovery, or closes an
empty failed admission. Sequential sessions require a fresh owned aggregate.

Online recovery is lease-authenticated: child recovery, exact-flight
retirement, device invalidation, replacement, restart-forbidden drain, and
terminal close retain the same epoch and request evidence. Clean shutdown is
available both after the exact terminal publication and before admission. A
queued natural terminal wins before the retained clock is sampled.
If cancellation or expiry cuts a request while one physical exchange is
already active, online progress can retire only that recorded exchange; it
cannot build another plan. The stale completion is consumed without reviving
the logical request, after which the exact terminal can close cleanly.

The owned-preparation allocation gate proves that service, cleanup, rooted,
child, executable, and fixed handshake storage are allocated before root
activation. Configure, source, and expected Ready bytes are encoded before that
point. Native spawn, scalar handshake I/O, exact Ready comparison, scalar close,
and ready/cleanup publication introduce no managed allocation while child or
root authority is live.

`progress` owns the one-flight exchange state and advances exactly one logical
transition per call: idle/backpressured, started, pending, completion-ready, or
committed. It checks exchange credit before creating a scheduler obligation,
records the sequence before beginning transport, and retains a received frame
without I/O while scheduler publication is backpressured. `Committed` means
both scheduler acceptance and process-side retirement succeeded.
`take_next_publication` forwards the scheduler's atomic globally ordered
dequeue, so service adapters never choose between physical token and terminal
rings.

Recovery first closes and reaps the old child. `recover_flight` probes for a
pinned completion, then an exact process reservation, before synthesizing a
bounded worker failure. The process submission is abandoned only after
scheduler retirement succeeds. Missing reservation evidence enters explicit
`ServiceRestartForbidden`; it can never start a replacement. Before a normal
restart, `invalidate_device_state` is a separate retryable phase that fails
active requests and releases their host page/table identities. Waiting requests
remain eligible for the replacement. Operators may instead `abandon_restart`
and incrementally `drain_restart_forbidden`; each call terminally releases at
most one request, and only `ServiceTerminalCloseReady` can close after the drain.

`shutdown_clean` rejects both an outstanding flight and any live request. The
terminal `close` path likewise cannot silently discard request, KV, or page
authority. Cancellation and deadline expiry remain ready-state operations and
cannot create new recovery obligations.

This owner implements deterministic failure orchestration, not retry policy,
backoff, health endpoints, model/device loading, or CUDA readiness. Those
instance-level policies remain above this package.
