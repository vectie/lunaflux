# Scheduler/worker service owner

This package is the single thread-confined join between one scheduler and one
private physical-worker transport. The scheduler retains reusable A/B storage;
the service owns one physical exchange and may prebuild one other
scheduler-owned plan while the serialized device worker runs. Scheduler plan
owners, process buffers, native handles, and completion owners remain in their
owning packages. Production preparation selects either the rooted single-worker
owner or one generation-scoped tensor-parallel group; request, scheduler, KV,
online-session, and publication code reaches either one only through the same
package-private physical-transport adapter.

The adapter rederives received and accepted authority from the retained scalar
flight sequence. No received/submitted/completed capability is boxed into the
warmed service state. The tensor-parallel variant provides the same contract:
healthy close, child-only recovery cleanup that retains roots, fresh nonzero
group generation and NCCL rendezvous, generation-bound Configure regeneration
from the scheduler's retired predecessor, failed-replacement cleanup, and
restart. Replaying old rank startup bytes or making tensor-parallel recovery
terminal is not an accepted fallback.

The service retains one backend-neutral `WorkerServiceIdentity`: model identity,
model generation, bootstrap-source digest, worker limits, and inference limits.
The existing `WorkerServiceBinding` remains the single-worker constructor input
for bootstrap digest and device ordinal, but is retained only in the private
physical binding. Tensor-parallel topology, per-rank bootstrap, and device
evidence likewise stay private. Each replacement reauthenticates the neutral
identity and exact scheduler predecessor through its selected transport; no
synthetic rank-zero binding is constructed.
The child response is evidence, never the source of expected deployment
identity.

Production construction starts from an immutable `SchedulerBlueprint`, two
ordinary caller-owned approved roots, and the exact executable/startup/source
inputs. `prepare_owned` allocates the service and cleanup publication shells,
instantiates a fresh scheduler, and preflights deterministic worker buffers
before root acquisition. It then creates and claims the rooted process as the
last fallible join. Its one-shot opaque result exposes either the ready service
or the only retained cleanup owner, never both. Once the private root pair is
acquired, both live and already-closed child/root failures publish
`OwnedServiceCleanupRequired`; only pre-acquisition rejection raises.
Owned preparation diagnostics use the same bounded physical-transport failure
vocabulary; exact root/process failures remain private evidence in the retained
cleanup owner so a future group preparation does not require a second service
error surface.

`new_fixture` is compatibility-only: its caller retains aliases to both mutable
owners and the resulting service is permanently ineligible for future online
session admission. There is no production-facing alias-taking constructor.

An owned preparation makes one permanent ownership-family choice.
`take_raw_ready` irreversibly selects compatibility dispatch before exposing
the service. `take_online` instead transfers a preallocated opaque
epoch-authenticated `OnlineWorkerLease` without ever exposing the service. The
lease retains the monotonic clock, startup-preallocated request lanes, each
lane's exact generation and token position, and one global publication cursor.
`try_admit_request` returns an opaque generation-authenticated
`LunaOnlineWorkerRequest`; saturation consumes neither the request nor a lane.
Every scheduler publication copies the lane route bound before admission, and
the lease dequeues those publications only in global scheduler order. It
exposes only sanitized value publications,
including explicit suppressed-token evidence after a cancellation or deadline
cut.

The lease is the lower-level engine seam for the owned online aggregate.
Production online construction may prepare an exclusive scheduler admission
and its monotonic read before rooted activation, then commit that exact shell
after startup without accepting a request through the lease. A preparation may
also transfer an empty lease for later aggregate-owned admission. Lower fixture
entry points are boundary-restricted to tests; an application cannot pair a raw
`TokenizedRequest` with the aggregate. No decoder, scheduler, process, request
handle, request identity, generation, or raw publication owner escapes.
The service and lease remain thread-confined; copying a current lease reference
does not create independent authority and is forbidden by the aggregate's
exclusive-owner discipline. After an exact healthy terminal is dequeued,
the exact request capability may release its semantic retention and return
only that lane to the fixed free ring while other lanes remain live. The
single-request `retire_terminal_request` method is a compatibility facade and
therefore additionally requires one live lane and a quiescent scheduler.
Worker, scheduler and publication history, lease epoch, and ownership family
remain unchanged.
Instance shutdown remains a separate terminal or empty transition; recovery
is driven by the owning online aggregate.

Online recovery is lease-authenticated: child recovery, exact-flight
retirement, device invalidation, replacement, restart-forbidden drain, and
terminal close retain the same epoch and every live request-lane authority.
Production preparation also receives one explicit immutable
`WorkerRestartBackoffPolicy`. The lease applies its existing injected monotonic
clock only after old-child cleanup and device invalidation, returns a cached
bounded wake instead of sleeping, and authenticates attempts with a private
non-reusing restart generation. Delay doubles to the configured cap; timestamp
overflow, clock rollback, generation exhaustion, or attempt exhaustion selects
the existing deterministic abandon/drain path. A successful spawn does not
reset crash-loop history. Reset additionally requires a committed sequence
strictly after the replacement's authenticated startup predecessor, equal to
the scheduler's current retired sequence, and at or after its stability
deadline.
Successful invalidation marks all live lanes before any later publication can
escape; after restart, each call terminalizes at most one retained waiting
request. WorkerFailed publications carry their original fixed routes and fresh
scheduler generations, so lanes remain exact through ordered retirement.
Clean shutdown is
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

`progress` owns one physical exchange plus at most one exact scheduler-owned
pending plan and advances exactly one logical transition per call: idle,
scheduler-advanced, scheduler-plan-pending, backpressured, started, physical
pending, completion-ready, or committed. `ServiceAdvanced` reports the
allocation-free local recompute-preemption transition and creates no process
flight. `ServicePlanPending(sequence)` reports that the other scheduler A/B
owner holds the exact N+1 obligation; only a scalar sequence is retained, never
a plan alias or boxed option. The pending plan is submitted only after N has
retired from both scheduler and process, and a later call may then stage N+2 in
the newly reusable side. It checks exchange credit before creating the first
scheduler obligation,
records the sequence before beginning transport, and validates exact received
retirement authority inside the private adapter before scheduler mutation. Each received frame is staged
into its paired typed completion owner exactly once; publication pressure moves
the private flight to `Accepted` and returns scalar `ServiceBackpressured`.
Retries from that state neither reread the frame, reopen the writer, advance its
epoch, nor construct a typed error. Scheduler retirement always precedes
process-side retirement, and `Committed` means both succeeded.
`take_next_publication` forwards the scheduler's atomic globally ordered
dequeue, so service adapters never choose between physical token and terminal
rings.

Recovery first closes and reaps the old child or every rank child. The TP owner
takes and authenticates its exact group failure report before cleanup and maps
it to the bounded backend-neutral worker-failure vocabulary. `recover_flight` probes for a
pinned completion, then an exact process reservation, before synthesizing a
bounded worker failure. The process submission is abandoned only after
scheduler retirement succeeds. A received frame that is wire-valid but
semantically invalid is first replaced by the scheduler's exact terminal
failure transaction; only then is received process authority retired. Missing
reservation evidence enters explicit
`ServiceRestartForbidden`; it can never start a replacement. Before a normal
restart, `invalidate_device_state` is a separate retryable phase that fails
active requests and releases their host page/table identities. Waiting requests
remain eligible for the replacement. Operators may instead `abandon_restart`
and incrementally `drain_restart_forbidden`; each call terminally releases at
most one request, and only `ServiceTerminalCloseReady` can close after the drain.
If N is lost while N+1 is pending, recovery retires or fails N first, then
retains a private pending-failure phase until N+1 is failed in sequence.
Publication backpressure or a stale capability consumes neither obligation;
restart and invalidation remain unavailable until both have retired.

The Luna worker-service maintenance surface is the cooperative host path for
recovery and terminal shutdown. `begin_recovery_maintenance` stops plan
construction and starts rooted single-child or whole-group cleanup. Each subsequent progress
call performs at most one child/root transition, one exact N or N+1 retirement,
one device invalidation, or one restart-forbidden request drain. Scalar
pending, advanced, cleanup-required, cleanup-stuck, recovery-ready,
restart-ready, and closed results, plus the distinct fail-close-ready result,
expose no process, root,
plan, request, history, or exit authority. Recovery-ready means child cleanup
has completed and the retained cooperative recovery still owns retirement and
invalidation work; it is not permission to abandon that recovery. Only
fail-close-ready means the online drain path cleared maintenance ownership and
proved that terminal close may proceed. Cleanup-stuck retains the exact lower
child/root cleanup owner after its bounded final-reap interval; readiness and
restart remain false, and later progress polls may still converge without
reconstructing ownership.
`maintenance_remaining_millis` is a nonwaiting lower-bound query during child
cleanup and returns zero during scheduler-only phases.

If replacement startup retains a failed child, `RestartCleanupRequired` enters
the same cooperative owner chain through
`begin_restart_cleanup_maintenance`. Each retry advances one lower child
transition; completion restores RestartReady without running the blocking
compatibility cleanup method. Online leases authenticate this transition to
the same request generation and semantic retention.

Normal recovery preserves a received completion under publication
backpressure, retires physical N before scheduler-only N+1, invalidates device
state, and ends restart-ready. Missing reservation evidence instead drains one
request per turn and proceeds to cooperative terminal close; online mode
continues past intermediate recovery-ready through exact retirement and
invalidation, and reports fail-close-ready only after its terminal drain has
cleared maintenance ownership. Clean
shutdown rejects live requests and either flight before beginning, reaps the
child before closing roots, and releases online semantic retention only after
the closed result. Blocking recovery/close methods remain explicit off-reactor
compatibility paths and reject while cooperative maintenance is active.

`shutdown_clean` rejects both an outstanding flight and any live request. The
terminal `close` path likewise cannot silently discard request, KV, or page
authority. Cancellation and deadline expiry remain ready-state operations and
cannot create new recovery obligations.

This owner implements deterministic failure orchestration and the explicitly
injected host-side worker-replacement backoff. It does not own container
restart policy, health endpoints, model/device loading, or CUDA readiness;
those deployment and instance-level policies remain above this package.

An exact online lease may project the scheduler's two Luna telemetry value
snapshots only after authenticating its production ownership epoch. The bridge
contains scalars only and exposes no scheduler, process, request, plan, page,
or mutable worker authority. A retained stale lease cannot poll a replacement
owner.
