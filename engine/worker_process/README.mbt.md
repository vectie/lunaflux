# Isolated-worker supervisor

The legacy transport supervisor owns one exact child channel plus physically
distinct A/B plan and completion frame owners. It admits at most two plans for
deterministic echo/transport fixtures. The production root-bound facade retains
the same preallocated storage but admits exactly one outstanding plan because
the real device child reads, executes, and completes plans serially.

Construction preflights the full bootstrap-source receiver capacity and exact
source digest before spawn, then performs `Configure -> BootstrapSource ->
ParentApprovalAttestation -> Ready`. The child canonically decodes the bounded
source bytes, compares their digest with Configure, consumes the exact one-shot
parent attestation, and returns the exact model identity, bootstrap/source
identities, model generation, predecessor, and runtime limits. The child never
receives the deployment verifier key; the attestation is bound to the admitted
manifest/source, pinned launch identity, generation, and ordinal and cannot be
replayed across a replacement. Startup framed
I/O has its own validated per-prefix/per-payload timeout; steady plan traffic
continues to use the separate I/O timeout.
Incompatible children are closed before publication. If both handshake and cleanup fail, preparation
returns opaque retained cleanup authority so the child is never abandoned.
Submission also rechecks the loaded model generation before any frame write.

A response side remains pinned after wire validation. The worker service
inspects its exact frame once, stages a typed completion, and keeps this
physical authority pinned while scheduler publication is backpressured; retry
does not reread the frame. Only `retire_received` permits that side to be
reused. Foreign, stale, out-of-order,
malformed, partial, timed-out, or closed-channel traffic fails closed. Native
process handles and transport buffers never escape.

The production root-bound facade also exposes one coherent owner-resident
nonblocking exchange. `begin_exchange` canonically encodes and reserves the
exact submitted plan sequence, then begins a pending write without performing
I/O.
Each `progress_exchange` call advances at most one native pending write or read
operation. Write completion arms the transactional read; read completion loads
and validates the response against the retained exact plan before publishing
the existing received side. The caller resolves it by the returned sequence
through `received_for_sequence`. All plan, phase, sequence, and pending-I/O
state stays inside the supervisor; no per-plan exchange owner or transport
capability is allocated or exposed. Internal non-reusing epochs are invalidated
on success, failure, recovery, or close. `has_exchange_capacity` preflights
this one-owner window before a caller creates a scheduling obligation.
The root-bound supervisor is thread-confined and exclusively owned; possession
of that owner authorizes owner-resident progress. The `UInt64` returned by
`begin_exchange` is only sequence correlation evidence, never a transferable
progress capability.

The first successful transport claim fixes a root-bound owner to legacy or
exchange mode. Legacy `submit`/`receive` remain compatibility-fixture APIs and
cannot overlap or follow exchange mode. Any pending transport, timeout,
channel, or validation failure fail-stops the child, invalidates the exchange,
and transitions the root-bound owner to `RootBoundRecoveryRequired`; the exact
reserved sequence remains reconstructible for scheduler failure retirement.

Deterministic worker/rank live-child fixtures traverse the production pinned
admission and are Linux execution tests (run skipped cases with
`moon test --include-skipped` on Linux). Portable tests retain source, capacity,
lifecycle, and closed-authority coverage without pretending Darwin can activate
an admitted executable. Low-level channel tests use a separately linked
test-native process object; no raw-path preparation entry or native symbol is
present in a production package or release archive.
Production service construction uses `prepare_with_approved_executable`, which
first preflights the source and process envelope, then binds each encoded
absolute model/kernel root label to the exact caller-owned approved capability
before any retained pair is activated or child is spawned. Failures retain the
role and payload-safe approved-filesystem cause; the same capability may
satisfy both roles. It then privately duplicates the caller-owned roots. Its root-bound owner
retains that exact opaque pair with the opaque descriptor-pinned executable
admission, process
limits, startup sequence domain, and encoded source. Replacement is
zero-argument and therefore cannot substitute another executable, limit set,
source, or root pair. Restart intentionally does not re-resolve ambient labels:
the retained pinned pair is the continuing authority.

The supervisor retains the immutable encoded source, and replacements receive
the same canonical bytes and pinned root capabilities. The legacy
`worker_echo` child proves two-slot protocol agreement only. The device worker
child reconstructs admitted inputs and readiness, then runs the serialized
steady-state plan/completion loop. Recovery is explicit and ordered: the old
child must first be closed and reaped, validated completions
remain retryable, and each unreturned `WorkerSubmission` must be committed as a
scheduler worker failure before `abandon_submission` retires its exact sequence.
Only after all obligations are retired can `recovery_startup_contract` derive
the non-reusing predecessor for a replacement child. The supervisor does not
silently infer scheduler mutation or discard in-flight work.

Recovery closes only the child; the root pair remains live across replacement.
Instance/service retirement attempts child and root cleanup independently.
Busy root close preserves retry authority, while a consumed close failure
remains bounded evidence without claiming a live pair. Failed replacement-child
cleanup stays inside the root-bound owner and must be retried before restart;
terminal service close is likewise retryable.

The cooperative maintenance API is the nonblocking prerequisite for a reusable
service loop. Generic begin revokes any child pending-frame aliases, invalidates
the proportional exchange snapshot, and makes transport progress unavailable.
It deliberately retains A/B side and sequence state as exact scheduler
retirement or abandonment obligations; it never silently commits those
external effects. Each generic progress call delegates at most one transition
to the private process owner and copies only scalar pending, advanced,
complete, cleanup-required, cleanup-stuck, or exit evidence. Cleanup-stuck
means the bounded final-reap interval expired while this exact owner remains
available for later nonblocking polls; it is not resource abandonment. No native process, buffer,
maintenance token, root capability, or internal-process result escapes.

Root-bound recovery and terminal close share that exact child maintenance but
have distinct typed starts. Recovery keeps the approved root pair live after
the child is fully reaped so replacement can use the same pinned authorities.
Terminal close also keeps roots live through the progress call that completes
child handle close; only a later progress call may attempt root-pair close.
Busy roots preserve the root-close phase for retry. A consumed root close
failure records bounded evidence, marks no false live authority, and makes the
next progress idempotently complete. Existing blocking recovery and close APIs
remain compatibility paths only while cooperative maintenance is inactive.

A failed replacement child has a distinct cooperative start,
`begin_replacement_cleanup_maintenance`. It delegates the retained child to
the same single-transition shutdown engine, keeps the exact root pair and
predecessor live, and restores RootBoundRecovering only after the child handle
is closed. Blocking `retry_replacement_cleanup` rejects while that maintenance
authority is active.

Both maintenance layers are thread-confined serialized owners. Remaining-time
queries delegate to the child monotonic bound without I/O and return zero for
the retained cleanup-stuck state. Recovery, replacement, and root close remain
unavailable until a later poll reaps and closes that same child. The release-C gate
pins direct and transitive supervisor progress as allocation-free apart from
typed exceptional error construction, and source checks freeze root-after-child
ordering and the opaque public surface. TERM/KILL timing, EINTR, delayed reap,
PID invalidation, and native close ambiguity remain proven by the lower
`internal/process` sanitizer and are exercised here through the same single-
transition delegation rather than a parallel cleanup engine.
