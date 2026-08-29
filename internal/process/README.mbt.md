# Private child-process channel

This package is the only owner of LunaFlux's native process ABI. It spawns one
exact executable without a shell and gives the child a private inherited
full-duplex socket on standard input/output. The parent endpoint is therefore
authenticated by construction; no filesystem socket or ambient listener is
created.

All public transfers are exact length-delimited frames over caller-owned fixed
storage. The two four-byte prefixes are allocated once at spawn. Parent-side
online callers begin owner-resident transfers and ask that same child for one
nonblocking native `send` or `recv` per progress call. One pending read and one
pending write may coexist, while a second transfer in either same direction is
rejected. All mutable pending state and retained buffer references live in the
long-lived `ChildProcess` or `InheritedChannel`;
begin and progress create no per-frame token or state allocation.
`EAGAIN`, `EWOULDBLOCK`, and `EINTR` remain pending. One absolute monotonic
deadline covers the complete prefix and payload and is never reset by partial
progress. `begin_write_frame_until` and `begin_read_frame_until` let an ordered
command/response pair share the caller's exact cut; an expired read admission
poisons the channel before consuming response bytes. Absolute cuts remain
bounded to the package's validated 600000 ms policy. A transactional read
requires separate caller-owned staging storage
and does not copy into its published destination until the complete declared
frame has been received and validated.

Cooperative shutdown is likewise owner-resident. `begin_shutdown_maintenance`
revokes retained frame aliases and captures the first absolute monotonic grace
interval without making a process-system call. Successive progress calls
perform at most one of: write half-close, one `waitpid(WNOHANG)`, TERM, one
poll, KILL, one poll/reap, or reaped-descriptor close. Grace and escalation
deadlines use the validated shutdown timeout and are never reset by ordinary
pending polls. After the KILL grace expires, one separately bounded final-reap
interval continues `waitpid(WNOHANG)` without another signal. A backward
monotonic sample, including a remaining-time result larger than the configured
phase timeout, is rejected as cleanup-required and cannot extend either
interval. EINTR consumes one poll and remains pending.

An idle supervisor may call `observe_idle_exit` only while both transfer
directions and cooperative maintenance are inactive. It performs one
`waitpid(WNOHANG)`: a live result is nonmutating, while an exact exit is reaped
once, copied into the existing maintenance evidence, and moves the owner
directly to reaped-descriptor close. The owner becomes failed immediately so
no later frame operation can mistake the dead child for a ready worker.

The exact reaped exit kind/code is packed into one scalar native result, then
stored transactionally in the same `ChildProcess`; polling creates no `Ref`,
collection, token, or second child capability. Successful reap sets the native
PID to invalid before any later signal or close, and an already-reaped poll
does not call `waitpid` again. ECHILD does not fabricate exit evidence or
discard the retained owner. Expiry of the final-reap interval returns the
distinct scalar `CleanupStuck` disposition and never reports unbounded Pending.
The same owner remains in a nonblocking WNOHANG retry state: later progress may
observe the exact reap and continue to descriptor close without reconstructing
or abandoning authority. A native wait, signal, monotonic-clock, or close
failure remains the separate cleanup-required result. Reaped close consumes
descriptor authority even when the close result is ambiguous; its sole retry
is the idempotent reaped-close phase.

The blocking `shutdown_write`/`wait`/`terminate`/`close` compatibility facade
is unchanged when no cooperative maintenance is active. Those operations are
rejected while maintenance owns the lifecycle, preventing a second cleanup
path from signaling or reaping the same child. Cooperative maintenance is a
serialized, thread-confined owner protocol; it is not safe for concurrent
MoonBit access from multiple OS threads.

MoonBit fixed arrays remain mutable aliases: the child retains the exact array
references but cannot prevent their holder from mutating them. Retained
references are cleared on completion, poison, and close. Therefore the
child and all retained source, staging, and destination arrays must be
confined to one higher-level owner and not exposed or mutated while active.
Read begin rejects destination/staging identity aliasing. This confinement is
part of the API contract rather than a claim of language-enforced ownership.
Because opposite directions may overlap, every entry point rejects exact
physical identity between an active write source and either active read array
before performing I/O. Callers must still avoid any aliasing not represented
by exact fixed-array identity. Both owners are single-event-loop confined and
are not thread-safe. Alias rejection does not poison or revoke the already
active opposite direction.

Short I/O,
EOF, timeout, invalid range, oversized declaration, and lifecycle failures
permanently poison stream alignment; only termination, wait, and close remain.
Failures are typed without paths, payloads, file descriptors, PIDs, or
`errno`. The compatibility exact loop uses monotonic
deadlines and retries interruption. Explicit `close` shuts the channel, kills
a still-live child, and reaps it; the external-object finalizer is only a
last-resort cleanup guard.

This is transport plumbing, not the worker protocol or supervisor. Higher
packages own framing, plan/completion authentication, readiness, restart, and
request-state recovery. `InheritedChannel` is the symmetric child-side owner;
its cooperative begin/progress APIs use the same independent-direction fixed
state, one-syscall progress, clean-prefix EOF sentinel, and fail-stop rules
without exposing file descriptors. Blocking framing methods remain compatible
when their own direction has no pending transfer. The current native branch is
POSIX; Windows process
creation and physical long-running soak/leak evidence remain open.

The startup-only child uses `expect_clean_eof` after readiness. It accepts only
an exact zero-byte parent half-close; any next-frame byte, including a partial
length prefix followed by EOF, poisons the channel and fails shutdown.

`spawn_with_approved_roots` borrows a reusable ordered model/kernel lease pair
and one already authenticated executable capability across native spawn. On
Linux it maps only fixed child descriptors 0-5, closes the complete descriptor
range from 6 through `UINT_MAX`, uses fixed argv0 `lunaflux-worker`, supplies an
empty environment, and calls `fexecve(5)` without reopening a path. The parent
blocks all signals before `fork`; while they remain blocked the child resets
every catchable disposition, installs the empty final mask, and only then
executes. Parent mask restoration is mandatory, and failure kills and reaps
the unpublished child. No root locator or executable path crosses argv or env.
Unsupported platforms fail closed. Raw pathname spawn declarations exist only
in white-box fixture source and are absent from the generated package API.

`exit_failure` is the child command's final payload-safety boundary. It uses
`_exit(1)` only after explicit owner cleanup so a rejected bootstrap terminates
nonzero without runtime diagnostics or buffered writes reaching the private
channel.

`scripts/validate-process-allocations.sh` inspects release-generated C for the
parent and inherited begin/progress functions. It permits typed `ProcessError`
construction on exceptional branches but rejects per-frame heap, array,
string, ref, or token construction; a dedicated fixed-array allocation keeps
the gate positively controlled.

`scripts/validate-process-maintenance-allocations.sh` applies the same
positive-controlled release-C check to cooperative begin/progress, their
transitive clock/poll/exit helpers, and scalar queries. The separate system
clang ASan+UBSan probe fixes one-call EINTR, exact normal/signal exits,
ECHILD retry, PID invalidation, double-reap prevention, TERM/KILL selection,
and consumed-close retry behavior. These are native POSIX/toolchain-sensitive
claims; they do not claim an off-reactor executor or reusable server yet.
