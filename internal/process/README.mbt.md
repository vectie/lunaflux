# Private child-process channel

This package is the only owner of LunaFlux's native process ABI. It spawns one
exact executable without a shell and gives the child a private inherited
full-duplex socket on standard input/output. The parent endpoint is therefore
authenticated by construction; no filesystem socket or ambient listener is
created.

All public transfers are exact length-delimited frames over caller-owned fixed
storage. The two four-byte prefixes are allocated once at spawn. Parent-side
online callers can begin opaque epoch-bound frame capabilities and advance
them with one nonblocking native `send` or `recv` per `progress` call.
`EAGAIN`, `EWOULDBLOCK`, and `EINTR` remain pending. One absolute monotonic
deadline covers the complete prefix and payload and is never reset by partial
progress. A transactional read requires separate caller-owned staging storage
and does not copy into its published destination until the complete declared
frame has been received and validated.

MoonBit fixed arrays remain mutable aliases: the capability retains the exact
array reference but cannot prevent its holder from mutating it. Therefore the
capability and all retained source, staging, and destination arrays must be
confined to one higher-level owner and not exposed or mutated while active.
Read begin rejects destination/staging identity aliasing. This confinement is
part of the API contract rather than a claim of language-enforced ownership.
Only one pending transfer in either direction may exist for a child, so a
read destination or staging buffer can never alias a concurrently transmitted
source. The worker exchange is sequential write-then-read.

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
it uses the same fixed framing and fail-stop rules without exposing file
descriptors. The current native branch is POSIX; Windows process
creation and physical long-running soak/leak evidence remain open.

The startup-only child uses `expect_clean_eof` after readiness. It accepts only
an exact zero-byte parent half-close; any next-frame byte, including a partial
length prefix followed by EOF, poisons the channel and fails shutdown.

`spawn_with_approved_roots` borrows a reusable ordered model/kernel lease pair
across native spawn, maps only fixed child descriptors 3 and 4, uses fixed
argv0 `lunaflux-worker`, supplies an empty environment, and closes descriptor
2 plus unrelated inherited descriptors. It carries no root locator through
argv or env. Legacy `spawn` remains only for fake/compatibility workers with
its existing argv/environment behavior.

`exit_failure` is the child command's final payload-safety boundary. It uses
`_exit(1)` only after explicit owner cleanup so a rejected bootstrap terminates
nonzero without runtime diagnostics or buffered writes reaching the private
channel.
