# Private child-process channel

This package is the only owner of LunaFlux's native process ABI. It spawns one
exact executable without a shell and gives the child a private inherited
full-duplex socket on standard input/output. The parent endpoint is therefore
authenticated by construction; no filesystem socket or ambient listener is
created.

All public operations are exact, bounded fixed-buffer transfers. Short I/O,
EOF, timeout, invalid range, and lifecycle failures are typed without paths,
payloads, file descriptors, PIDs, or `errno`. The native loop uses monotonic
deadlines and retries interruption. Explicit `close` shuts the channel, kills
a still-live child, and reaps it; the external-object finalizer is only a
last-resort cleanup guard.

This is transport plumbing, not the worker protocol or supervisor. Higher
packages own framing, plan/completion authentication, readiness, restart, and
request-state recovery. The current native branch is POSIX; Windows process
creation and physical long-running soak/leak evidence remain open.
