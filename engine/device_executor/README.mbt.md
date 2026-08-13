# Static device executor

This package prepares and runs one exact Phase-1 full-sequence profile. Startup
validation joins the immutable model, memory, profile, kernel-contract, artifact,
weight, and device-target identities before opening native resources. It then
allocates reusable token and activation/workspace storage, loads AOT modules and
entry points once, and builds immutable launch arguments or vendor GEMM plans.

`PreparedDeviceExecutor` is deliberately single-owner and thread-confined. The
device worker must serialize `dispatch`, `copy_terminal_row`, and `close`; none
may run concurrently or reentrantly. Its mutable lifecycle fields are not
synchronization primitives. This matches LunaFlux's one-owner-per-device worker
architecture and avoids adding a lock to the token-step path.

The caller retains ownership of the already-open context and device weights.
Preparation acquires an opaque lifetime lease on the exact weight allocation,
so an early caller close fails retryably with `Busy` instead of invalidating
prepared arguments. Executor close invalidates dispatch first, then releases
the lease with its other resources. Close is explicit, reverse ordered,
idempotent after success, and retryable after a child close failure. A failed
child close blocks only its parent while independent siblings continue
closing. Once close begins, dispatch and output copy remain unavailable even
when cleanup fails; the owner may retry close.

Preparation returns `PreparationOutcome`. `Ready` carries the executor. If
construction fails after native resources have opened and the first cleanup
attempt also fails, `CleanupRequired` carries an opaque
`FailedDevicePreparation`. Its bounded cause and cleanup failure are inspectable,
and `retry_close` retains explicit release authority until cleanup succeeds.
No opened resource is abandoned to GC-only cleanup.

This is not a KV decode executor. `FullRecompute` remains a complete stateless
full-sequence rerun, and terminal logits are copied one bounded row at a time for
Phase-1 correctness work.
