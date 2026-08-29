# Private generic tensor-collective owner

The device package owns the generic rank-local collective contract because it
is the first layer that owns the existing `Context`, `Allocation`, and `Stream`
wrappers. The contract is model-neutral: a SHA-256 identity, rank/world, and
fixed parallel scalar site arrays copied at startup. It imports no model family,
model identity, scheduler, rank-group, configuration, or filesystem authority.

Runtime admission requires an explicit caller range whose minimum is at least
NCCL 2.14.3. The loader must resolve nonblocking communicator initialization,
async-error polling, exact all-reduce/all-gather symbols, and CUDA stream query.
`probe_tensor_collective_runtime` exposes only typed availability and the
observed version for product diagnostics. It creates no group ID, communicator,
context, stream, rank, or device allocation; exact version admission remains a
separate operation.

The thread-confined owner creates and retains one communicator for its exact
context, rank, contract, nonzero group generation, and authenticated predecessor
plan sequence. Rank/world, widths, contract digest, maximum live query tokens,
and fixed site identities are copied once at startup. Each hot-path submit
therefore accepts only generation, plan/global/site sequence, operation/kind,
live query tokens, and existing device resources; it constructs no
digest-bearing claim object. Checked Int64 multiplication derives transfer
elements from `live_query_tokens * per_token_width`, so a paged step still emits
one collective per site. Vocabulary all-gather is rank-major: each rank sends
`live_query_tokens * local_width` elements and receives
`live_query_tokens * full_width` elements.

Duplicate, stale, skipped, or substituted scalars latch failure. Startup and
submitted work are polled without blocking. Only completed poll publishes
sequence progress; a failed poll retains resource guards until abort. Only
abort can consume a failed owner; healthy close consumes destroy. A non-success
terminal native result requires worker-process termination even though cleanup
authority is consumed.

`internal/tensor_parallel_collective` is the sole production constructor of a
contract. It binds an independently expected model identity to an immutable
Llama rank plan and derives the canonical contract digest. OCI labels, ambient
configuration, and caller-provided vendor diagnostics are never authority.

Fake-native and sanitizer evidence does not prove physical NCCL/CUDA startup,
multi-rank rendezvous, dead-rank timeout/drain, BF16 numerics, leak freedom on a
real driver/runtime, or throughput. Those remain Linux multi-GPU campaign gates.
