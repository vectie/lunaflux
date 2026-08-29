# Tensor-parallel rank-local KV plan

`engine/tensor_parallel_kv_plan` is a startup-only, authority-free projection
from one semantic model plan, its current worker model generation, an opaque
tensor-parallel runtime admission, and an exact rank-ordered set of admitted
device plans into persistent rank-local K/V arena geometry.

The scheduler remains the sole owner of `PageAllocator`, `BlockTableArena`, and
every logical page-table transition. This package only consumes an
allocator-issued `PageId`. It maps that exact index and generation to the same
local page slot on every rank and returns scalar byte regions or offsets. The
immutable plan and its fixed rank-ceiling array are fully created at startup;
successful page and element projections do not resize collections or acquire
resources.

Admission cross-authenticates every device-plan rank, world, process ordinal,
and target against the existing runtime admission; that admission is the sole
source of per-rank KV reservations. The current nonzero model generation is
stored and re-authenticated on every scalar projection. Global KV heads
must divide the world size exactly. V1 then uses
`global_kv_heads / world_size` heads per rank while retaining the canonical
layer-major split-key/value BF16 layout. Checked arithmetic proves that local
logical page bytes partition the full logical page exactly. For a multi-rank
plan, each local arena must also be strictly smaller than the corresponding
full-head arena; small aligned geometries that would reserve a fully replicated
arena are rejected rather than misreported as sharded.

The result contains no device address, allocation, context, stream,
communicator, scheduler mutation method, or backend handle. It does not spawn
workers or change the one-scheduler/one-worker-per-device architecture.

`rank_binding` publishes an output-only proof for later worker bootstrap. Its
opaque SHA-256 contract digest uses fixed little-endian scalar encoding and
binds model identity/generation, rank/world/ordinal/target, complete local-head
geometry and strides, logical page capacity, arena bytes, and the authoritative
rank KV reservation. There is no public constructor for either proof or digest.
