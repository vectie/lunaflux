# Fixed-page KV metadata allocator

This package owns bounded host metadata for physical KV page identities. It
preallocates page generations, distinct active-request and cached-prefix
reference counts, residency state, and an intrusive FIFO free queue.

Allocation creates one active reference. Sharing with another live request
retains an active reference. Publishing a page into a prefix cache retains a
cached reference before the request releases its active owner. A prefix hit
retains another active reference without consuming the cached owner. Cache
eviction releases the cached owner. A page becomes reusable only when both
owner classes reach zero.

`PageId` is a `#valtype` fixed-width value combining an index and generation.
On the native backend this is the representation invariant that keeps the two
components in registers or inline fixed-array slots rather than allocating a
page object. Reuse increments the generation; stale identities are rejected. A
slot at its configured terminal generation is retired instead of wrapping and
aliasing an old identity.

`PageRunBuffer` is caller-owned reusable output storage. Successful one-page,
run, retain, and release operations do not grow collections or return mutable
internal arrays. `debug_snapshot` and `debug_check_invariants` are explicit
startup/test diagnostics and may allocate.

Source and native-object inspection show no collection growth and no allocation
call on the successful native allocation/run/reference paths. Failure paths may
allocate a checked error value. The stronger release claim of zero general-heap
traffic remains subject to Phase 3's named hot-path allocation-instrumentation
gate.

This package does **not** allocate physical device KV memory, define the model
KV layout, upload request block tables, or launch paged-attention kernels. Those
Phase 3 device-arena and page-table integrations are later workstreams.
