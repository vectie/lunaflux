# Fixed-capacity KV block tables

`BlockTableArena` owns only bounded host metadata mapping each request's
logical KV page positions to canonical `PageId` values. It does not own device
memory and never changes a page allocator's active or cached reference counts.

The scheduler owns the cross-package transaction:

1. allocate or retain physical pages through `PageAllocator`;
2. append their copied identities with `append_page`, `append_run`, or
   `append_pages`;
3. on rollback or cancellation, drain removed identities with `truncate`,
   `reset`, or `reset_and_release` into a reusable `BlockTablePageBuffer`;
4. release the corresponding active or cached references explicitly.

Appending does not validate allocator residency. Releasing a non-empty table
is rejected so outstanding physical release obligations cannot be discarded.
Every mutation uses startup-preallocated storage; allocating snapshots and
invariant checks are diagnostics only.

`BlockTableIdStorage` is the canonical fixed-capacity container for scheduler
metadata that must retain opaque table identities without optional-value
boxing. It copies identities inline behind an occupancy bitmap, never exposes
its inert empty sentinel, and does not acquire or release table ownership.

`BlockTableAllocationCheckpoint` provides an authenticated transaction for
table allocation and reversible mapping suffixes. It records allocation FIFO
links, prior generations, and only first-touched mapping baselines in storage
allocated by `BlockTableArena::new`; checkpoint is O(1), while rollback scales
with allocations and touched mappings. Appends may target existing or newly allocated
tables while the checkpoint is open. Before rollback, the scheduler must call
`detach_suffix_for_checkpoint` for every appended table, then roll back the
table checkpoint, and only then roll back physical page allocation. The arena
verifies all existing tables are back at their exact baseline and every new
provisional table is empty. Release, truncate, and reset operations are rejected
while the checkpoint is open.
Release native-object inspection shows no allocation call in the successful
checkpoint, detach, commit, or rollback closure; failure paths may allocate.

Release native-object inspection shows no allocation call on successful
`BlockTableIdStorage` read, write, clear, or occupancy-check paths. Failure
branches may allocate checked error values. A stronger zero-general-heap claim
still depends on the Phase 3 allocation-instrumentation gate.
