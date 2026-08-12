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
