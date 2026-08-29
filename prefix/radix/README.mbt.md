# Bounded radix prefix index

This package owns logical token-prefix discovery only. Physical KV ownership
remains in `kv/page_allocator`; callers coordinate cached-page reference changes
transactionally around successful insertion and eviction.

The index is a compressed, startup-preallocated token radix. Each node owns a
nonempty linked label in the fixed token-cell arena; unary nonterminal nodes are
merged after eviction. Full-page anchors live on token cells, so descendants
share structural page runs without copying another cached-page reference.
Entry, node, token, page, identity, and security-scope arenas never grow after
`PrefixIndex::new`. Direct lookup, publication, reference changes, and eviction
allocate no general heap objects. Separate deterministic AVL indexes resolve
the full exact salted root identity and each `(root, parent, first_token)` edge,
so lookup is `O(log roots * exact_identity_compare + traversed_edges * log
nodes + compared_tokens)`. Index payloads may move during AVL deletion, while
the radix root and node slots named by generations remain stable.

Capacity meanings are exact: `entry_capacity` bounds live identity/prefix
records, `node_capacity` bounds compressed nodes, `token_capacity` bounds
aggregate label-token cells, `page_capacity` bounds distinct structural
`PageId` anchors, `max_scope_bytes` bounds one startup-copied identity scope,
and `max_tokens_per_entry` bounds one published or looked-up token sequence.
The checked startup formulas are `25 * entries + 21 * nodes + 5 * tokens +
10 * pages` integer cells, `entries + nodes + tokens + 4 * pages` boolean
occupancy cells, `(entries + 1) * max_scope_bytes` byte cells, and `3 * entries`
identity-reference cells. The additional scope row is pre-mutation install
scratch. The constructor revalidates the same formulas before any raw
multiplication or backing allocation.

The identity-root table is also fixed at `entry_capacity`: at most one distinct
identity exists per live entry, so this bound cannot reject a state that the
entry arena could otherwise represent.

Only complete pages are admitted and returned. A planned `try_plan_publish`
outcome yields an opaque one-shot `PrefixPublishPlan`, snapshots the exact
token/page buffers into fixed scratch, binds the exact opaque request key inside
the plan, and marks the page references a physical owner must preflight and
retain before metadata commit.
`publish` accepts no substitute key and reauthenticates that authority, the
exact buffers, priority, root shape, and adoption set; `abort_publish` cancels
the transaction before physical references are changed. Publication-plan
generations never wrap: terminal exhaustion disables further transactional
publication while existing lookup and eviction remain valid. The index itself never
mutates the page allocator. Identity comparison includes the model artifact and
plan digests, tokenizer digest, cache security scope, and page-layout version.
The index stores token IDs and opaque `PageId` values, never raw prompts,
decoded text, GPU pointers, or device objects.
The production API enforces semantic cache permission independently of its
caller: `Disabled` lookup is an empty miss, `ReadOnly` may look up but cannot
open a publication plan, and only `ReadWrite` may publish.

The warmed scheduler calls `try_plan_publish`, whose opaque
`PrefixPublishStart` reports planned, duplicate, capacity-exhausted, or rejected
through scalar predicates. Duplicate prompts, full metadata arenas, oversized
configured scopes, and terminal plan generation therefore construct no error.
The scalar start plus opaque-plan transaction is the only publication API.

Duplicate physical pages are rejected before mutation by copying `PageId`
values into fixed startup scratch, heapsorting by exact `(index, generation)`,
and scanning adjacent values. This preserves caller order, uses no hash-only
authority or allocator-universe assumption, and costs `O(pages log pages)`
comparisons and swaps while the distinct logical page-capacity envelope remains
unchanged.

Eviction is also two phase: `try_plan_eviction` returns a scalar start from
which a planned outcome yields an opaque one-shot `PrefixEvictionPlan`. A fixed
intrusive min-heap selects the lowest priority,
then oldest-recency, then
lowest-index zero-active-reference entry, and snapshots the exact page
identities that would become unowned. The scheduler authenticates their cached
references before `commit_eviction`; commit reauthenticates the plan and output
buffer before its first mutation, and only afterward does the scheduler release
those physical references. `abort_eviction` preserves metadata after a failed
physical preflight. Active entries are never candidates, overlapping publish
and eviction plans are rejected, and shared anchors remain until their final
structural user is removed. Eviction-plan generations also never wrap;
terminal exhaustion preserves all existing metadata and lookup authority while
disabling further mutation.
Recency-clock exhaustion compacts used entries through fixed startup scratch
heapsort ordered by exact `(recency, entry index)`, then rebuilds the victim
heap. This rare maintenance step is deterministic `O(entries log entries)` and
allocates no steady-state storage; ordinary victim updates remain `O(log E)`.
The scheduler similarly uses `try_plan_eviction`; an empty cache and terminal
eviction generation are scalar `PrefixEvictionStart` outcomes. The scalar start
plus opaque-plan transaction is the only eviction API. `PrefixRequestKey`,
caller-owned fixed buffers, and one-shot publication/eviction authorities are
the single authoritative surface; no allocating logical compatibility path can
bypass physical cached-reference ownership.
