# Bounded radix prefix index

This package owns logical token-prefix discovery only. Physical KV ownership
remains in `kv/page_allocator`; callers coordinate cached-page reference changes
transactionally around successful insertion and eviction.

The first implementation is an uncompressed, startup-preallocated token trie.
Its entry, node, and copied-page arenas never grow after `PrefixIndex::new`.
Lookup, insertion, reference changes, and eviction allocate no general heap
objects. Child discovery scans the bounded node arena, so lookup is explicitly
`O(usable_tokens * node_capacity)`. This conservative representation keeps
ownership and rollback exact; compression or indexed edges require benchmark
evidence and must preserve the same public contract.

Capacity meanings are exact: `entry_capacity` bounds live identity/prefix
records, `node_capacity` bounds aggregate stored trie-token nodes,
`page_capacity` bounds aggregate copied `PageId` cells, and
`max_tokens_per_entry` bounds one inserted or looked-up token sequence. Token
storage is the aggregate node arena; there is no second aggregate token arena.

The identity-root table is also fixed at `entry_capacity`: at most one distinct
identity exists per live entry, so this bound cannot reject a state that the
entry arena could otherwise represent.

Only complete pages are admitted and returned. Identity comparison includes the
model artifact and plan digests, tokenizer digest, cache security scope, and
page-layout version. The index stores token IDs and opaque `PageId` values, never
raw prompts, decoded text, GPU pointers, or device objects.
