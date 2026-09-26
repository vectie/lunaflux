# Cold KV page payloads

`Plan` derives a bounded host footprint and pure scalar copy mapping from a
canonical device KV layout. A host slot stores a complete page's K and V across
every layer, with device alignment gaps excluded. `Cache` composes this plan,
the residency state machine and a startup-owned `device.TransferPool`.

Preparation rejects a mismatched allocation size before acquiring resources.
After native acquisition begins it always returns cleanup authority, including
partial construction failure. Check `is_prepared` before use;
call `close` even after failed preparation. Spill and restore return an opaque
operation and use nonblocking `poll`. A cancelled restore still drains DMA but
does not publish `DeviceReady`. Enqueue/poll failure poisons the cache; explicit
close drains every pending lane and preserves retry ownership if it fails.

The worker retains device-page reservations until completion or successful
drain. A cached entry is an opaque payload identity, not a token-prefix match.
The higher-level prefix owner must establish model/tokenizer/layout/scope
identity and exclude active readers before spilling. These bytes never move
directly into a graph pointer: restores target the fixed persistent arena.

The paged executor's opt-in startup methods bind this owner to its exact layout
and fence staging while transfers are outstanding. Production scheduler/worker
wire admission of logical cold-prefix entries remains a separate integration
gate, tracked in [STREAMING.md](../../docs/STREAMING.md).
