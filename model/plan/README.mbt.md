# Semantic model plans

`model/plan` owns architecture-neutral model graph semantics. Dense decoder
attention never derives persistent-cache ownership from operation order:
`KvLayerId` binds each rotary and causal-attention operation to the matching
layer in `KvCacheGeometry`.

`ModelKvExecution` distinguishes two complete graph contracts:

- `StatelessFullContext` supplies no runtime metadata and reads or writes no
  persistent KV state.
- `PagedKeyValue` supplies explicit live step counts, query positions, packed
  query-row and page-table offsets, sequence lengths, and physical page
  indices. Counts bound every CSR offset and payload read. Each attention
  operation declares `KvCacheReadWrite` for one exact layer.

Validation requires exact runtime-input order, mode consistency, geometry-
wide one-to-one attention-layer coverage, and monotonically ordered layer
identities. Positioned rotary and paged attention have distinct semantic
capability IDs from their full-context counterparts.

These values are semantic evidence only. They own no block table, device
allocation, byte layout, kernel ABI, or execution support. Kernel catalog v1
therefore rejects paged plans before shape matching; a later catalog version
must name every runtime and persistent-state operand before cached execution is
admitted.
