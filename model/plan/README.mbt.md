# Semantic model plans

`model/plan` owns architecture-neutral model graph semantics. Dense decoder
attention never derives persistent-cache ownership from operation order:
`KvLayerId` binds each rotary and causal-attention operation to the matching
layer in `KvCacheGeometry`.

`ModelKvExecution` distinguishes two complete graph contracts:

- `StatelessFullContext` supplies no runtime metadata and reads or writes no
  persistent KV state.
- `PagedKeyValue` supplies `StepCounts` to every operation so kernels never
  infer live rows from the maximum profile. Rotary additionally consumes query
  positions; attention consumes the complete packed query-row/page-table
  contract and declares `KvCacheReadWrite` for one exact layer. Counts bound
  every activation, CSR offset, and payload read.

Validation requires exact runtime-input order, mode consistency, geometry-
wide one-to-one attention-layer coverage, and monotonically ordered layer
identities. Positioned rotary and paged attention have distinct semantic
capability IDs from their full-context counterparts.

These values are semantic evidence only. They own no block table, device
allocation, byte layout, kernel ABI, or execution support. Kernel catalog v1
therefore rejects paged plans before shape matching. Catalog v2 can resolve the
full live semantic graph, while its current launch-contract admission remains
deliberately limited to positioned rotary and paged attention; full-graph
execution remains fail closed.
