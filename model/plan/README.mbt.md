# Semantic model plans

`model/plan` owns architecture-neutral model graph semantics. Dense decoder
attention never derives persistent-cache ownership from operation order:
`KvLayerId` binds each rotary and causal-attention operation to the matching
layer in `KvCacheGeometry`.

Every `ModelPlan` also owns one mandatory typed view of a complete
`ModelNumericSchema`. Its generic tensor table declares every referenced
parameter and every scale, zero-point, or codebook tensor with an exact shape
and per-tensor storage contract. Typed metadata references wrap the existing
`TensorRef`; no second integer identity exists. One per-operation execution
contract is required at every operation index. Construction rejects missing
tensors, metadata tensors used as direct semantic operands, incompatible
storage/encoding and compute representations, mixed tensor-input requirements,
tensor shapes that do not match their exact operation input position, unused
parameters, orphan metadata, and producer-to-consumer activation dtype changes.

`model/numeric_contract` is the only canonical numeric-schema encoder. Its schema digest covers
tensor order, roles, shapes, representation bytes, exact metadata ordinals,
operation order, and execution-contract bytes. `ModelPlan` is the sole plan-
identity authority: its constructors accept verified content plus explicit
opaque construction limits, validate and own the complete semantic graph, then
stream a versioned canonical encoding directly into SHA-256. That encoding
binds execution mode, every shape constraint, KV geometry, workspace, every
ordered operation field, derived capability order, final output, and the raw
numeric-schema digest. Callers cannot supply a `PlanDigest` or `ModelIdentity`.

The fixed `lunaflux.model-plan.v1\0` encoding uses explicit byte tags,
little-endian 64-bit integers, and raw IEEE-754 `Double` bits. Exact byte-count
preflight occurs before numeric mapping or plan-owned proportional storage;
operation, tensor, aggregate-input, per-operation input, runtime-input,
per-operation output, and canonical-byte ceilings fail closed. Resource-limit
failures intentionally precede ordered semantic graph errors: in particular,
the hard one-output envelope prevents a hostile output list from being walked
before the later ordered arity check reports semantic mismatches inside the
admitted envelope.
The same streaming walk must produce the preflighted byte count before the plan
can mint its identity. The private SHA-256 stream owns one reusable block and
one reusable message schedule per construction; transforms allocate no
per-block scratch. Family-level legacy digest helpers remain only as
immediate slice-B removal debt and are not plan-construction authority.

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
