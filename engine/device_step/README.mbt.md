# Reusable device-step staging

This package translates one authenticated `SubmittedSchedulePlan` into seven
fixed-capacity little-endian Int32 device operands:

1. counts: prefill rows, decode rows, total rows, query tokens, page entries;
2. token IDs;
3. absolute token positions;
4. CSR query-row offsets, including the terminal token count;
5. full sequence lengths after each row;
6. CSR page offsets, including the terminal page count;
7. physical page indices.

The isolated-worker path accepts the equivalent canonical
`ValidatedPlanFrame` through `stage_frame`. It validates sequence, generation,
counts, tokens, capabilities, CSR tables, page-generation structure, and
physical-page bounds directly from fixed wire storage before host mutation or
device upload. No scheduler heap-owner capability is required in that path;
the in-process `stage` method remains for the compatibility boundary.

Rows retain protocol order: prefill first, then decode. Prefill positions are
`sequence_token_start + local_query_index`; decode has one query at
`sequence_token_position`. Every row carries exactly
`ceil(sequence_length / tokens_per_page)` pages. A positive `PageId`
generation is only a structural check here; generation and active-residency
authorization occurred in `scheduler/core` before this exact-epoch submitted
plan was committed, and the scheduler retains active references through
completion. Generations are deliberately stripped before upload, so the GPU
receives only range-checked physical indices. A future wire transport must
carry equivalent generation and page-lease authority.

The owner is confined to the worker that owns its borrowed device context.
Host buffers and seven matching device allocations are created once at
startup. A stage performs a complete read-only preflight before writing host
storage, including plan-sequence and stage-epoch authorization. The startup
limits bind the loaded model's exact maximum sequence length and token ID;
protocol admission is never trusted as the only semantic envelope. Staging
then uploads an invalid zero counts header, all payloads, and the real counts
header last. Any upload failure poisons the owner, preventing kernel binding,
finish, or another stage until deterministic close.

This staging slice exposes neither launch arguments nor any allocation alias.
A future executor in this owner package must mediate launch under lifecycle
authorization. `finish` retires the exact stage and advances the accepted
sequence predecessor; replacement owners may start from an explicit committed
predecessor seed. No close authority escapes.

Startup may also admit a `PagedKvExecutionBlueprint` for one exact paged
kernel profile. This inert value cross-checks the semantic model and KV layout,
device-step capacities, catalog-v3 launch contracts, activation/workspace
memory plan, and content-addressed artifact entries. Its layer-ordered rotary
and attention steps retain only semantic references, bounded launch metadata,
and scalar offsets/spans into separately owned descriptor, activation, and KV
arenas. It opens no native resource and exposes no allocation, device address,
kernel argument, or launch method.

The narrow blueprint accepts only a `KvSubsequence` launch-contract set.
`admit_paged_graph_execution_blueprint` separately requires `FullGraph` and
binds token IDs, exact verified weight regions, every activation/workspace
region, and persistent KV. Scope mismatches fail closed; both results remain
inert and expose no resource authority.

`admit_device_worker_bootstrap` hashes the complete admitted full-graph
blueprint and artifact evidence into a bounded canonical binary schema. The
digest covers model and plan identities, the bounded process-visible device
ordinal, target, catalog-v3 ABI, exact descriptor/scalar limits, live profile,
KV geometry, arena sizes, module digests, symbols, entry points, launch
geometry, and every physical operand binding. Startup frame v3 carries that
digest and ordinal across Configure/Ready. Protocol and inference envelopes
must fit the hashed device limits; the manifest itself owns no module bytes or
device resource.

`prepare_paged_graph_executor` is the first owner-mediated synchronous
execution path for a `FullGraph` blueprint. It validates all immutable evidence
before resources, leases the caller-owned weight allocation, and privately owns
its descriptor buffers, activation/workspace arena, persistent KV arena,
stream, modules, functions, and prebuilt arguments. None of those resources or
arguments escape. Exact opaque capabilities enforce `stage -> execute ->
sample_completion -> finish`; `stage_frame -> execute ->
sample_completion_frame -> finish` is the equivalent isolated-worker path.
The latter authenticates the exact retained frame owner and epoch and appends
the canonical completion while deliberately leaving the writer open. Its
aggregate owner must finish the executor before submitting the writer, so a
finish failure cannot publish a completion. Every launch synchronizes, and any
partial launch failure permanently poisons execution and descriptor state. The
completion phase reads only each producing row's retained BF16 vocabulary
logits into startup-owned fixed storage, rejects non-finite values, applies
greedy or counter-addressed stochastic selection using the row's exact
`(sampling seed, output index)`, and appends to the exact frame-bound completion
writer. All reads and selections finish before the first completion entry is
written. The writer remains open for explicit submit after executor finish or
abort after any failure; a readback or invalid-logit failure poisons the
executor.
Close invalidates execution first and then attempts every independent resource
in reverse dependency order; failed cleanup retains explicit retry authority.

This is a full AOT graph dispatch owner, but not yet serving or numerical-
correctness evidence. The separate native release gate in
`tests/device_step_alloc` currently instruments the warmed descriptor
`stage_frame`/`finish` path over prebuilt received frames, proves record and
fixed-array positive controls independently, and exercises every fixed H2D
call through a bounded test seam.
Generated-C allocation review covers the launch lifecycle, while a
positive-controlled full `stage`/`execute`/`sample_completion`/`finish` runtime
gate, physical CUDA model correctness (including logits and sampled tokens),
sanitizer, leak, and benchmark evidence remain open before Phase 3 promotion.
