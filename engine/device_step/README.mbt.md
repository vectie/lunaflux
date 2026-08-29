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
physical-page bounds directly from fixed wire storage in one scan. Accepted
cells populate private reusable host scratch during that scan, but no device
upload or staged authority is published until every field passes. No scheduler
heap-owner capability is required in that path;
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
startup. A stage authorizes plan sequence and stage epoch, then validates and
fills private host scratch in one bounded scan. The startup limits bind the
loaded model's exact maximum sequence length and token ID; protocol admission
is never trusted as the only semantic envelope. Staging uploads six payload
operands and publishes the real counts header last. It performs no diagnostic
zero-count upload. Any upload failure poisons the owner, preventing kernel
binding, finish, or another stage until deterministic close.

`prepare` remains the world-one constructor and accepts only the model plan's
complete canonical KV geometry. `prepare_rank_local` is the narrow
tensor-parallel constructor: it consumes the admitted tensor-parallel KV plan,
authenticates the exact model identity and generation, obtains one rank's
canonical local layout, and re-derives its local KV-head geometry from the
semantic model plan. It retains no rank, topology, weight, collective, or
device-owner authority. Both constructors enter the same seven-buffer
allocation and cleanup transaction only after all immutable evidence is
validated.

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

The full-graph blueprint also retains one opaque authenticated graph-memory
state: absent, capture-unsupported, or a positive aligned declared upper bound.
Missing and unsupported accounting never acquire a fabricated zero-byte
scalar. Bootstrap admission re-derives the state from the exact Phase-5
authorization before hashing it into the existing canonical authorization
section.

`admit_device_worker_bootstrap` hashes the complete admitted full-graph
blueprint and artifact evidence into a bounded canonical binary schema. The
digest covers model and plan identities, the bounded process-visible device
ordinal, target, catalog-v3 ABI, exact descriptor/scalar limits, live profile,
KV geometry, arena sizes, module digests, symbols, entry points, launch
geometry, and every physical operand binding. Startup frame v4 carries that
digest, the independently admitted bootstrap-source digest, and the ordinal
across Configure/Ready. Protocol and inference envelopes
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
finish failure cannot publish a completion. Preparation explicitly selects
ordered eager, required capture, or capture with authenticated eager fallback.
Capture occurs only during startup from the exact prebuilt AOT sequence; warm
execution either launches that immutable graph exec or enqueues the same
ordered base and then waits once on the reusable completion event. No live
descriptor value constructs or updates graph nodes. Any partial eager launch,
graph launch, record, or wait failure drains the retained stream and
permanently poisons execution and descriptor state. Completion follows its
retained startup placement. Host mode reads each producing row's BF16
vocabulary logits into fixed storage and applies greedy or counter-addressed
stochastic selection using the exact `(sampling seed, output index)`.
Embedded-CUDA mode accepts greedy rows only, launches the fixed reducer already
co-compiled into the authenticated LM-head module, and copies one eight-byte
result cell per producing row into startup-owned storage. Greedy compatibility
is fused into the mandatory row preflight rather than a second scan. All reads
and selections finish before the first completion entry is written. The writer
remains open for explicit submit after executor finish or abort after any
failure; a readback or invalid-logit failure poisons the executor.

The reusable FP8-v3 frame route keeps one scalar admission record for the
executor lifetime. Each accepted frame mutates that record in place and clears
it on consume or poison, so publication does not box per-frame evidence. It
shares the same single validation/host-staging scan and retained summary plus
frame-owner/epoch authentication as BF16. A poisoned graph remains poisoned;
only a freshly prepared executor is recovery authority.

Fused residual/RMSNorm preparation also separates qualification from the token
path. Qualification accepts only the seven-argument canary ABI and owns its
four-byte device cell plus fixed host observation. `ProductionFastPath`
accepts only the distinct six-argument ABI; it allocates no canary, appends no
diagnostic kernel argument, performs no canary atomic, and performs no
per-execution host/device proof copy or CPU validation. Artifact, model,
target, fallback, and operation-adjacency authentication still completes once
during startup, before the module or executor is opened.

Close invalidates execution first and then attempts every independent resource
in reverse dependency order; failed cleanup retains explicit retry authority.

This full AOT graph dispatch owner is composed by the BF16 and symmetric-I8
worker/service paths. The narrow `tests/device_step_alloc` executable still
instruments warmed descriptor staging and fixed-H2D behavior. The separate
`tests/device_worker_alloc` executable prepares the public BF16
`DeviceWorkerOwner` through approved-root weight and schema-v2 manifest
admission, then exercises mixed/full-batch
`stage_frame`/`execute`/`sample_completion_frame`/`finish` cycles with fake
device modules, nonuniform BF16 logits, an independent scalar sampling oracle,
warmed allocation checks, hostile frames, and deterministic resource closure.

That positive control proves the production MoonBit executor and ownership
path, not emitted CUDA numerics or a spawned physical service. An approved
full-model kernel bundle, physical logits and sampled-token agreement, device-KV
cached decode, successful physical child/service execution, full sanitizer/leak
coverage, soak, and benchmarks remain required for promotion.
