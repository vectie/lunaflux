# LunaFlux architecture

This document defines the target architecture. Implemented evidence and open
gates are recorded in [STATUS.md](STATUS.md); present-tense topology below does
not by itself claim that the online service or device-KV path exists today.

## System context

LunaFlux is an instance-level execution engine. Deployment systems are callers,
not libraries.

~~~mermaid
flowchart LR
    D["Direct operator"] --> E["LunaFlux instance"]
    N["LunaNexa adapter"] --> E
    E --> A["Assigned accelerator set"]
    E --> M["Read-only model mount"]
    E --> T["Metrics receiver"]
~~~

## Process topology

The single-GPU path has one MoonBit service process and one isolated device
worker. The service process owns API tasks, tokenizer tasks, the scheduler, KV
metadata, and worker supervision. The worker owns the CUDA context, streams,
graphs, device allocations, and kernel launches.

Multi-GPU adds one worker per device. It does not add a process per API or
scheduler subsystem.

~~~mermaid
flowchart TB
    API["Native API and OpenAI adapter"]
    TOK["Tokenizer pool"]
    MAIL["Bounded request mailbox"]
    SCH["Single-owner scheduler"]
    RAD["Radix prefix index"]
    KV["KV page allocator"]
    PA["Schedule plan A"]
    PB["Schedule plan B"]
    W["Device worker"]
    EX["Static graph executor"]
    KC["Kernel catalog"]
    ABI["Private CUDA ABI"]
    GPU["CUDA device"]

    API --> TOK --> MAIL --> SCH
    SCH <--> RAD
    SCH <--> KV
    SCH --> PA
    SCH --> PB
    PA --> W
    PB --> W
    W --> EX --> KC --> ABI --> GPU
~~~

The scheduler builds plan N+1 while the worker executes plan N. Plan descriptor
memory is double-buffered and preallocated. Completion travels through a
bounded single-producer/single-consumer channel with monotonically increasing
plan sequence numbers.

## Request lifecycle

~~~text
Admitted
  → Tokenizing
  → Waiting
  → Prefill ─┐
             ├→ Decode → Finishing → Finished
             └→ Finishing → Finished

Any non-terminal scheduling state may become Cancelled or Failed.
~~~

Outer protocol validation establishes the admitted request and the tokenizer
owns the `Tokenizing` transition. It produces an immutable token buffer before
the scheduler receives a `TokenizedRequest`; scheduler-owned slot state begins
at `Waiting`. Before taking that slot, the scheduler validates selected model
identity, context and physical-page envelopes, deadline viability, and the
already-bounded request options it supports. It never tokenizes text.

Cancellation increments a request generation. A completion for an older
generation may retire GPU work but cannot publish output or reuse released
request metadata.

## Scheduler

The scheduler is a deterministic, single-writer state machine. Inputs are:

- waiting requests;
- completion records;
- cancellations and deadlines;
- page availability;
- resolved runtime capacity and authenticated prefill/decode row capability
  recipes;
- the configured step token budget.

Each iteration:

1. retire the completed plan and commit generated tokens;
2. release terminal request pages and prefix references;
3. preserve the configured emergency decode-page reserve;
4. continue eligible decode rows, subject to fairness;
5. admit prefix-rich waiting requests;
6. use remaining token budget for chunked prefill;
7. apply the provenance-bound capability recipe for each selected row;
8. write the next immutable plan and submit it.

Policy is decode-first with bounded waiting-time aging. Initial preemption is
recompute-only; host KV swapping is excluded. The same scheduler snapshot and
inputs must produce the same plan.

The current scheduler implements the bounded host-control portion of this
iteration: it owns a tokenized-request registry, FIFO waiting queue,
cancellation/deadline transitions, separate terminal and generated-token
rings, host page/block-table owners, and distinct paired reusable A/B plan and
completion owners. `build_next` transactionally
activates eligible requests, reserves decode resources before prefill, emits the
protocol's prefill-first row order, and restores exact plan/table/page identity
on failure. Exact-epoch completion leases and ordered full-batch retirement
preflight publication and KV-release obligations before state mutation, then
publish final-prefill/decode tokens and recycle exact owners. Corresponding
plan/completion data can now be detached from heap capabilities through
canonical bounded wire frames with transactional untrusted receive and exact
service-side completion-plan authentication. The worker-side device path now
stages an authenticated received-plan frame and, after execution, writes a
completion frame from exact frame-bound logits and scalar sampling fields
without scheduler heap-owner values. The private native layer now owns an
exact-path shell-free child and inherited socketpair with bounded I/O,
timeouts, and reap. The supervisor owns two physical plan/completion frame
pairs, permits two monotonic submissions, receives oldest first, and retains a
validated completion epoch until scheduler acceptance succeeds. A separately
linked inherited-channel child proves the complete framing lifecycle with
deterministic protocol completions. Before plan traffic, the parent sends a
fixed Configure frame followed by the canonical bounded bootstrap source. The
child decodes and verifies its independently bound digest before Ready. Configure
binds the exact model identity, an admitted-bootstrap SHA-256 derived from
graph/artifact evidence, a bootstrap-source SHA-256 derived from canonical
`EncodedBootstrapSource` bytes, process-visible device ordinal, model
generation, predecessor, worker limits, and inference
limits; the supervisor publishes protocol readiness only after an identical
Ready response. The admitted-bootstrap SHA-256 is derived from the admitted full-graph blueprint
and artifact bundle's canonical module, symbol, launch, layout, operand,
device-step envelope, and exact assignment evidence. A distinct service-owned
immutable binding supplies the expected bootstrap and bootstrap-source digests,
ordinal, and inference limits rather than trusting the child's echo; scheduler-retained
identity, generation, predecessor, and exact worker limits complete the
comparison at join and replacement.

The canonical paged-Llama v1 execution source is admitted synchronously by
`engine/execution_manifest_file` through a caller-owned approved root. Its
opaque inert aggregate retains the exact plans, FullGraph contracts, admitted
artifacts, and blueprint needed by device-worker admission without owning a
filesystem or device resource.

Filesystem authority is separate from bootstrap-source path labels. The host
independently opens deployment-approved model and kernel roots into opaque
capabilities; strict relative descendants are resolved component-by-component
without following symlinks and remain pinned across namespace replacement.
The production spawn boundary duplicates those neutral internal capabilities
into fixed child descriptor roles and retains the same pinned pair across
replacement. Passing roots through argv, ambient environment, or trusting
decoded absolute labels is forbidden.

The aggregate device-worker readiness owner admits independently expected
model metadata and startup, a precomputed immutable weight-file inspection,
and one opaque aggregate paged execution admission before opening resources.
Admission binds the inspection's exact layout to the execution aggregate and
derives the canonical bootstrap. Preparation compares the complete received
startup contract, opens the assigned ordinal and verifies its semantic target,
completely re-admits the retained inspected file without repeating inspection,
and prepares the complete paged executor. It exposes only the readiness
contract, and only while context, weights, and executor are all live; cleanup
retries in executor-to-weights-to-context order. Device/model loading completes
before a production child sends Ready. Before initial root-bound pair
activation or spawn, each encoded absolute model/kernel label must match the
exact corresponding caller-owned approved directory capability through
no-follow opaque identity admission. Replacement continues from the retained
pair and does not re-resolve ambient labels. The root-bound service admits one
outstanding plan from the reusable A/B storage because the production child
reads, executes, and publishes completion serially; two-slot overlap remains a
legacy echo/transport fixture and future event-loop work. A failed child is closed and
reaped before the service commits each exact submitted plan as a scheduler worker failure and
retires its supervisor obligation in sequence order. The service then
terminally invalidates every remaining device-backed request and
releases its host KV identities; waiting requests own no device state and may
survive. Only then can the supervisor derive the next non-reusing predecessor
for a replacement. The implemented thread-confined worker service owns this
join: it retains a single production flight identity, records a plan before
transport, pins validated
responses through scheduler backpressure, and exposes request/event methods so
callers do not retain a separate mutable scheduler API alias. Production
construction now starts from an immutable scheduler blueprint and ordinary
approved roots, constructing both mutable owners internally; the old
alias-taking constructor is explicitly fixture-only. Its independent
binding is retained unchanged across replacements. Restart
policy/backoff and live overlap remain open. Global fairness/preemption,
prefix integration, and network ingress also remain open. The owned service
now contains a restricted permanent
Raw-versus-Online ownership lease with exact request/publication sequencing,
trusted monotonic admission/expiry, recovery, and close authority. The public
in-process `LunaOnlineInstance` above it owns one scheduler and rooted worker
across sequential healthy request epochs. The off-reactor preparation boundary
consumes a trusted receipt into a revocable `LunaPreparedRequest` shell around
one exact preallocated claim. That claim binds the model, tokenizer digest,
inference envelope, absolute deadline, token buffer, and incremental-output
owner. The online instance retains no tokenizer or raw receipt state. Busy,
draining, and exhausted-epoch outcomes precede destructive claim transfer;
foreign preparation bindings fail before semantic-event or scheduler mutation.
Begin also preflights exact event-epoch headroom before publishing Accepted or
consuming the claim. Each transferred claim is authenticated by an opaque Luna
ticket. `take_event` issues the only request/event-epoch credit; its view grants
read authority and its ACK alone advances the semantic owner. Framed and other
protocol adapters receive only that view. Token decoding, cancellation,
deadline enforcement, and terminal failure remain owned by the instance.
Final acknowledgement retires only request-local state; the lease epoch,
publication history, plan predecessor, and worker remain live. Explicit
instance drain is the only healthy worker-close path, while authenticated
worker, protocol, or device failure remains close-only. The byte-BPE layer has
reusable operation-budgeted Luna work; incremental text conversion,
tokenizer-pool orchestration, adapter dispatch, and network transport remain
above this aggregate. Child
ownership, executable and fixed handshake storage are preallocated; native
spawn, scalar handshake
validation and cleanup, and owner publication allocate no managed objects
while rooted authority is live.
Capability IDs are copied from authenticated model-generation recipes;
scheduler policy does not inspect model operations or select kernels.

The scheduler package may depend on request, KV, prefix, and capability types.
It may not depend on API, tokenizer implementation, model-family adapters,
CUDA, or kernel implementation packages.

## KV memory

The worker preallocates a device KV arena at startup. The scheduler owns a
matching fixed-size page table:

- page ID: index plus generation;
- intrusive free-page queue;
- per-request dense block table;
- reference count for active requests and cached prefixes;
- deterministic eviction metadata;
- no per-step page object allocation.

An allocation is valid only when index and generation match. Reusing a released
index increments its generation so stale plans are detected.

Page layout is selected by the immutable model plan and kernel capability
manifest. The v1 engine has one page size and one full-attention layout.

## Prefix reuse

The radix index answers which token prefix is reusable. The page allocator
answers where reusable KV resides. These are separate packages.

Each radix node stores token edges, immutable page runs, security/cache scope,
reference count, and deterministic eviction metadata. It never stores a GPU
tensor or device pointer.

Only full pages are shared in v1. A partial tail remains request-private.
Eviction removes only zero-reference cached runs. Prefix identity is salted by:

- model artifact and model-plan digest;
- tokenizer digest;
- adapter identity when adapters are introduced;
- RoPE/scaling configuration;
- KV layout and page-size version;
- cache security scope;
- hashes of non-text inputs when such inputs are later supported.

## Model planning and loading

The loader parses bounded JSON and safetensors metadata without executing model
code. It validates dtype, shape, layer count, head relationships, vocabulary,
RoPE parameters, tensor names, byte ranges, and total materialization size.

A model-family builder converts a validated ModelSpec into an immutable
ModelPlan:

- tensor placements and sharding;
- ordered operator graph;
- KV layout;
- workspace bounds;
- legal batch and sequence shape classes;
- required kernel capabilities.

The scheduler sees only ModelPlan capabilities. Model-family branching ends at
plan construction.

Weights are streamed or mapped into their final device destination. A
multi-device loader must never materialize the complete model independently on
every worker.

The current single-device loader resolves a strict relative descendant beneath
an independently approved pinned root with component-wise no-follow traversal.
Its one-shot API performs two bounded reads: it validates the complete digest
and exact selected-model tensor vocabulary before opening the destination
arena, then copies source-ordered chunks into final aligned regions while
hashing the exact bytes again. Inspection-based worker preparation adds an
earlier complete device-free admission and privately retains that validated
locator; later loading reopens only that descendant and still completely
re-admits it before allocation. Payload I/O reuses caller-owned fixed host
storage apart from the bounded immutable safetensors header required by the
parser. No model-sized host snapshot or ambient path authority is retained.
Terminal source-file close must succeed before readiness is published. If it
fails, the otherwise-ready allocation is closed; a simultaneous allocation-
close failure returns retryable cleanup authority at the distinct
`SourceClose` stage.

## Static device execution preparation

Device preparation keeps semantic planning, storage planning, and executable
artifacts as separately checked evidence:

1. A static device plan binds one model identity, exact weight layout, resolved
   kernel catalog, target, operation graph, and workspace bound.
2. An activation plan computes BF16 value lifetimes, reuses aligned slots only
   after the final consumer, retains the terminal output, and appends one shared
   workspace to a single startup arena.
3. An exact execution profile derives immutable token-staging, weight,
   activation, workspace, operation, and terminal-output views for one admitted
   batch and sequence shape.
4. A launch-contract set fixes every AOT-backed operation's profile-specific
   entry point, dimensions, ordered semantic operands, byte counts, alignments,
   and workspace claim.
5. Artifact admission verifies each content-addressed module once and maps
   every required stable entry point to a bounded function symbol.

No one item is execution evidence by itself. Before launch, a prepared executor
must prove identity, device target, catalog version, exact operation order, and
every launch operand against the actual token, weight, activation, and
workspace allocation regions. Dispatch may then consume only precomputed
records; it must not allocate, inspect model-family names, or reinterpret
manifest byte claims.

Two execution contracts coexist. The stateless `FullPrefill` and
`FullRecompute` profiles execute every supplied token and remain the Phase-1
baseline. The separately versioned paged graph carries exact live-step counts,
token positions, row/sequence descriptors, page tables, and persistent split
K/V state through all-AOT admission. Its synchronous owner privately owns every
mutable device resource and consumes only prevalidated, prebuilt launch
records. This establishes a true incremental execution contract, but STATUS
remains authoritative for the still-open physical-kernel, numerical,
allocation-instrumentation, sampling/readback, and serving gates.

## Kernel architecture

The engine begins with a kernel catalog, not a universal compiler.

The constrained LunaTile IR describes:

- typed tensor views with shape, strides, dtype, address space, and alignment;
- tile loops and parallel mappings;
- load, store, asynchronous copy, barrier, and pipeline stages;
- matrix multiply-accumulate;
- reductions, broadcasts, and elementwise expressions.

Compilation stages are:

~~~text
validate
→ canonicalize layout
→ choose tile and warp mapping
→ plan register/shared memory
→ vectorize
→ introduce async copies and pipeline
→ lower to CUDA
→ compile AOT
→ benchmark and publish capability manifest
~~~

### Typed offline specialization

LunaTile is not only a safer syntax tree for generated CUDA. Its type and
capability boundary must make invalid model, layout, storage, and kernel
combinations unrepresentable where practical, and reject every remaining
incompatibility before artifact construction. Tensor shape, strides, dtype,
address space, alignment, quantization layout, model identity, target profile,
and kernel capability are semantic inputs, not untyped generator options.

An offline specializer consumes an authenticated `ModelPlan`, static device
plan, exact execution profile, and kernel-catalog target. It may partially
evaluate stable facts such as dimensions, tensor byte offsets, layout strides,
RoPE parameters, quantization codebooks, expert strides, and profile-specific
entry points. It then emits deterministic AOT input plus a content-addressed
specialization record containing:

- every typed input identity and semantic version;
- the baked constants and layout/codebook digests;
- the compiler, flags, target, and stable entry points;
- the declared floating-point contract: bit-exact operation order or a named
  tolerance and token-agreement policy;
- references to differential, real-tensor, dispatch-canary, numerical, and
  benchmark evidence.

Generated source is neither a public API nor trusted execution authority. It
cannot select a model, reinterpret a tensor, acquire a device, or run because a
request arrived. Artifact admission accepts it only when its specialization
record, module digest, launch contract, and typed runtime capability all agree.
This is controlled offline partial evaluation, not a universal compiler or a
production JIT.

The evidence ladder follows the same boundary: an independent scalar referee,
adversarial synthetic blocks, admitted real tensor rows, proof that the
specialized dispatch actually ran, logits and deterministic token agreement,
then an end-to-end mixed-workload benchmark. Strict floating-point builds may
claim bit equality only when operation ordering is preserved; reassociating or
vendor paths require an explicit tolerance contract. A microbenchmark win is
insufficient when the operation is not an end-to-end hotspot.

This subsection is target architecture. The current repository has typed model,
device, launch, and artifact-admission foundations, but no LunaTile IR,
specialization compiler, or specialization-record implementation yet; those
belong to Phase 5.

The initial catalog uses a narrow cuBLASLt path only where one semantic
operation is representable by its single-GEMM ABI, and uses custom AOT families
for the remaining admitted operations. A content-addressed module may contain
multiple semantic families and profile-specific stable entry points. Launch
contracts bind those entry points to exact shapes and ordered operands;
artifact admission binds them to digest-verified module bytes and bounded
function symbols. Production startup selects exact artifacts by GPU
architecture, dtype, layout, and shape. Missing support is a startup error.

Developer JIT, when introduced, writes only to a content-addressed developer
cache. It is disabled in production and never runs because of a live request.

## Native ABI

internal/cuda is the only package allowed to bind CUDA directly. It presents
private extern functions and safe MoonBit wrappers.

- Device, context, stream, event, graph, allocation, and module handles are
  opaque external values.
- Every owned handle has explicit close or destroy.
- Borrowed host buffers remain borrowed for call duration only.
- No raw pointer, CUDA enum, or vendor error string crosses into public APIs.
- Size conversion is checked before the FFI call.
- Destruction is idempotent or returns a typed lifecycle error.
- C stubs include moonbit.h and are exercised under AddressSanitizer.

NCCL follows the same pattern in a later phase.

## API architecture

The native typed protocol is canonical. OpenAI compatibility translates at the
outer edge and cannot change scheduler or model vocabulary.

API tasks are asynchronous and bounded. Slow clients receive bounded buffering
and eventually backpressure or cancellation. They cannot hold scheduler locks;
the scheduler owns no asynchronous socket operation.

## Configuration and capability resolution

Configuration is parsed into focused records. Startup constructs an immutable
ResolvedPlan containing:

- effective model and tokenizer identity;
- device inventory and memory budget;
- KV capacity;
- maximum supported context and concurrency envelope;
- selected kernels and reasons;
- unsupported requested capabilities;
- safety reserves and graph-capture shape classes.

The CLI command lunaflux plan prints this structure before model
materialization. Components receive only the configuration subset they use.

The implemented subset is `ResolvedRuntimeCapacity`: it checks scheduler,
cache, model-shape, worker-protocol, page, block-table, and output-publication
envelopes and materializes bounded host-owner limits. The full device/kernel
`ResolvedPlan` and its CLI explanation remain target behavior.

## Failure model

- API failure cannot corrupt scheduler state.
- Tokenizer failure rejects one request.
- Invalid model metadata prevents startup.
- Kernel incompatibility prevents readiness.
- Worker failure fails in-flight requests, invalidates its device generation,
  and makes the instance unready.
- A stale completion or plan sequence is rejected.
- GPU OOM after successful startup is treated as an allocator or accounting
  defect, not ordinary backpressure.
- Drain rejects new admissions and allows bounded existing work to finish.

## Observability

Metrics are bounded by model-plan identity and instance identity, never raw
request IDs as metric labels. Required signals include:

- TTFT and inter-token latency distributions;
- queue, tokenization, prefill, decode, and stream-blocked time;
- scheduled prefill/decode tokens per step;
- batch rows and token-budget utilization;
- KV pages free, active, cached, hit, evicted, and preempted;
- worker submission, execution, synchronization, and kernel timing;
- cancellation, deadline, validation, worker, and device failures;
- model load, warm-up, graph-capture, and readiness time.

Tracing may carry opaque request correlation, but payload capture is opt-in and
outside default telemetry.

## Package dependency law

~~~text
contracts  ← api, tokenizer, engine
model      ← loader and architecture builders
kv/prefix  ← scheduler
kernels    ← planner and worker
device     ← worker
internal/cuda ← device only
~~~

Forbidden edges are enforced by a repository script once those packages exist:

- scheduler → api, model-family, device, internal/cuda;
- contracts → any implementation package;
- model → api or scheduler policy;
- public packages → concrete internal types;
- any package → LunaNexa or another MoonSuite repository.
