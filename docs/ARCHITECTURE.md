# LunaFlux architecture

This document defines the target architecture. Implemented evidence and open
gates are recorded in [STATUS.md](STATUS.md); present-tense topology below does
not by itself claim physical-CUDA readiness or release performance.

## System context

The [counter-driven compiler repair](COMPILER_COUNTER_REPAIR_2026-09-29.md)
keeps optimization in the existing typed chain: semantic/numeric program →
legal schedule domain → resource/measured selection → ownership, storage and
effects → terminal device lowering. Instruction geometry, CTA row ownership
and complete-head epilogue ownership are independent choices. Address sharing
and publication coalescing must be represented before emission; a renderer
must not silently change a reduction law or infer a different ownership map.
Source coverage and physical performance completion are tracked separately.

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

The advanced-family serial diagnostic path uses the same canonical plan and
completion boundary. `engine/serial_generation` owns fixed-capacity greedy
continuation after an actual final-prefill token: it preserves the reserved
page/capability recipe, feeds each result into the next decode, and stops at
length, EOS or cancellation. It owns no CUDA, model-family or network state.
`integration/serial_decoder_pipeline` owns two-rank execution and commits one
completion only after activation transfer and both ranks retire. The caller
drains/releases request state before reuse; GLM adapters only bind checkpoint
layers and device effects. This diagnostic path does not replace the shared
continuous-batching scheduler or claim real-checkpoint numerical validation.

The bounded DSpark queue separates session and request lifetimes. Resident
rank, stream, AOT, activation edge, connection and verification-snapshot owners
live in the outer rank session; immutable prompts are prepared under one
aggregate host budget before CUDA. Individual ordinary or speculative requests
retire shared rank frontiers only after the remote receipt. A speculative
release then resets the ordinary coordinator's publication frontier locally,
without sending a second network/device release. The next request begins at
position zero while transport epochs continue monotonically. Final session
teardown still drains activation/model effects before closing borrowed state
and rank allocations. No payload hash or request-path authentication is part
of this lifetime transition. Physical coverage is recorded separately in the
DSpark report; sequential reuse is not continuous batching or concurrency.

The post-v1 [two-Spark workstream](TWO_SPARK.md) retains this ownership model
across two explicitly assigned nodes. Its pure topology/capacity compiler,
root-free startup envelope and local lease are separate from existing local
admission. Network execution evidence joins the existing rank worker, whose
remote mode enforces a local lease. Authenticated remote sessions carry the
existing rank protocol through fixed mailboxes into the same worker service
and online API owners. Recovery requires release of both old ranks before a
fresh group is activated. The operator-launched rank entry point and host
protocol tests exist; physical two-node execution is not yet qualified. See
the workstream document for deployment and TLS-provider limitations.

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
on failure. Eligible aged prefills precede decode, while a persistent decode
cursor provides global round-robin service among ready decode rows.
Recompute-only preemption selects a non-inflight victim in persistent cursor
order, releases its device-derived KV pages, retains dense generated-token
history, and replays the prompt plus generated history except the latest
retained decode input. Exact-epoch completion leases and ordered full-batch retirement
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
fixed Configure frame, the canonical bounded bootstrap source, and one
parent-approval attestation. The child decodes and verifies the source plus
that exact launch-bound attestation before Ready. Configure
binds the exact model identity, an admitted-bootstrap SHA-256 derived from
graph/artifact evidence, a bootstrap-source SHA-256 derived from canonical
`EncodedBootstrapSource` bytes, process-visible device ordinal, model
generation, predecessor, worker limits, and inference
limits; the supervisor publishes protocol readiness only after an identical
Ready response. The admitted-bootstrap SHA-256 is derived from the admitted full-graph blueprint
and artifact bundle's canonical module, symbol, launch, layout, operand,
device-step envelope, and exact assignment evidence. The service retains a
backend-neutral identity containing model identity and generation,
bootstrap-source digest, worker limits, and inference limits. Single-worker
bootstrap digest/ordinal and tensor-parallel rank/topology evidence stay in
private physical bindings rather than being synthesized into one public shape;
the scheduler predecessor completes the comparison at join and replacement.

Plan construction fills one scalar row draft and its retained
token/page/capability tables in a single pass. Startup-sized open-addressed
identity tables provide `O(1)` duplicate and completion-slot lookup;
the worker validates one received frame once, stages it directly, and publishes
completion against the same retained epoch.

The deployment approval verifier key is not a public runtime input. Startup
receives exact bounded bytes once through a narrow FD-7 capability, constructs
the verifier privately, and deterministically wipes and closes the source
before child activation. The deployment key never crosses into the child.
After verification, the parent derives a fresh per-child one-shot attestation
bound to the exact manifest, approved source, executable launch identity,
generation, and ordinal. Missing, stale, substituted, and replayed records fail
closed; absent external approval preserves the standalone baseline path.

The canonical paged-Llama v2 execution source is admitted synchronously by
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
physical plan because the production child reads, executes, and publishes
completion serially. During that flight the scheduler may construct and retain
exact plan N+1 in the other A/B owner; a later call submits it only after N is
retired from both scheduler and process. Concurrent physical two-slot exchange
remains legacy transport evidence and future event-loop work. A failed child is closed and
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
alias-taking constructor is explicitly fixture-only. Its independent binding
is retained unchanged across replacements. An explicit immutable online
restart policy now applies capped exponential delay only after exact child
cleanup, scheduler obligation retirement, and device invalidation. The
existing lease clock drives cooperative cached scalar wake bounds; no owner
sleeps. Attempts use private non-reusing generations, fail closed on clock,
timestamp, generation, or attempt drift, and reset only after a committed
sequence later than the current replacement's authenticated predecessor equals
the scheduler's retired sequence beyond the stability deadline. One serialized
spawned-child physical request has passed; concurrent physical child traffic
remains open. Global scheduler fairness,
recompute-only preemption, and compressed prefix reuse are implemented.
The owned service
now contains a restricted permanent
Raw-versus-Online ownership lease with exact request/publication sequencing,
trusted monotonic admission/expiry, recovery, and close authority. The public
in-process `LunaOnlineInstance` above it owns one scheduler and a rooted single
worker or generation-scoped tensor-parallel group across a startup-bounded set
of live request lanes. The cooperative fixed-lane
preparation boundary consumes a trusted receipt into a revocable
`LunaPreparedRequest` shell around one exact-generation claim. A central FIFO
quantum owner advances retained UTF-8 tokenization, token copying, and
incremental-output setup. The claim binds the model, tokenizer digest,
inference envelope, absolute deadline, and generation-leased token/output
storage; its lane returns only after lower scheduler retirement or terminal
close. The online instance retains no tokenizer, pool, or raw receipt state.
Busy, draining, and exhausted-epoch outcomes precede destructive claim transfer;
foreign preparation bindings fail before semantic-event or scheduler mutation.
Begin also preflights exact event-epoch headroom before publishing Accepted or
consuming the claim. Each transferred claim is authenticated by an opaque Luna
ticket. One global event credit serializes caller-visible publication across
lanes; its view grants read authority and its ACK alone advances the exact
semantic owner. Framed and other protocol adapters receive only that view.
Token decoding, cancellation, deadline enforcement, and terminal failure remain
owned by the instance. Final acknowledgement plus authenticated lower
retirement recycles only that lane; other lanes, the lease epoch, publication
history, plan predecessor, and worker remain live. Explicit instance drain owns
healthy worker close. Authenticated worker, protocol, or device failure
cooperatively restarts the worker before publishing the affected request's
recovered terminal. The byte-BPE layer,
canonical request-frame scanner, and request-preparation pool have reusable
operation-budgeted Luna work. The pool captures trusted receipt time before
byte one and imports a validated frame view directly into its preallocated
token, semantic, and output owners without object-form request materialization.
The object materializer remains a synchronous compatibility path; adapter
dispatch and network transport remain above this aggregate. Native one-shot,
reusable max-one, fixed-lane pipeline, serialized OpenAI HTTP, and bounded
OpenAI connection-pool owners compose that boundary. The pool provides exact
fixed-capacity concurrent-client arbitration; none of these owners claims an
unbounded or fleet reactor.
The production runtime chooses between those owners from the authenticated
preparation lane count: one lane keeps the singleton compatibility owner, while
two or more lanes select the startup-preallocated native-framed or OpenAI
connection pool. Instance admission already requires that lane count to equal
the scheduler's total request slots, so ingress cannot create more request
authorities than the service was built to own. For OpenAI Responses, the
inference listener also serves the exact bounded `/healthz`, `/readyz`, and
`/metrics` routes, so inference and observation have one origin and there is no
second control listener. Native-framed mode retains a separate loopback-only
HTTP control listener. Authenticated `private_network_plaintext` mode is
limited to OpenAI on `0.0.0.0` or `::` inside an isolated deployment network;
LunaNexa maps that wildcard publication to an inspected private container
address. This grants neither public exposure nor TLS authority.
Child ownership, executable and fixed handshake storage are preallocated; native
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

The implemented radix uses fixed startup arenas for path-compressed token
edges, immutable full-page anchors, copied security/cache scope bytes, active
references, and deterministic priority/LRU eviction metadata. It never stores
a GPU tensor or device pointer. Scheduler lookup is bounded by the last fully
reusable page before the request-private tail; acquired references and block
tables are transactional, and cache publication occurs only after final
prefill. Aged fairness remains absolute while unaged requests may use reusable
depth as a deterministic priority.

Only full pages are shared in v1. A partial tail remains request-private.
Eviction removes only zero-reference cached runs. Prefix identity is salted by:

- model artifact and model-plan digest;
- tokenizer digest;
- adapter identity when adapters are introduced;
- RoPE/scaling configuration;
- KV layout and page-size version;
- cache security scope;
- hashes of non-text inputs when such inputs are later supported.

The experimental [streaming workstream](STREAMING.md) adds a worker-local
pinned-host payload tier beneath prefix matching. Its immutable plans and pure
residency transitions are interpreted through fixed storage and explicit CUDA
owners. Restores copy complete K/V pages into the existing stable device arena;
graph staging is fenced while any copy is outstanding. The scheduler must retain
the matching page-generation reservation until completion or proven drain.
Explicit versioned runtime descriptors for BF16, I8, FP8, Qwen3, Mistral and
tensor parallel admit the capability through startup v5. The neutral
[streaming IR](STREAMING_ARCHITECTURE.md) drives cooperative
worker commands and atomic multi-page restore publication in the existing radix.
Cancellation retains reservations until completion or post-reap invalidation;
replacement binds a new child or whole-group epoch. The immutable all-rank
barrier retains each completed shard until the group transaction resolves.
Physical model parity and performance
qualification remain release gates.

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

The [`parallel` IR proposal](PARALLEL_IR.md) extends physical execution planning
with a pure, immutable memory-effect graph. Existing TP execution plans can
project into `compiler/parallel_ir`: byte-range aliases determine dependencies,
collective sequences determine rank agreement, and startup lowering emits
ordered or compute/communication queues with explicit completion events.
The default executor remains ordered. An opt-in CUDA/NCCL adapter binds these
queues to separate streams and preallocated events; physical qualification is
still required. No parallel graph analysis enters the
request scheduler or token-step path.

Weights are streamed or mapped into their final device destination. A
multi-device loader must never materialize the complete model independently on
every worker.

The current single-device loader resolves a strict relative descendant beneath
an independently approved pinned root with component-wise no-follow traversal.
Its inspection reads bounded headers and validates the selected-model tensor
vocabulary and placement before opening the destination arena. Loading reads
only selected tensor ranges into final aligned regions, with no complete-file
hashing, copied-byte hashing, or per-tensor checksums. Sharded loading reads
only the rank's actual source segments rather than scanning other ranks' bytes.
The family-neutral sharded reader uses this same no-payload-hash rule for GLM,
DeepSeek and MiniMax. Supplied inventory hashes are declared labels, not
runtime-verified payload identities; optional acquisition verification is an
external concern. `label:HEX` inventory entries explicitly carry a supplied
checkpoint label without representing a shard checksum.
Inspection-based preparation retains geometry and locators; loading rechecks
bounded structure and same-file metadata, not cryptographic integrity.
Payload I/O reuses caller-owned fixed host
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
5. Artifact admission retains each declared AOT module once and maps
   every required stable entry point to a bounded function symbol.

The shared artifact/file loader and later tensor-parallel structural join do
not hash manifest or CUBIN bytes. Digests are deployment-supplied labels, not
runtime integrity claims. Bounded reads, module sizes, symbol resolution,
immutable ownership and launch compatibility still apply. Optional acquisition
verification remains external to startup and dispatch.

Configuration, runtime-descriptor, instance-policy and execution-manifest
snapshot readers follow the same rule: no byte hashing or checksum comparison.
Launch-file and two-Spark startup readers also consume supplied labels without
payload hashing. Executable activation opens the staged regular executable and
retains its descriptor without reading the complete binary or copying it into
a sealed memfd. Materialized executable inspection reads metadata only.
Deployment must keep the opened inode immutable; this path does not protect
against in-place binary mutation or claim verified payload integrity.
Qwen/Mistral descriptor and worker bootstrap configuration reads do not repeat
that work. Existing digest fields remain supplied labels for compatibility;
they do not authenticate snapshots. Bounded parsing, model/shape/rank joins,
capacity planning and deterministic close remain execution correctness checks.
The family-neutral GLM-hybrid, advanced-decoder and joint-diffusion artifact joins, fused
runtime sidecars, row variants and paged-composite loader also do not hash
opaque module payloads or repeat sidecar integrity scans. Offline exporters may
still create cache/plan identities; loading consumes their supplied labels.
The Qwen token-ID bridge does not read its own executable, launch or capacity
receipt for integrity checks. Only tokenizer parsing requires a file read;
the supplied tokenizer digest remains a label, not a verified checksum.
Token-step worker plan/completion traffic has no full-frame checksum scan at
encoding, greedy continuation, publication or reception. Legacy checksum words
are written as zero and ignored on reception. Structural validation and
generation/sequence/KV ownership remain mandatory; parent and worker binaries
are upgraded together rather than mixing checksum-enforcing old receivers.
The same no-checksum-scan rule applies to rank-group control/payload frames
and graph telemetry sidecars. Rank identity format parsing occurs on Configure;
steady frames compare the locked binding without repeating hexadecimal-format
validation. Bounds, sequencing and ownership checks remain, and checksum-field
compatibility requires coordinated parent/child upgrades.

Native-framed service traffic follows the same rule: request reception,
text-request encoding and per-token event encoding have no checksum state or
whole-frame FNV pass. They retain bounded parsing, UTF-8, sampling and explicit
semantic event lifetimes. The five-byte token event regression reduces charged
CPU steps from 428 to 215; this is not an end-to-end throughput result.

Startup Configure/Ready, rank-envelope and selected-topology frames also omit
redundant FNV scans. Rank Configure no longer computes/verifies a frame-wide
SHA-256, reconstructs an entire canonical frame or compares it byte by byte.
Its legacy digest slot carries the already-admitted worker contract identity,
checked against the locked binding, not a checksum of transport bytes. Decode
snapshots once after its structural and binding checks. Parent-approval records
do not self-hash their transport tail; one-time launch/manifest authorization
remains separate. Source identity binding at startup, compiler/cache identities
and algorithmic token/prefix hashing are not payload-hardening scans. All
startup peers must upgrade together; no mixed-version compatibility is claimed.

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

Performance specialization is shape- and capability-driven. Model builders
produce semantic operations and legal shape classes; they do not choose CUDA,
WMMA, a warp count, or a vendor library. LunaTile strategy selection separates
decode GEMV, small-row tiled GEMM, large-row tiled GEMM, prefill attention, and
decode attention before backend lowering. A backend may then use generated
tiles or an AOT vendor implementation. Likewise, an execution-graph bucket is
the portable contract; CUDA Graph is one backend realization. See
[PERFORMANCE_ROADMAP.md](PERFORMANCE_ROADMAP.md) for the current measured gaps
and optimization order.

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

The current software foundation implements typed LunaTile views,
instructions, constraints, validation, canonical serialization, deterministic
memory planning, and canonical CUDA AOT planning input. Generic programs remain
semantically neutral. Closed BF16 lowering families now emit deterministic
source and recipes for embedding, RMSNorm, positioned RoPE, residual add,
projection, gated MLP, language-model head, and paged attention. An offline
builder compiles an already produced candidate set twice and publishes exact
CUBIN bytes and receipts. Strict post-compile binders join those bytes to final
catalog-v3 launch contracts, and the inert `luna_kernel_bundle` package proves
full model-plan coverage while producing canonical schema-v2 manifest bytes and
a content-addressed module inventory. Greedy sampling may use a separately
authenticated AOT device reducer that publishes one fixed result cell per
producing row. Temperature, top-k, and top-p retain an explicit startup-selected
host mode; the authenticated bootstrap records that placement and it is never
chosen inside the token step.

Runtime manifest admission independently rederives the complete model, memory,
KV, launch, and artifact contract before loading exact module bytes. The paged
executor can use ordered eager execution or an explicitly selected startup-only
capture policy. The BF16 production worker derives capture with validated eager
fallback only from authenticated capture-safe Luna graph metadata; absent or
capture-unsupported metadata remains ordered eager. The current I8 production-
worker constructor retains the eager default. Symmetric-I8 weight-only and
finite-E4M3 FP8 software target admission is closed over exact `sm89`, `sm90`,
and `sm120`; adjacent architectures remain rejected. I8 materialization,
catalog-v4 contracts, manifest/bootstrap joins, and the same serialized
executor/service path also exist as typed software control paths. FP8 likewise
has authenticated materialization, AOT launch, reusable paged-executor,
descriptor, worker, and service joins. These are software capabilities, not
physical quantized correctness or performance claims. External deployment
approval remains binding evidence. LunaFlux does not independently validate
the deployment's detached-signature scheme; it privately authenticates the
startup-supplied approval binding and never accepts caller-constructed approval
claims.

Optional production-fast-path V2 spans are executable in both the legacy Llama
and generic numeric-BF16/Mistral worker APIs. Residual plus RMSNorm is replaced
by one prepared launch; positioned QKV/RoPE/KV-write plus read-only paged
attention is replaced by two. Startup authenticates the exact catalog,
modules, plan adjacency, fallback identities, raw-pointer ABIs, and live device.
Token execution performs no artifact authentication, filesystem validation,
canary transfer, or qualification scan. Residual V2 preserves the authenticated
CUDA graph policy; the diagnostic qualification ABI remains eager-only.
Descriptor-pinned optional artifacts cross the exact runtime
descriptor/bootstrap into deployed children for both Llama and numeric BF16;
the QKV composite is admitted only after the child authenticates its live
device identity. Missing artifacts preserve the standalone graph.
Current-source physical CUDA/performance qualification remains an open gate.

A lower production-V2 physical harness and a literal spawned
`device-greedy`/`device-greedy-fused-v2` harness now reach these production
boundaries and compare the fixed eight-byte device result with an independent
host full-logits referee. Only their software/static gates have run in this
worktree; they carry no current physical or performance result.

Paged profile-priority capture binds the exact launch set, target/profile,
mixed row/cache/position/page geometry, diagnostic page-table trace digest, and
bounded sorted counters. It has no execution or page ownership. A separate
LunaTile promotion owner remains evidence-gated and inert: even with sealed
paired wins and external approval, no generic fixture operation becomes a
catalog entry until an authenticated real-operation specializer maps exact
operands, shape, and numerical contract.

The product-owned BF16 candidate exporter, offline builder, final-contract
binder, deployment materializer, and spawned runtime now form one bounded
approved-model path. Its pinned tiny-BF16 r14 campaign passed physical paged
prefill and same-page decode with exact token agreement and deterministic
cleanup. A later bounded loopback campaign also crossed the actual native
listener with exact events and network/KV cleanup. This is deliberately
narrower than release promotion: broad/concurrent listener serving, true
cached-prefix execution, broad shapes and contexts, physical
sanitizer/leak evidence, and benchmark comparison remain open. The I8 path has
typed software admission but no positive physical execution evidence.
Tensor-parallel physical promotion and a suitable trained production tokenizer
also remain open.

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
After bearer authentication, the HTTP parser writes directly into a
startup-preallocated typed handoff. One receipt remains bound from byte-one
capture through JSON parsing, chat-template expansion, semantic validation,
and request admission. The route constructs no JSON tree, `String`/`Bytes`
prompt copy, canonical request frame, `GenerateRequest`, or `TextInput`.

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

The compatibility command `lunaflux legacy-config-plan` prints this structure before model
materialization. Components receive only the configuration subset they use.

The implemented subset includes a strict byte-bounded configuration document
reader, focused immutable records, deterministic startup decisions,
`ResolvedRuntimeCapacity`, and inspection-safe operator preflights. Runtime
capacity checks scheduler, cache/prefix, model-shape, worker-protocol, page,
block-table, and output-publication envelopes and materializes bounded
host-owner limits. The explicitly named `legacy-*` model-root commands admit
the model root and report an explicit not-ready result rather than fabricating
resolution. Their separate-root forms accept separately
approved model/kernel roots and an independently digest-pinned strict runtime
descriptor. That startup-only boundary composes model metadata, weight
inspection, KV layout, paged execution-manifest admission, bootstrap source,
startup contract, and an inert device-worker plan while opening no device and
retaining no filesystem capability. Canonical `doctor`, `plan`, and
`inspect-kernels` instead consume the same digest-suffixed deployment admission
as live run/bench, close every root, and publish only root-free evidence without
device, process, or listener authority. A focused ops report projects the
legacy separate-root admission into checked weight, activation/workspace, KV, artifact, worker, and
inference scalars without filesystem or device authority. Scheduler/cache and
service capacity remain explicitly unavailable because descriptor v1 does not
admit their configuration. Broad physical device/kernel promotion, benchmarks,
and listener-level live traffic remain open.

The production run boundary accepts exactly one absolute deployment-root label
with an independent `#sha256=` digest for its fixed launch file. Launch-file
evidence is root-free. The runtime owner opens separate model, kernel, and
policy authorities, closes policy authority after the pure instance join,
checks the exact assigned CUDA target before process/listener activation, and
passes model/kernel authority into the existing online service exactly once.
The worker executable is opened componentwise without following symlinks and
snapshot-hashed through that same descriptor. Linux live admission copies the
authenticated bytes into an exactly sealed private memfd; process activation
duplicates the opaque capability and uses `fexecve` without a second pathname
open or hash. The retained path and digest remain deployment evidence only.
Live activation fails closed on platforms without this pinned execution route;
root-free materialization evidence is a distinct type that no spawn API accepts.

Opaque CLI inference authentication is admitted through a deployment-created,
preconnected Unix stream at inherited descriptor 6, separate from the
descriptor-5 drain capability. A fixed bounded frame is read exactly once, the
channel is closed, and source scratch is wiped before model, device, worker, or
listener construction. Digest-bound instance-policy v3 supplies the complete
OpenAI Responses construction contract, including the maximum accepted
credential length. Its transport is either exact loopback plaintext or exact
private-network plaintext on a wildcard container listener; native v1/v2
reject credentials and OpenAI v2 remains fail-closed. Neither mode establishes
TLS or public routing.
The retained opaque API-auth policy has one idempotent full-buffer close shared
by every verifier alias. OpenAI server/pool terminal drain and startup rejection
propagate that close; HTTP reuse wipes prior used head/body cells and terminal
close wipes its complete fixed request storage. Constant-time credential
comparison touches the configured credential length, not the larger maximum
input capacity, while retaining bounded preallocated storage.

Offline release materialization does not weaken that live identity rule.
`ApprovedRootMaterializationView` first authenticates one pinned no-follow
source root at its actual staging label, then carries a separately validated
canonical target label without pretending that target exists on the release
host. Only mapped descriptor, policy, tokenizer, and worker-file loaders may
borrow this view. They reconstruct the same launch-selected semantic recipe
with final target labels while reading exact staged bytes. The no-overwrite
materializer keeps the output beneath an authenticated cleanup claim until the
root-free join evidence and exact bundle both verify; failure deletes only that
claim. Live startup continues to require real target-namespace identity.

## Advanced workload isolation

### Shared foundation, distinct execution engines

Text output also preserves family-neutral numerical planning. A learned
residual-stream head is an immutable precision operation, separate from the
ordinary final RMSNorm/vocabulary projection/sample operation. CUDA lowering
composes these contracts; a prepared text frame owns only bounded scratch and
function resources and lends its launches to the decoder queue. DeepSeek's
adapter binds the learned function/base/scale checkpoint planes, while GLM
retains its unweighted mean. Invalid learned controls cannot reuse old output.
Neither the precision IR nor warmed queue dispatch branches on model names.

LunaFlux remains one framework and repository. Autoregressive and diffusion
execution are separate engines over shared model storage, LunaTile compilation,
device allocation, AOT kernels, execution graphs, and lifecycle primitives.
They may be packaged into separate binaries; neither imports the other's
scheduler. Diffusion iterations are not synthetic token-decode steps.

The autoregressive engine owns token admission, prefill/decode, KV and prefix
reuse. The diffusion engine owns conditioning, joint latent state, denoising
iterations and VAE decode. Model builders describe these stages as immutable
plans. Pure transformations select layouts, lifetime reuse and schedules;
prepared backends own explicit buffer writes, submission, completion and close.
CUDA-specific choices remain in device lowering, not family builders.

For H3, video and audio use distinct shifted-flow timestep grids but one joint
denoiser evaluation per iteration. Both predictions must observe the old joint
state before either latent is updated. Allocate latent/prediction/workspace
storage before iteration, retain invariant conditioning, and reuse storage;
functional planning must not imply full-latent copies each iteration. A failed
partial iteration is terminal, never replayed against partly updated state.
Cancellation stops new submissions but cannot release in-flight resources.

The first milestone is one complete bounded H3 request, not a universal media
runtime. Detailed implementation and measurements are tracked in
[MINIMAX_H3_EXECUTION.md](MINIMAX_H3_EXECUTION.md). Shared foundation does not
imply that current H3 candidates already provide executable inference.

DeepSeek V4, GLM 5.3, and MiniMax H3 extend startup planning without adding
family branches to scheduler, KV, device core, or the private CUDA ABI:

~~~text
exact family config/weights
  → family-owned immutable plan projection
  → advanced-decoder | GLM-hybrid | joint-diffusion requirement vocabulary
  → family-neutral CUDA source, AOT candidate, and artifact admission
  → focused integration joins for authenticated family storage
  → runtime construction only after independent execution authority
~~~

Semantic capability admission, source candidates, structural candidate
coverage, uploaded buffers, and inert artifact joins are distinct evidence
levels. None may be named or treated as executable readiness. Workload-specific
candidate-evidence adapters remain separate so advanced-decoder, GLM-hybrid,
and joint-diffusion packages do not acquire transitive dependencies on one
another. Numerical CUDA text is further isolated in family-neutral, focused
owners: scaled forward/inverse RoPE, interleaved RoPE with bit-exact prefix copy, prefix
RMSNorm with optional suffix copy, staged joint attention, mHC block control,
mHC pre-reduce, mHC post-combine, mHC head-reduce, deterministic compressed
indexing, deterministic window/compressed-index joining, causal short
convolution, KDA input/output-gate/output projection, bias-free MoE router
projection, safe decay control, sigmoid-gated RMSNorm, normalized recurrent
delta, VAE latent destandardization, channel-major pointwise projection,
weight-normalized Conv1d and ConvTranspose1d, block-FP8/UE8M0 projection,
bit-exact packed-KV assembly, alias-free SnakeBeta activation,
four-stream hyper-connection mixing,
non-overlapping compressed-KV preparation, MoE sigmoid/correction, grouped MoE
routing, selected-expert SwiGLU, BF16 binary composition, indexed parameter
banks, 97-stage audio AMP workspace planning, and packed or separate-weight
SwiGLU MLP are independent
renderers; thin workload lowerers authenticate only their
requirement identity, exact geometry, launch ABI, target/compiler policy, and
inert evidence. DeepSeek mHC is decomposed into versioned block-control,
pre-reduce, post-combine, and head-reduce semantics. Block control has an exact
eight-operand BF16/F32 source, pre-reduce has a separate four-operand BF16/F32
source, post-combine has a six-operand BF16/F32 source, and head-reduce has its
own six-operand BF16/F32 source. Replicated Query-A and the separate replicated
pre-normalization KeyValue projection now reuse the same family-neutral
block-FP8 renderer as Output-B with profile-owned storage/layout binding.
Independent inert joins authenticate each storage boundary without granting
interpretation, materialization, upload, or launch authority; artifact
admission accepts KeyValue and then fails closed at Query-B. Ratio-128
deterministic compressed indexing likewise has
a dedicated bounded I32 producer. Downstream compressed attention is split
into index join, shared-KV preparation, selected sparse attention, and inverse
RoPE. The join has a separate eight-operand I32 source that preserves official
prefill padding, decode ring order, and literal window-then-compressed
concatenation without sorting or deduplication. Selected sparse attention has
an independent eight-operand correctness source with shared BF16 K/V, I32
indices, stable ordered-F32 softmax, and an F32 denominator-only sink. A
separate seven-operand borrowed-buffer preparation source lays out the full
ordinary BF16 K/V prefix plus completed ratio-128 rows during prefill and the
physical 128-slot circular window plus completed compressed prefix during
decode. It does not learn compression or own cache state. A separate in-place
three-operand inverse scaled-YaRN source retains the first 448 BF16 head
components bit-exact and conjugate-rotates the final 64 through F32 with one
BF16-RNE write. The following official projection is split into a grouped BF16
Output-A candidate and a single-rank block-FP8 Output-B candidate. Output-B
uses dynamic per-row/block-128 UE8M0 activation scales, E4M3 weights, UE8M0
weight scales, ordered F32 block accumulation, and one BF16-RNE output. Separate
inert compositions bind the official Query-A, KeyValue, Output-A conversion,
and Output-B source storage/layouts to their candidate operands without granting
interpretation, conversion, materialization, upload, tensor-parallel collective,
or launch authority. None
of these candidates grants live KV authority. GLM Flash's
KDA short convolution is a
family-neutral per-stream renderer with explicit CSR sequence offsets and
history handoff. A separate serial recurrent-delta renderer owns normalized
Q/K, per-component decay, delta/state update, and explicit F32 initial/final
state without absorbing projection, cache ownership, or throughput authority.
Separate family-neutral renderers own its BF16 Q/K/V/forget/beta projection,
F32 safe-lower-bound decay control, two-stage output-gate projection,
sigmoid-gated per-head RMSNorm, and final bias-free BF16 output projection.
Both GLM profiles also use a family-neutral bias-free router projection with
F32 activation input, row-major BF16 expert weights, ordered F32 accumulation,
and raw F32 logits. A distinct five-operand renderer produces both the
uncorrected sigmoid scores and correction-biased choice scores, failing closed
if the output buffers alias. A separate serial correctness renderer scores each
group from its top two corrected scores, selects groups and eight experts from
corrected scores, then gathers, normalizes, and scales only the uncorrected
scores. Separate routed-expert, shared-expert, and routed/shared-add candidates
now consume that result. A generic indexed layout authenticates the exact
per-expert checkpoint names, ordinals, shapes, and bank offsets; a thin GLM
join binds all six bank digests and operand sizes without owning packing,
upload, or execution. Full then uses an exact non-aliasing BF16 outer residual
add. Flash instead uses the official four-stream mHC composition
`post * branch + combineᵀ * residual`, with control inputs rounded to BF16,
ordered F32 matrix accumulation, and one BF16 output round. Upstream Flash
hyper-control is represented by two separate family-neutral serial correctness
sources: learned function/base/scale projection plus collapse, followed by an
explicit positive-matrix Sinkhorn stage with one initial column normalization
and 19 alternating row/column passes. Their ABIs keep post, initial matrix,
collapsed state, and final combination storage distinct. They grant no live
state, cache, artifact, or launch authority; lower-ID tie ordering is explicit
and not claimed equivalent to PyTorch's unsorted `topk`.
The DSA path then uses a family-neutral bit-exact K/V assembly source. Full
splits packed KV-B `[T,64,448]`, broadcasts the rotated shared 64-wide suffix,
and produces contiguous K/V `[T,64,256]`; Flash uses a distinct four-operand
zero-suffix ABI to split packed `[T,64,512]`. This owns no cache handoff,
allocation, or execution authority.
MiniMax's joint plan binds attention rotary width and Q/K RMS epsilon instead
of hiding them in a family adapter. Its complete converted 703-tensor and
original 560-tensor F32 VideoVae layouts and
1,087-tensor F32 AudioVae vocabularies now admit bounded host materialization,
while its
separate conditioning reference binds request-derived geometry and ordered
missing execution prerequisites without acquiring artifact or execution
authority. Exact F32 video/audio latent destandardization prefixes retain the
official per-channel parameters. AudioVAE then uses a separate channel-major
F32 pointwise input projection followed by an exact two-stage, per-output-
channel weight-normalized kernel-7 Conv1d. Its normalized-weight workspace and
normalize-before-convolve order are explicit. The immediately following first
BigVGAN upsampler is a separate per-input-channel weight-normalized kernel-9,
stride-5 ConvTranspose1d from `[2,1024,207]` to `[2,512,1035]`, with its own
explicit normalized workspace. Its first AMP activation is represented
truthfully as anti-aliased 2x upsampling, F32 SnakeBeta, and stride-2
downsampling through two explicit 8,478,720-byte workspaces. The following
first three-way AMP residual stage has an exact 97-launch plan: 126 checkpoint
regions occupy one canonical packed-parameter layout, a disjoint 58,167,296-
byte workspace retains the original upsampler output and precomputed block-0
activation as separate inputs, all dilated weight-normalized convolution pairs
and residual updates execute in `30+33+33` order, and one final launch performs
the ordered three-way average. The following `decoder.ups.1.0` transition is a
separate per-input-channel weight-normalized kernel-9, stride-5
ConvTranspose1d candidate from `[2,512,1035]` to `[2,256,5175]`, with an
explicit normalized-weight workspace. Its following three-way AMP stage is
also exact: `resblocks.3–5` bind 126 parameter regions, kernels 3/7/11,
dilations 1/3/5, a 119,465,984-byte disjoint workspace, and 97 launches over
`[2,256,5175]`, including the precomputed first activation and ordered final
average. The following `decoder.ups.2.0` is now an exact per-input-channel
weight-normalized kernel-4, stride-2 ConvTranspose1d transition from
`[2,256,5175]` to `[2,128,10350]`. Its first alias-free activation,
`resblocks.6.activations.0`, owns distinct length-12 up/down filters and two
`[2,128,20700]` workspaces. Complete `resblocks.6–8`, later upscalers, VAE
encode, and the remaining decoder bodies remain fail-closed. The current exact implementation status and
open promotion gates are
recorded in [MODEL_FAMILY_SUPPORT.md](MODEL_FAMILY_SUPPORT.md).

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

### Quantized query/KV execution prefix

`compiler/precision_ir.QuantizedQueryKvPrecision` retains immutable matrix,
activation-quantization, rounding and aggregate workspace decisions. The CUDA
source package alone emits FP8 conversion, reduction and device instructions.
`integration/quantized_query_kv_frame` prepares explicit buffers/functions and
borrows six launches into a caller-owned decoder queue. The DeepSeek adapter
alone resolves official tensor names and model dimensions. Learned F32 RMSNorm
and unweighted BF16-expression head normalization are distinct semantic stages;
neither may silently replace the other. This prefix does not own rotary, KV
cache mutation, compression, attention, tensor-parallel placement or serving.

### Intra-kernel physical planning

The projection compiler now lowers selected schedules into immutable fragment,
operand-storage and copy-ownership plans in `compiler/physical_tile_ir` before
CUDA instruction emission. The attention schedule owns a closed physical
program covering direct, grouped, split-key, shared-score and query-fragment
terminals. Scalar and query-fragment online-softmax state dialects live in
`compiler/attention_physical_ir`; storage and effect choices are retained
through backend lowering rather than reconstructed by source emitters.
Single-launch and partitioned attention share this dispatch. The inter-launch parallel/execution IR remains
separate: it cannot describe a kernel's register liveness or shared-memory
publication by itself. See [the migration status](PHYSICAL_TILE_IR_2026-09-29.md)
for exact coverage and remaining families; this is not a claim that every
backend path has been unified.

Projection emission now requires a retained `PhysicalProjectionTileCompilation`.
Row-domain normalization, rotary/KV writes and sampling retain distinct programs
in `compiler/elementwise_physical_ir`. `compiler/fusion_regions` validates
observable outputs and ordered effects before choosing executable partitions;
`compiler/resource_policy` separates resource eligibility, measured choices and
explicitly unmeasured fallback. These are AOT planning layers, not token-path
interpreters. See [completion scope and validation](COMPILER_ARCHITECTURE_COMPLETION_2026-09-29.md).

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
