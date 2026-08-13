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
validated completion epoch until scheduler acceptance succeeds. The production
worker executable, restart/readiness, and overlap remain open. Global
fairness/preemption, generated-text decoding, and prefix integration also
remain open.
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

The current single-device loader realizes this contract with two bounded reads
of one approved regular safetensors file. It validates the complete digest and
exact selected-model tensor vocabulary before opening the destination arena,
then copies source-ordered chunks into final aligned regions while hashing the
exact bytes again. It retains no model-sized host snapshot. This is a read-only
mount contract; it does not claim atomic path-race exclusion for an adversarial
writable directory.

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

The constrained MoonTile IR describes:

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
