# LunaFlux architecture

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

Admission validates model identity, context bounds, sampling bounds, stop-set
limits, deadline viability, and security cache scope before allocating request
state. Tokenization produces an immutable token buffer.

Cancellation increments a request generation. A completion for an older
generation may retire GPU work but cannot publish output or reuse released
request metadata.

## Scheduler

The scheduler is a deterministic, single-writer state machine. Inputs are:

- waiting requests;
- completion records;
- cancellations and deadlines;
- page availability;
- immutable model and kernel capabilities;
- the configured step token budget.

Each iteration:

1. retire the completed plan and commit generated tokens;
2. release terminal request pages and prefix references;
3. preserve the configured emergency decode-page reserve;
4. continue eligible decode rows, subject to fairness;
5. admit prefix-rich waiting requests;
6. use remaining token budget for chunked prefill;
7. resolve operation shapes to kernel capability IDs;
8. write the next immutable plan and submit it.

Policy is decode-first with bounded waiting-time aging. Initial preemption is
recompute-only; host KV swapping is excluded. The same scheduler snapshot and
inputs must produce the same plan.

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

The initial catalog uses cuBLASLt or CUTLASS for GEMM and adds custom AOT
kernels only for measured serving hotspots. Production startup selects exact
artifacts by GPU architecture, dtype, layout, and shape class. Missing support
is a startup error.

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

