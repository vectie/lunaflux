# LunaFlux

> The Phase 1 physical-hardware promotion gate remains open while checked
> foundations now span offline reference correctness and bounded host-side
> online control: focused and resolved configuration, tokenizer and model
> admission, file-to-device loading, semantic and device plans, AOT execution
> contracts, reusable worker messages, generational KV metadata and block
> tables, prefix indexing, sampling, and scheduler request ownership. The
> repository does not yet claim serving readiness, physical-CUDA release
> evidence, a continuous-batching execution loop, or GPU performance.

LunaFlux is a MoonBit-native, high-throughput language-model inference engine.
Its design combines prefix-aware scheduling, deterministic paged KV memory,
continuous batching, and a small tile-oriented kernel layer without carrying
the Python runtime and compatibility debt of existing inference systems.

**LunaNexa manages the fleet. LunaFlux executes inference.**

~~~mermaid
flowchart LR
    C["OpenAI-compatible or native clients"] --> F["LunaFlux"]
    N["LunaNexa runtime adapter"] --> F
    F --> W["MoonBit scheduler and workers"]
    W --> K["AOT kernel catalog"]
    K --> G["CUDA devices"]
~~~

## Product boundary

LunaFlux owns:

- tokenizer and immutable model-plan construction;
- safetensors loading and device-aware weight placement;
- request admission, continuous batching, chunked prefill, and cancellation;
- fixed-page KV allocation and radix-indexed prefix reuse;
- sampling, streaming events, metrics, and instance health;
- one worker per accelerator and the GPU execution protocol;
- kernel capability selection and the constrained MoonTile kernel IR.

LunaFlux does not own:

- cluster placement, node enrollment, deployment rollout, or tenant governance;
- an artifact registry, container orchestrator, agent runtime, or application UI;
- arbitrary Python model code or untrusted runtime extensions;
- automatic cross-node inference before a topology-specific benchmark gate.

## Design principles

1. MoonBit owns the serving control path.
2. CUDA details stop at one private native ABI.
3. The scheduler is a deterministic single owner of request and KV state.
4. Prefix discovery and physical KV allocation are different concerns.
5. Production kernels are selected from an AOT capability manifest.
6. Instance, model, and hardware incompatibilities fail during startup;
   request-specific unsupported options fail before scheduler activation.
7. Performance is benchmark evidence, not an architectural claim.
8. Feature breadth follows correctness and steady-state performance.

## Initial release scope

The first useful release targets a single CUDA GPU and one dense,
decoder-only Llama-style architecture:

- BF16 weights loaded from safetensors;
- tokenizer.json BPE tokenization;
- greedy, temperature, top-k, and top-p sampling;
- native streaming protocol plus OpenAI-compatible endpoints;
- continuous batching, paged KV, and chunked prefill;
- prefix caching after the uncached path is proven correct.

Quantization, tensor parallelism, MoE, multimodal models, LoRA, speculative
decoding, and prefill/decode disaggregation are later gated capabilities.

## Target repository map

Directories for API, metrics, and benchmark work are created only when their
vertical phase begins; the map below records the intended ownership boundaries.

~~~text
contracts/          public protocol and lifecycle vocabulary
cmd/lunaflux/       command-line entry point
config/             validated configuration records
api/                native and compatibility endpoints
tokenizer/          tokenization implementations
model/              model specifications and immutable plans
scheduler/          admission and batch planning
kv/                 page allocator and request block tables
prefix/             radix prefix index
sampling/           token selection
engine/             reference execution and static device preparation
kernels/            exact kernel catalog, launch contracts, and reference kernels
device/             safe public device abstractions
internal/cuda/      private native bindings and stubs
metrics/            bounded telemetry
benchmarks/         reproducible performance workloads
docs/               contracts, architecture, decisions, and phase gates
~~~

Implemented packages currently cover `contracts/`, focused `config/` records,
the byte-BPE foundation and bounded selected `tokenizer.json` adapter in
`tokenizer/`, validated artifact readers, exact Llama weight bindings, bounded
host materialization for reference execution, and two-pass bounded streaming
from an approved safetensors file into final aligned device regions. That
loader retains opaque, retryable allocation-cleanup authority if a
primary load failure and its first close both fail; partial ownership is never
discarded. The device foundation also includes an immutable semantic-to-device plan, a
single activation/workspace arena plan and allocator, and exact stateless
`FullPrefill`/`FullRecompute` profiles. Kernel packages admit exact startup
bindings, content-addressed AOT families and profile-specific entry points,
launch ABIs, digest-verified module/function artifacts, and bounded production
manifest/file admission under an approved read-only mount. The private
CUDA/device seam owns explicit resources, checked transfers, module/function
loading, synchronous AOT launch, and narrow synchronous BF16 cuBLASLt GEMM
plans.

Deterministic reference kernels, greedy sampling, and the
architecture-neutral `engine/reference/` interpreter remain the correctness
oracle. The `lunaflux reference` command runs digest-pinned whole-file
correctness bundles; it is never a production fallback. An exact,
thread-confined Phase-1 device executor now binds admitted AOT artifacts,
vendor projection plans, and preplanned memory into an ordered synchronous
full-sequence run. It retains explicit cleanup authority even when construction
and cleanup both fail. A separate versioned paged graph now carries live-row
counts, token positions, page tables, and persistent split K/V state through
exact all-AOT catalog, contract, artifact, memory, and physical-blueprint
admission. Reusable device-step buffers upload that bounded descriptor without
steady-state allocation. The owner-mediated synchronous paged executor leases
weights, privately owns activation/workspace and KV allocations, preloads every
module/function and argument list, and fail-stops after any partial graph
launch. It also retains exact BF16 vocabulary-row geometry and startup-owned
readback/sampling scratch: after a successful graph it reads only producing
rows, rejects non-finite logits, deterministically selects by the request's
canonical `(seed, output index)`, and freezes the exact scheduler completion
lease before retirement. Canonical request/streaming events and
scheduler/worker messages now have bounded immutable contracts.
Worker-protocol foundations include reusable
fixed-capacity plan and completion buffers, authenticated epochs and row
drafts, provenance-bound capability recipes, whole-build checkpoints, and
explicit final-prefill sampling semantics. A separate canonical little-endian
worker-wire layer copies exact plan and completion identities, tables, sampling
replay state, and outcomes into startup-sized frames; untrusted receives check
all bounds and semantics before replacing an authenticated frame epoch. The
service authenticates received completion frames against the exact retained
plan before populating its paired completion owner, while normal scheduler
backpressure remains retryable. The worker side writes those frames directly
from authenticated received-plan rows; scheduler heap-owner capabilities do
not cross the wire boundary. Device-step staging likewise consumes validated
plan frames directly in its isolated-worker path. After exact graph execution,
that path authenticates the retained frame owner and epoch, reads each
producing BF16 logits row, applies the frame's scalar greedy/stochastic replay
fields, and writes the canonical completion frame without reconstructing
scheduler plan, sampling-parameter, or completion-owner objects.
Positive-controlled release instrumentation covers both encode/receive paths
inside the scheduler token-step window. Host-side KV metadata includes a
generational fixed-page allocator, a fixed-capacity request block-table arena,
and inline optional-free page and table identity storage. Logical full-page
prefix reuse has a fixed-capacity token trie isolated by model, tokenizer,
cache scope, and layout identity.

Startup-only runtime capacity resolution checks scheduler, cache, model-shape,
worker, page, block-table, and output-publication envelopes. The scheduler
foundation owns a fixed request registry and waiting queue, authenticates the
loaded model and row recipes, and performs bounded tokenized admission,
cancellation, deadline expiry, and terminal-notice publication. Its
transactional `build_next` path activates FIFO requests, reserves decode work
before prefill, preserves the emergency page reserve for prefill, serializes
protocol rows in the required order, and submits into distinct reusable A/B
plan owners. Exact plan, block-table, and page checkpoints restore identities
and FIFO state after a rejected build. Paired completion owners issue exclusive
leases for exact plan epochs; ordered full-batch retirement preflights output,
terminal, and KV-release obligations before publishing tokens or recycling
resources. Idle and plan-buffer pressure are allocation-free value outcomes,
and completion-slot lookup is fixed-indexed. Greedy and
bounded temperature/top-k/top-p host sampling are implemented with fixed
scratch and counter-addressed replay semantics. These remain foundations:
the private native layer can now spawn one exact worker executable without a
shell or PATH lookup and exchange bounded fixed-buffer bytes over an inherited
socketpair with monotonic deadlines and deterministic reap. Protocol-aware A/B
supervision now binds monotonic submitted plans to physically distinct frame
owners, receives responses strictly in order, and pins each completion epoch
through scheduler publication backpressure. A separately linked child-side
runtime now proves three real socket-framed plan/completion exchanges, A/B/A
reuse, EOF, zero exit, and reap. Startup is now an exact checksummed
`Configure`/`Ready` exchange binding model identity, the admitted-bootstrap
digest derived from graph/artifact evidence, the bootstrap-source digest
derived from canonical `EncodedBootstrapSource` bytes, exact process-visible
device ordinal, generation, predecessor, and all worker/inference limits; an
incompatible child cannot become protocol-ready,
and double-failure cleanup retains explicit authority. The shipped child is a
deterministic protocol executable, not the CUDA worker. The admitted full-graph
blueprint and artifact bundle now derive the admitted-bootstrap digest from a bounded canonical
schema that also binds the exact device-step limits and assignment.

A new aggregate `engine/device_worker` owner provides the child-side readiness
foundation without publishing a wire frame: inert admission retains the exact
expected startup contract and bounded model, file, device, memory, kernel, and
artifact evidence; preparation opens the assigned ordinal, verifies its target,
streams and owns the verified weights, and constructs the complete paged
executor. Readiness can be queried only while context, weights, and executor are
all live, and dependency-ordered cleanup remains retryable after compound
failures. Source-locator/config delivery into the child, `Ready` emission,
plan/execution forwarding through that owner, restart policy/backoff, live
overlap, and physical-CUDA promotion evidence remain open.
The supervisor now closes and reaps the old child before accepting exact
ordered submission abandonment and derives a non-reusing predecessor only
after every retained completion or failed submission is retired; a real
three-child gate proves continuation through sequence 5. A thread-confined
worker service now encapsulates the scheduler and process owners, retains two
scalar flight identities, retries pinned frames through scheduler
backpressure, commits bounded worker-failure completions before process
abandonment, and starts replacements only after all obligations retire. Its
independent immutable binding pins the expected bootstrap and bootstrap-source
digests, assigned device ordinal, and inference limits; construction and restart additionally
require exact equality with the scheduler-retained worker limits, model
identity, generation, and predecessor sequence. Its real-child gate proves two
outstanding plans, output backpressure, worker failure, non-reusing restart,
binding preservation, and balanced KV ownership. Still open are global
fairness/preemption and prefix
integration,
shipped and numerically validated paged-kernel artifacts, generated-logit
physical-CUDA validation, online transport integration, a positive-controlled
full graph-executor allocation gate, and the remaining physical-CUDA gates are
open. Later
packages are created only when their vertical phase begins; empty
architectural packages are deliberately avoided.

## Validation

~~~sh
moon info
moon fmt
moon check --target native --deny-warn
moon test --target native --deny-warn
scripts/validate-boundaries.sh
scripts/validate-cuda-abi.sh
moon run --target native cmd/lunaflux
~~~

## Documents

- [Product contract](docs/PRODUCT_CONTRACT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Detailed implementation plan](docs/PLAN.md)
- [Technical-debt policy](docs/DEBT_POLICY.md)
- [Benchmark contract](docs/BENCHMARKING.md)
- [Architecture decisions](docs/DECISIONS.md)
- [Implementation status](docs/STATUS.md)
