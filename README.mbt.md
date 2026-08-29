# LunaFlux

> The supported single-GPU BF16 path has exact sealed-final30-source physical
> evidence on `sm120`: authenticated model execution, paged KV, required CUDA
> graph capture, spawned-worker execution, bounded broad native serving,
> concurrent progress, malformed/foreign isolation and recovery, sanitizer/ABI
> gates, and exact resource balance.
> Checked foundations also cover focused configuration, tokenizer/model
> admission, native and OpenAI ingress, continuous scheduling, compressed
> prefix reuse, sampling and stops, AOT capability admission, telemetry, and
> operator preflight. Authenticated loopback Responses, the pinned physical
> cached-prefix check, and one real-timestamp LunaFlux measurement now pass.
> First-release promotion remains open for routable TLS/control approval, the
> full pinned vLLM/SGLang comparison, final OCI/SBOM/provenance evidence, and
> LunaNexa routability. I8 and FP8 software admission now includes exact
> `sm120`, but positive physical quantized execution and positive tensor
> parallelism on homogeneous peer-capable NCCL hardware remain post-v1 gates.
> The corrected historical Phase 2/3 v2 soak completed two 24-hour balance
> passes. Current v3 smoke plus both 2,300-wave fast and timer diagnostics pass.
> The exact final30 source passes 2,297/2,297 Linux tests and a fresh complete
> single-GPU BF16 campaign. The exact-final29 v3 24-hour rerun also passes after
> host loss interrupted the preceding final7 attempt without terminal evidence.
> The newer exact current-source archive passes the complete 2,448/2,448 native
> matrix and aggregate dependency/debt boundary locally, is sealed and uploaded
> read-only, and carries no new physical claim until its post-soak campaigns run.

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
- kernel capability selection, the constrained LunaTile kernel IR, and offline
  specialization and artifact admission.

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

For the focused edit/check workflow, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

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
loader retains opaque, retryable allocation-cleanup authority if allocation
close fails after either a primary load failure or terminal source-file close;
partial ownership is never discarded. The device foundation also includes an
immutable semantic-to-device plan, a pure complete numeric tensor-layout
planner, a single activation/workspace arena plan and allocator, and exact
legacy stateless `FullPrefill`/`FullRecompute` profiles. Catalog v4 static
planning retains the exact numeric schema, operation execution digest, and
catalog-owned AOT entry point, while the legacy profile bridge fails closed
for v4. Kernel packages admit exact startup
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
launch. It also retains exact BF16 vocabulary-row geometry. An authenticated
AOT greedy reducer can return one fixed eight-byte result per producing row;
the explicit host-sampling mode retains startup-owned readback/scratch for
temperature, top-k, and top-p. Both paths reject non-finite logits, preserve
deterministic tie/replay semantics, and freeze the exact scheduler completion
lease before retirement. Canonical request/streaming events and
scheduler/worker messages now have bounded immutable contracts. A pure
fixed-buffer framed service codec covers every native request and event variant
without claiming listener readiness. Incremental output state copies tokenizer
pieces into caller storage, validates split UTF-8, and withholds cross-token
stop strings without constructing per-token collections or text values.
A transport-neutral admission owner captures monotonic receipt before parsing,
binds the exact model and tokenizer, preserves that absolute deadline, and
separates scheduler stop tokens from private incremental string-stop state.
A reusable operation-budgeted `LunaTokenizerWorker` owns the authoritative
byte-BPE path. Above it, a fixed-lane `LunaRequestPreparationPool` retains
canonical UTF-8, advances text tokenization, token copying, and incremental
output setup through a central FIFO quantum, and transfers an exact-generation
claim whose token and output storage remains leased until scheduler retirement.
Saturation and draining do not consume the receipt; deadline, cancellation,
work-ceiling, stale-generation, and explicit lane-return paths fail closed.
The legacy object-form preparation facade remains synchronous. The live Luna
pool can instead capture receipt time before byte one, scan canonical request
frames, bind model identity, and import input/stop/cache data directly into its
fixed lane through one operation budget, without materializing a
`GenerateRequest`. Object-form materialization remains a proportional
compatibility path. This preparation owner deliberately has no socket or
listener authority; the bounded native-framed and OpenAI servers own network
ingress above it. The OpenAI route likewise parses authenticated HTTP JSON
directly into fixed typed handoff storage: the same receipt survives byte-one
capture, authentication, parsing, template expansion, semantic validation, and
admission without a JSON tree, prompt `String`/`Bytes` copy, canonical-frame
render/checksum/reparse, or intermediate `GenerateRequest`/`TextInput`.
Worker-protocol foundations include reusable
fixed-capacity plan and completion buffers, authenticated epochs and row
drafts, provenance-bound capability recipes, whole-build checkpoints, and
explicit final-prefill sampling semantics. A separate canonical little-endian
worker-wire layer copies exact plan and completion identities, tables, sampling
replay state, and outcomes into startup-sized frames; untrusted receives check
all bounds and semantics before replacing an authenticated frame epoch. The
service authenticates received completion frames against the exact retained
plan and retirement authority before staging each frame exactly once into its
paired typed completion owner. Normal scheduler publication pressure retains a
private accepted-flight state and returns scalar backpressure; retry does not
reread wire bytes, reopen the completion writer, advance its epoch, or allocate
an error. The worker side writes those frames directly from authenticated
received-plan rows; scheduler heap-owner capabilities do not cross the wire
boundary. Device-step staging likewise consumes validated
plan frames directly in its isolated-worker path. After exact graph execution,
that path authenticates the retained frame owner and epoch, reads each
producing BF16 logits row, applies the frame's scalar greedy/stochastic replay
fields, and writes the canonical completion frame without reconstructing
scheduler plan, sampling-parameter, or completion-owner objects.
Scheduler plan construction fills one scalar row draft and its retained
token/page/capability tables in a single pass, then commits directly into the
paired reusable owner. Startup-sized open-addressed identity tables provide
`O(1)` duplicate and completion-slot lookup. The worker validates each
wire plan once into reusable staging and retains that same frame epoch through
direct device staging and completion publication.
Positive-controlled release instrumentation covers both encode/receive paths
inside the scheduler token-step window. Host-side KV metadata includes a
generational fixed-page allocator, a fixed-capacity request block-table arena,
and inline optional-free page and table identity storage. Logical full-page
prefix reuse now uses a compressed, startup-preallocated token radix isolated
by model, tokenizer, cache scope, and layout identity. The scheduler and KV
owners share only authenticated page identities: lookup, active references,
request-private tails, post-prefill publication, zero-reference eviction,
rollback, cache policy, and bounded hit/miss/eviction telemetry are integrated
without importing device implementation packages. Exact salted roots and
parent/token children use balanced indexes, zero-reference eviction uses an
exact min-heap, duplicate page validation uses fixed-scratch heapsort, and
changed-path compaction never rescans an unaffected subtree.

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
resources. Token and terminal storage remains physically separate, but one
monotonic publication sequence makes their global dequeue order fail closed.
Idle and plan-buffer pressure are allocation-free value outcomes,
and completion-slot lookup is fixed-indexed. Device greedy sampling and bounded
temperature/top-k/top-p host sampling are implemented with fixed scratch and
counter-addressed replay semantics. Restrictive top-k is `O(V log K)`; the
unrestricted/top-p path retains canonical `O(V log V)` ordering so floating-
point accumulation and draw mapping remain byte-for-byte replayable. Native
and OpenAI request paths share the canonical sampling, seed, stop-token, and
incremental stop-string vocabulary.
These remain foundations:
the private native layer can now spawn one exact worker executable without a
shell or PATH lookup and exchange bounded fixed-buffer bytes over an inherited
socketpair with monotonic deadlines and deterministic reap. The legacy A/B
supervisor binds monotonic submitted plans to physically distinct frame
owners, receives responses strictly in order, and pins each completion epoch
through scheduler publication backpressure; its echo fixture proves three
socket-framed exchanges and A/B/A reuse. The root-bound production facade
instead grants one outstanding credit to a serialized device child. Startup
sends an exact checksummed `Configure`, the canonical bounded bootstrap source,
and one parent-approval attestation before accepting `Ready`. The exchange
binds model identity, the admitted-bootstrap
digest derived from graph/artifact evidence, the bootstrap-source digest
derived from canonical `EncodedBootstrapSource` bytes, exact process-visible
device ordinal, generation, predecessor, and all worker/inference limits; an
incompatible child cannot become protocol-ready,
and double-failure cleanup retains explicit authority. The device child imports
pinned root roles, reconstructs the model and executor, publishes Ready only
after exact resource readiness, and then executes bounded frames. The admitted full-graph
blueprint and artifact bundle now derive the admitted-bootstrap digest from a bounded canonical
schema that also binds the exact device-step limits and assignment.
The deployment verifier key arrives once through a startup-owned bounded FD-7
capability, is consumed privately, and is wiped and closed before child
activation. The child never receives it: the parent instead creates a fresh
one-shot attestation bound to the admitted manifest, approved source, worker
launch identity, generation, and ordinal; replay, substitution, absence, and
stale generation fail closed. Baseline inference remains standalone when
external approval is absent.

The public in-process `LunaOnlineInstance` prepares that owned scheduler and
rooted worker once, then admits a startup-bounded set of streaming requests
under fresh opaque Luna tickets. Per-request lower authority is held in fixed
lanes, while one global event credit preserves deterministic publication
order. Final event acknowledgement and authenticated lower retirement recycle
only that lane; the lease epoch, scheduler publication history, plan
predecessor, other requests, and physical worker continue. Explicit instance
drain is the healthy shutdown path. Worker, protocol, or device failure uses
cooperative recovery and publishes the failed request's canonical terminal
before the replacement is reused. Transport-neutral framed service and native
TCP/OpenAI owners compose this aggregate without exposing scheduler aliases.
OpenAI Responses serves `/healthz`, `/readyz`, and `/metrics` on that same HTTP
origin; native-framed mode keeps its separate loopback HTTP control listener.
Authenticated `private_network_plaintext` OpenAI policy admits only wildcard
container listeners for an isolated deployment network. It grants neither host
exposure nor TLS authority, and LunaNexa maps the wildcard publication to the
inspected private container address before probing or invocation.

Startup filesystem authority now has a narrow native foundation: independently
approved absolute directories become opaque pinned capabilities, strict
relative files are opened by component with `openat`/no-follow/type checks,
and positional reads are protected from concurrent close by atomic lifecycle
leases. Paths, descriptors, native errors, and metadata timestamps do not
escape the public API. An ASan probe and 1,024-cycle descriptor-balance soak
exercise the exact C translation unit. Startup callers can request one bounded
immutable whole-file snapshot under a single lifecycle lease; the native read
admits the initial size before its sole payload allocation and rejects
truncation, trailing growth, or same-handle size/mtime/ctime change before
publication. Weight, kernel-artifact, model-configuration, and offline
reference-bundle loaders now consume these capabilities directly. The offline
CLI is the only production boundary that opens its absolute deployment root,
and it closes that root before execution or output. Weight inspection privately
retains its strict relative locator and fixed host storage carries bounded
positional reads into device transfer.
The bounded `engine/execution_manifest_file` aggregate admits one
digest-pinned canonical paged-Llama v2 manifest on the same root, deriving
catalog v3, static/memory plans, full-graph contracts, artifact admission, and
the inert device-step blueprint from already-admitted typed inputs rather than
deserializing semantic plans.
Terminal file-close failure cannot publish a ready allocation; allocation-close
failure in that path returns retryable cleanup authority at `SourceClose`.
The sealed final30 `sm120` campaign proves this loader as part of the
authenticated BF16 model, spawned-worker, broad listener, sanitizer, and
resource-balance path. Physical cached-prefix reuse and a bounded benchmark
measurement pass; broader contexts, full performance comparison, and
unsupported numeric/topology targets remain separate release gates.

A new aggregate `engine/device_worker` owner provides the child-side readiness
foundation without publishing a wire frame: inert admission retains the exact
expected startup contract and bounded model, file, device, memory, kernel, and
artifact evidence; preparation opens the assigned ordinal, verifies its target,
streams and owns the verified weights, and constructs the complete paged
executor. Readiness can be queried only while context, weights, and executor are
all live, and dependency-ordered cleanup remains retryable after compound
failures. Bootstrap-source reconstruction, device-owned `Ready`, and serialized
plan/execution forwarding now pass through this owner. The sealed final30
campaign proves one authenticated physical child and a bounded broad native-
framed campaign with concurrent requests, saturation backpressure,
cancellation, typed rejection, malformed-frame isolation, same-owner recovery,
restart, and drain. Broader contexts and full comparative performance evidence
remain open.
The supervisor now closes and reaps the old child before accepting exact
ordered submission abandonment and derives a non-reusing predecessor only
after every retained completion or failed submission is retired; a real
three-child gate proves continuation through sequence 5. A thread-confined
worker service now encapsulates the scheduler and root-bound process owners,
owns startup-bounded online request lanes around one serialized physical
production flight, retries pinned frames through scheduler
backpressure, commits bounded worker-failure completions before process
abandonment, and starts replacements only after all obligations retire. Its
independent immutable binding pins the expected bootstrap and bootstrap-source
digests, assigned device ordinal, and inference limits; construction and
restart additionally require exact equality with the scheduler-retained worker limits, model
identity, generation, and predecessor sequence. Its echo gate proves two-slot
transport; its service gate proves one-flight backpressure, worker failure,
device-state invalidation, non-reusing bounded restart backoff,
binding preservation, and balanced KV ownership. Global fairness,
recompute-only preemption, compressed prefix integration, fixed-lane online
transport, and mixed/full-batch host allocation evidence are implemented.
Phase 5 also provides semantically neutral LunaTile validation, one exact
opaque BF16 residual-add specialization contract, canonical CUDA AOT planning
input, deterministic offline specialization records, externally approved
capability manifests, AOT artifact admission, and a host decoded-plan
canary. The generic LunaTile serial oracle now anchors a separately identified
parallel SIMT source candidate with exact block/warp/lane mapping, uniform
pipeline/barrier structure, lifetime-planned shared storage, and conservative
global read/write-region validation. A disjoint exact-`sm120` BF16
`m16n16k16` WMMA source candidate binds the authenticated graph, serial oracle,
SIMT fallback, alignment, target, and resource identities. Test-only probe and
campaign packages compose deterministic double compilation, independent CPU
and serial-CUDA numerics, dual-tool SASS observation, resource inspection,
memcheck/racecheck/initcheck, and a non-circular sealed evidence record. The
typed release binder can admit only that exact sealed record and remains
`manifest_bindable=false` with no promotion or runtime-loading authority.
Neither new candidate has executed on NVIDIA hardware yet; their physical
campaigns and every performance or production-promotion gate remain open. The
decoded-plan canary is not dispatch evidence. Production-fast-path V2 code is
separately implemented for residual plus
RMSNorm and for positioned QKV/RoPE/KV-write followed by read-only paged
attention. Startup admission authenticates exact artifacts, plan adjacency,
fallback identities, and the live device once; prepared execution replaces the
standalone spans without canary traffic, cryptography, filesystem work, or
diagnostic host/device transfers in a token step. Residual V2 preserves the
authenticated CUDA graph policy; only the qualification ABI remains
eager-only. Both legacy Llama and generic
numeric-BF16/Mistral worker APIs can execute these optional spans and preserve
the standalone route when they are absent. Descriptor-pinned optional artifacts
now cross the exact runtime descriptor/bootstrap into deployed children for
both Llama and numeric BF16; the QKV composite binds only after the child has
authenticated its live device identity. Neither fused V2 route has
current-worktree physical CUDA or performance qualification. A lower
production-V2 campaign harness and a literal spawned `device-greedy`/
`device-greedy-fused-v2` harness now exercise the deployed source boundaries
and exact eight-byte device result against an independent host full-logits
referee. Their source/static gates pass; neither harness has run on NVIDIA
hardware, so they add no physical or performance evidence.

The paged profile-priority contract now captures exact mixed row, cache,
position, page-slice, and launch identities plus bounded sorted counters; its
page-table trace digest is diagnostic evidence, never page ownership. A
separate LunaTile optimizer-promotion owner can consume sealed paired evidence
and external approval, but stays inert and non-bindable. No current generic
fixture operation maps to a real catalog operation; an authenticated
real-operation specializer must bind exact operands, shape, and numerical
contract before any runtime selection can exist.

Focused
configuration inspection and the `run`, `doctor`, `plan`, `bench`, and
`inspect-kernels` commands form the Phase 6 operator foundation. Their
canonical forms all accept the same digest-suffixed deployment root and share
the production runtime-instance admission. `doctor`, `plan`, and
`inspect-kernels` close all roots and publish only root-free semantic evidence;
they open no device, spawn no child, and bind no listener. Older model-root and
separate-root preflights remain only under visibly named `legacy-*`
compatibility commands. That strict loader composes model, weights, KV,
execution-manifest, bootstrap, startup-contract, and inert device-worker
admissions. The separate-root `legacy-doctor` adds a typed report of
weight, activation/workspace, KV, artifact, execution, and inference ceilings;
its checked device total never double-counts workspace or artifact file bytes,
and unavailable scheduler/service capacity is not fabricated. Host evidence
remains available across a physical probe mismatch while readiness stays
false. Phase 6 now includes a compiler/JIT-free OCI source contract, mandatory
exact context/base verification, a Linux-only build wrapper, hostile static
gates, and the deployment runbook in `docs/DEPLOYMENT.md`. No Linux/CUDA image,
approved base/builder provenance, final-rootfs/SBOM scan, or reproducible final
image digest was produced on this macOS host. A later pinned Linux/CUDA
campaign crossed the approved deployment, spawned-child, generated BF16 AOT,
paged-KV, and two-token execution path. A later bounded native-listener
campaign passed the same exact request over loopback with balanced network,
KV, listener, and child ownership. The sealed final30 campaign adds
broad bounded native serving with malformed/foreign isolation and recovery,
physical eight-token prefix reuse with independently fixed outputs, one real-
timestamp LunaFlux measurement, and exact post-campaign GPU process/memory
balance. Still open are broader contexts, prefix-rich and prefix-cold comparison
profiles, full performance promotion, final release evidence, and
the optional positive I8 and tensor-parallel/NCCL gates. Later packages are
created only when their vertical phase begins; empty architectural packages
are deliberately avoided.

The production one-argument form is literal:
`lunaflux run ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>`. The suffix is
an independent digest of the fixed `lunaflux.launch.json` descendant. Its
strict v1 envelope binds independent model, kernel, and instance-policy roots,
descriptor/policy files, the worker executable, and an explicit tagged Luna
deployment approval. Startup closes launch and policy authority before
publication, joins the existing service composition, and can publish Ready
only after the exact assigned CUDA target and listener are live. This macOS
host must therefore fail the physical gate and cannot demonstrate Ready. The
2026-08-27 CUDA campaign passed the primitive, generated BF16-family,
paged-attention-fixture, and capture-required graph probes. The later r14
campaign also passed one approved tiny-BF16 deployment through a spawned child
and the owned service path for the exact two-token continuation `1031,2185`,
with one-page KV residency and complete cleanup. That bounded pre-listener
fixture is not traffic readiness, general serving, or performance evidence. A
later r17 campaign passed one bounded native loopback traffic slice with the
same tokens, exact five-event order, restored KV balance, and full cleanup. It
does not establish TLS, public exposure, concurrency, broad serving, or
performance.

The finite 10,000-request balance gate has passed. The frozen v1 low-rate
24-hour soak aborted because its harness incorrectly required every balanced
wave to enter the two-row histogram bucket. The corrected v2 policy keeps
batching as cumulative functional evidence and distinguishes scheduled,
attempted, accepted, and actual cancellation waves. The digest-pinned v2
campaign started under launchd on 2026-08-24 and completed two independent
24-hour passes before its exact keepalive label was removed. Each pass ran for
86,400,002 milliseconds and closed with balanced request, connection, queue,
active-request, KV, pending-work, and child-process resources. The separate
`com.vectie.lunaflux.phase23.soak.20260824.rerun1` attempt is not promotion
evidence: despite its rerun name, its frozen service binary identifies the
already rejected v1 policy and reproduced that harness false negative.
The current source is now explicitly v3 because malformed connection-terminal
input contributes one typed network rejection. Its digest-bound smoke and
2,300-wave fast real-worker diagnostic pass with complete resource balance.
The historical v2 records are not relabeled. The exact-final29 v3 24-hour run
passed with 86,400,012 milliseconds of wrapper time and 86,400,002 milliseconds
measured by the worker, 86,400 requests over 43,200 cycles, positive required
wave counters, 833 balanced accepts/disconnects, `kv_free8`, and closed
queue/active/KV/pending/child resources. Its exact evidence manifest SHA-256 is
`a49609d819e143adf1e63d99f0ee97014a58d095baaffd0f9a3c2eb43ff8788f`;
the downloaded evidence archive SHA-256 is
`3a105bb7ef9665c86e8dabec1808ef61da838c3579bcb4929e05948e623d4fb0`.
Host loss interrupted the preceding final7 attempt without terminal evidence.

## Validation

The commands below are phase/release gates. During ordinary edits, use
`scripts/check-local.sh` with the affected packages as described in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); unrelated hardware, sanitizer,
soak, benchmark, and release campaigns are not part of that loop.

~~~sh
moon info
moon fmt
moon check --target native --deny-warn
moon test --target native --deny-warn
scripts/validate-boundaries.sh
scripts/validate-cuda-abi.sh
scripts/validate-oci-packaging.sh
moon run --target native cmd/lunaflux -- doctor ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>
moon run --target native cmd/lunaflux -- plan ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>
~~~

Canonical `plan` authenticates the complete deployment semantic join used by
`run` and `bench`, then closes all root authority and reports readiness false.
The older caller-declared configuration explainer remains available only as
`legacy-config-plan` and is not deployment admission.

## Documents

- [Product contract](docs/PRODUCT_CONTRACT.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Detailed implementation plan](docs/PLAN.md)
- [Technical-debt policy](docs/DEBT_POLICY.md)
- [Benchmark contract](docs/BENCHMARKING.md)
- [Architecture decisions](docs/DECISIONS.md)
- [Implementation status](docs/STATUS.md)
