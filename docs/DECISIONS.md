# LunaFlux architecture decisions

## ADR-0001 — Independent sibling repository

**Status:** accepted

LunaFlux is an inference execution engine. LunaNexa is a model and cluster
control plane. They have different trust boundaries, release evidence,
dependencies, and failure domains.

LunaFlux therefore lives in its own repository and exposes a provider-neutral
runtime contract. LunaNexa may own an adapter but LunaFlux never imports it.

## ADR-0002 — MoonBit owns the control path

**Status:** accepted

Configuration, tokenization, model planning, scheduling, KV metadata, prefix
reuse, sampling, worker coordination, API contracts, and telemetry are
first-party MoonBit native code.

GPU vendor libraries remain external dependencies behind a private C ABI.
This is MoonBit-native ownership, not a claim that CUDA is reimplemented.

## ADR-0003 — Radix discovery plus fixed-page storage

**Status:** accepted

Token radix trees efficiently discover reusable prefixes. Fixed-size page
arenas make GPU KV capacity deterministic and prevent fragmentation.

The prefix index stores generational page runs and never owns GPU tensors.
The page allocator remains authoritative for physical memory.

## ADR-0004 — AOT production kernels

**Status:** accepted

Production startup selects digest-pinned kernels from a capability manifest.
It does not compile because a request arrived. A developer JIT may be added
later but is disabled in production.

This makes readiness, latency, attack surface, and reproducibility tractable.

## ADR-0005 — Constrained LunaTile before universal DSL

**Status:** accepted

LunaFlux needs tile operations for a finite kernel catalog. It does not need to
rebuild TileLang or TVM before serving one model.

LunaTile grows only from measured kernel requirements. Vendor GEMM remains
valid when it is faster or more reliable.

`LunaTile` is a LunaFlux inference component, not part of the MoonBit language
or runtime. New first-party inference languages, compilers, artifact formats,
and operator-facing tools use the `Luna` prefix so their ownership is explicit;
the `Moon` prefix is reserved for the surrounding MoonBit ecosystem.

## ADR-0006 — One dense decoder family first

**Status:** accepted

The initial model plan supports one validated Llama-style dense decoder in
BF16. Breadth follows the immutable plan interface after correctness, paging,
batching, and streaming are proven.

This prevents hundreds of model conditionals from shaping the scheduler.

## ADR-0007 — One scheduler owner

**Status:** accepted

Request lifecycle, page ownership, prefix references, admission, and batch
membership are mutated by one deterministic scheduler owner. API and worker
tasks communicate through bounded typed channels.

This trades uncontrolled shared concurrency for reproducible decisions and
simple invariants while still overlapping CPU planning with GPU execution.

## ADR-0008 — Worker isolation per device

**Status:** accepted

Each accelerator is owned by one worker process. The worker contains CUDA
context and graph state. A worker failure invalidates a device generation
without corrupting scheduler memory.

The initial service has one scheduler process and one worker, not a process per
subsystem.

## ADR-0009 — Typed offline specialization, not source-driven execution

**Status:** accepted; implementation scheduled for Phase 5

Model-specific kernel specialization is controlled partial evaluation over
authenticated semantic inputs. A future LunaTile specializer receives typed
model, layout, execution-profile, target, and kernel-capability evidence and
may bake stable dimensions, offsets, layouts, tables, and entry points into a
content-addressed AOT artifact.

Generated C, CUDA, or another target language is an output representation, not
the type system or an execution authority. It cannot select a model, reinterpret
storage, acquire a device, or compile because a request arrived. Production
artifact admission requires the generated module, specialization record,
launch contract, compiler policy, and runtime capabilities to agree exactly.

Each specialized family requires an independent scalar referee, adversarial
and real-tensor differential evidence, a dispatch canary, declared numerical
semantics, logits/token comparison, and an end-to-end benchmark. This prevents
syntactic code-generation success or an isolated microbenchmark from being
mistaken for inference correctness or useful speedup.

## ADR-0010 — Local tensor parallelism is one rank-group capability

**Status:** accepted; implementation begins in Phase 7

Tensor parallelism preserves the single scheduler and canonical request
contract. Startup admits one explicit same-host rank ordering and derives one
immutable rank-group capability from the selected model plan, homogeneous
device targets, supported connectivity, per-rank memory envelopes, and exact
collective contract. Device discovery cannot silently reorder ranks or replace
an unsupported topology.

The model-family planning boundary owns row/column tensor placement and the
stable collective sites required by those placements. The scheduler continues
to publish one semantic schedule plan with one group generation; a private
execution bridge derives rank-specific views and collective sequence numbers.
No tensor-parallel or model-family branch enters admission, fairness, prefix,
sampling, or public request APIs.

Each rank retains one existing device-worker ownership domain, one local weight
arena, and one local KV arena. The loader reads only the authenticated source
ranges required by that rank directly into its final allocation; no rank may
materialize a complete copy of a tensor declared sharded. Logical page and
request ownership remain singular in the scheduler, while the rank-group owner
requires the same authenticated page-table transition to succeed for every
per-rank arena before publication.

Collective order is part of the immutable rank plan rather than backend
behavior. Every launch is authenticated by group generation, plan sequence,
collective sequence, operation identity, rank, and world size. Timeout, NCCL
failure, or loss of any rank invalidates the whole group generation, fails its
affected requests once, and drains every surviving rank without waiting for a
failed collective to make progress. NCCL handles and vendor diagnostics remain
inside a narrow private native ABI with explicit close ownership.

Initial support is local tensor parallelism only. Heterogeneous targets,
unsupported connectivity, non-divisible head or shard dimensions, pipeline
parallelism, and cross-node execution fail at startup. Broader layouts require
new positive capability and benchmark evidence; they are not compatibility
fallbacks.

## ADR-0011 — Numeric storage and operation execution are separate contracts

**Status:** accepted; implementation begins in Phase 8

Quantization is not a model-wide dtype switch. One complete model numeric
schema is composed from immutable per-tensor storage contracts and
per-operation execution contracts. Together they own weight representation,
weight and activation scale semantics, conversion rules, accumulation,
output, KV representation, codebook layout, and zero-point policy. The exact
composition is canonical, digest-bound, embedded in model specification and
plan identity, and admitted before any file, kernel, or device authority is
granted. BF16 is an explicit contract, not an implicit fallback.

Each stored tensor owns its exact dtype, scale granularity and layout,
zero-point policy, and codebook policy. Each semantic operation separately
owns an immutable execution-numeric requirement derived by its model-family
plan builder. Quantized projections may consume FP8 or I8 weights while
normalization, rotary, attention, residual, sampling, output, and KV paths
retain their declared representations. The core plan validates the complete
tensor table and ordered operation requirements but never derives precision
from operation kind alone. This prevents one global precision flag from
silently changing unrelated numerical behavior.

Operation-execution schema v2 distinguishes graph-bound activation input from
internal activation compute and makes activation-scale policy mandatory. The
initial finite E4M3 W8A8 contract is exactly BF16 graph input, finite E4M3
internal activation compute, dynamic per-tensor F32 activation scaling, finite
E4M3 tensor input, F32 accumulation, and BF16 graph output. Its operation
conversion describes the activation conversion; weight conversion remains in
the tensor storage contract. Future I8 weight-only execution therefore keeps
BF16 activation input and compute with absent activation scaling and exact
activation conversion. Operation and complete-schema canonical domains are v2;
v1 bytes are rejected rather than defaulted.

Every scale, zero point, or codebook stored with a model is an explicit typed
tensor reference with an exact shape and layout in the model plan. Actual
calibration values are not hidden in kernel manifests, global configuration,
or backend state. Kernel catalog and artifact records bind the exact
operation-numeric requirement, semantic capability, target capability, and
model identity. Hardware support is necessary but never sufficient: a format
is runnable only when schema validation, plan binding, AOT artifacts,
correctness evidence, and target admission all agree.

Model-family packages alone map authenticated family metadata into semantic
operations, numeric requirements, and tensor bindings. Scheduler, request,
prefix, KV ownership, and public service APIs remain architecture- and
quantization-neutral. New formats or families version canonical domains and
add positive capability evidence; they do not add aliases, default coercions,
or compatibility branches. The initial Phase 8 sequence keeps KV in BF16,
admits finite FP8 E4M3 W8A8 first, then symmetric I8 weight-only with
per-output-channel scales, and expands to another dense decoder family only
after those boundaries are proven.

## ADR-0012 — Rank-group replacement rebuilds one complete generation

**Status:** accepted; implementation begins in Phase 7

A local tensor-parallel worker is a generation-scoped rank group, not a set of
independently restartable processes. Loss, timeout, or collective failure at
one rank invalidates the NCCL membership and every rank-local execution owner
for that generation. Recovery drains and reaps the complete group before any
replacement is published. It never substitutes one rank into an existing
communicator or reuses a rendezvous identifier.

One private rank-group runtime owner retains a duplicated pair of approved
model and kernel roots, the authenticated immutable bootstrap source, the
generic planning inputs needed to rebuild every rank, and the exact accepted
scheduler predecessor sequence. For each initial start or replacement it mints
a fresh nonzero rank-group generation and fresh NCCL rendezvous identifier,
rederives every generation-bound group, device, execution, collective, KV, and
artifact digest, prepares all children, and publishes the group only after
every rank returns an exactly authenticated Ready response. Failed startup
remains owned cleanup authority; healthy drain, fault recovery, and final close
have distinct transitions. Approved roots close only after all child owners are
terminally reaped and no replacement can still be created.

The parent sends each child one bounded, canonical, versioned rank Configure
contract rather than an application-defined byte blob. The contract composes
the exact worker-startup frame, bootstrap-source frame, ordered selected local
topology declaration, and rank bootstrap envelope and binds their digests to
the outer rank-wire binding. Nested frame lengths, versions, reserved fields,
checksums, identities, generations, rank, world size, device ordinal, and
predecessor sequence are validated before root, device, artifact, or
collective authority is acquired. A replacement is derived from inert inputs;
captured bytes from an earlier generation are never replay authority.

The tensor-parallel bootstrap-source recipe owns one identical, rank-ordered
deployment-capacity declaration for the complete group. Each entry binds its
process-visible ordinal, memory ceiling, and exact weight, activation/workspace,
KV, and collective reservations. The same source also binds the accepted
collective-runtime version interval. Every rank startup in a group must carry
the same source digest; per-rank policy variants are invalid. Children combine
that source-owned declaration with an independently admitted topology, and
derive the complete KV plan from those facts without inspecting another
rank's shard or fabricating another rank's device plan.

The group bootstrap additionally binds the exact collective-runtime version
used to create its fresh rendezvous identity. Each child admits precisely that
version before opening a communicator. These public contracts use a neutral
collective-runtime version vocabulary; vendor loading and NCCL-specific ABI
policy remain private to the device implementation.

Every child independently reopens the two inherited approved roots, probes the
process-visible local device inventory, admits the exact selected topology,
loads and validates its AOT execution manifest, materializes only its declared
weight shard, authenticates the rank envelope against the rebuilt plans, and
opens rank-local device and collective resources before sending Ready. The
child cannot trust an ordinal or digest merely because the parent supplied it,
and ambient extra devices cannot silently change rank order.

WorkerService sees this owner only through the bounded physical-transport
contract and retains scalar flight identity. Scheduler, request, KV policy,
prefix policy, sampling policy, and public service APIs do not branch on rank
count, process topology, NCCL state, or model family. Rank-specific native and
wire failures are mapped to the neutral physical-transport failure vocabulary
at that boundary while exact diagnostic ownership remains private.

## ADR-0013 — External KV transport is a replaceable capability

**Status:** accepted; deferred beyond the local-runtime v1 boundary

LunaFlux will reuse an established external transfer and storage system such
as Mooncake when cross-node KV movement or disaggregated prefill/decode becomes
an admitted product capability. It will not reimplement RDMA transports,
distributed object storage, metadata discovery, or cross-node replication in
the LunaFlux request path. The current v1 runtime remains local: this decision
does not add a network fallback, cross-node authority, or an unproven release
claim.

The integration boundary is a small provider-neutral private capability, not
Mooncake types in the scheduler or public API. LunaFlux remains authoritative
for request identity, logical page ownership, prefix reuse, eviction, page
generation, and local arena lifecycle. The external provider may move or retain
only immutable, fully published page runs whose authenticated transfer contract
binds the model, tokenizer, numeric and layout identities, page geometry,
source generation, destination generation, byte range, and content digest.
Publication at the destination remains a LunaFlux page-table transition after
transfer verification; transport completion alone never publishes ownership.

Mooncake C++, Rust, Python, metadata, credential, topology, and fleet-service
types may not enter first-party control-path packages. A future implementation
must place any native client behind a narrow private ABI or use a separately
owned sidecar protocol with bounded canonical messages and explicit close
semantics. The deployment environment owns provider configuration, service
discovery, credentials, and lifecycle. The scheduler observes only neutral
capability, admission, transfer, cancellation, and terminal-result vocabulary.

Startup must positively admit the exact provider protocol and implementation
version, topology, integrity mode, capacity ceilings, timeout policy, and
required feature set. Missing or mismatched capability fails startup when the
model plan requires external transfer. No silent TCP, local-copy, or eager
recompute fallback is allowed unless that alternative is separately declared,
budgeted, benchmarked, and digest-bound in the accepted plan. Provider failure
cannot mutate local ownership: incomplete destinations are discarded, source
ownership remains explicit, and retry authority is bounded and generation
scoped.

Enabling this capability requires a new phase with hostile protocol tests,
deterministic cancellation and cleanup evidence, corruption and stale-generation
rejection, bounded-memory proof, cross-node physical validation, and workload
benchmarks. Until those gates pass, LunaFlux exposes no disaggregated-serving or
external-KV readiness claim.

## ADR-0014 — Embedding OpenAI authority is distinct from qualification and CLI admission

**Status:** accepted; embedding composition only

LunaFlux has one implemented OpenAI Responses server owner, but three callers
have different authority: a validation campaign, an embedding deployment, and
the one-argument production CLI. They must not share a readiness token merely
because they reuse the same codec and listener mechanics.

Qualification remains loopback-only and publishes
`OpenAiQualificationReady`, which is explicitly excluded from production
readiness. An embedding deployment may instead build an opaque
`LunaApiAuthPolicy`, explicitly attest that it owns an authenticated external
ingress, and pass those values in a distinct `RuntimeOpenAiProductionPolicy`.
The production policy has no conversion from qualification policy or evidence.
Only this distinct capability may consume the singular prepared service into
the OpenAI server and publish the owner's ordinary `Ready` phase.

The embedding production binding is restricted to an already-admitted
loopback listener. The approval value is a caller authority assertion, not a
TLS certificate, credential resolver, network probe, or readiness result.
LunaFlux still authenticates inference requests with the caller-built Bearer
policy. The embedding owner remains responsible for external TLS, public
network policy, authenticated drain invocation, and the lifetime of the
surrounding process. LunaFlux exposes only loopback observational
health/readiness routes; their admitted address plus runtime health, readiness,
metrics, and `begin_drain` remain owned by the same singular
`RuntimeInstanceOwner`. No second mutable lifecycle owner is introduced.

This decision does not authorize the one-argument CLI to source a secret from
argv, environment, the instance-policy document, or an implicit filesystem
location. Standard instance admission therefore remains fail-closed for
OpenAI policy while the production CLI lacks a separately designed credential
and control-capability transport. The inference HTTP parser remains scoped to
inference routes. A separate loopback owner now serves observational health and
readiness routes; no HTTP drain handler exists.

Promotion of the opaque CLI/deployment path requires a later decision that
selects a non-ambient secret/control transport, binds exact control routes to
the singular owner, authenticates drain, proves listener-first shutdown, and
passes deployment-level TLS, restart, and hostile-boundary tests. Embedding
readiness is not evidence for those gates.

## ADR-0015 — Opaque CLI drain uses one inherited local capability

**Status:** accepted; local capability implemented

An embedding caller that exclusively possesses the singular
`RuntimeInstanceOwner` invokes `begin_drain` directly. The opaque one-argument
CLI instead requires one deployment-created, preconnected Unix stream socket
at inherited descriptor 5. Possession of that descriptor is the local,
kernel-mediated drain capability. LunaFlux validates that the descriptor is a
connected Unix stream, moves it to a close-on-exec nonblocking descriptor, and
closes the inherited number before constructing model, device, or listener
authority.

The capability speaks exactly one fixed v1 command and one fixed response.
The only command requests drain. Its response distinguishes a newly accepted
request, an already-draining owner, and an already-closed owner. Fragmentation
is supported without allocation in progress; malformed, truncated, trailing,
or replayed input terminates the channel. A valid request is latched before
the singular runtime owner begins drain, so later peer loss cannot undo it.
The runtime flips readiness before listener cleanup, closes inference and
observational listeners and service resources deterministically, writes the
bounded response when the peer remains present, and closes the drain
descriptor last before publishing `Closed`.

LunaFlux owns only validation and bounded parsing of this inherited local
capability, its mapping into the singular lifecycle owner, and deterministic
descriptor closure. It exposes no drain path or port, no HTTP drain route, no
generic command bus, and no token in argv, environment variables, global
state, or an implicit file. Operational `/healthz` and `/readyz` remain
observational. An instance-policy drain path, when present, can describe an
embedding or external proxy contract; it does not create a LunaFlux HTTP
handler.

The deployment embedding, including LunaNexa when used, owns creation and
exclusive transfer of the socketpair, external caller authentication and
authorization, generation fencing, audit, process identity, public routing,
and TLS or mTLS termination. LunaFlux does not import deployment-product types
and does not claim public reachability or TLS from descriptor possession. The
separate inherited credential capability is specified by ADR-0016 and is never
combined with this drain channel.

Promotion evidence for the opaque deployment path includes the native ABI
inventory and sanitizer gates, missing and wrong-descriptor rejection, exact
fragmented protocol and response tests, malformed/truncated/trailing/replay
tests, peer-loss and idempotence tests, allocation-free polling evidence, CLI
activation ordering, and deployment-level tests proving authenticated external
control, TLS, generation fencing, restart, and listener-first shutdown. The
local capability evidence alone does not satisfy those external gates.

## ADR-0016 — Opaque CLI inference authentication uses a read-once descriptor

**Status:** accepted; local capability implemented

The one-argument CLI accepts inference credentials only through a separate
deployment-created, connected Unix stream at inherited descriptor 6. It is not
the descriptor-5 drain channel and is not a generic command bus. LunaFlux moves
the descriptor to a close-on-exec nonblocking owner, admits one fixed versioned
frame with a bounded nonempty credential and write-side EOF, closes the channel,
copies the value into the existing opaque constant-work Bearer policy, and
wipes the source buffer before model, device, worker, or listener construction.
That policy owns one idempotent full-buffer wipe shared by every verifier alias.
HTTP operation reuse wipes its used request cells, terminal close wipes its
complete fixed request storage, and OpenAI server/pool drain plus all startup
rejection paths propagate the singular close before abandoning ownership.

Instance-policy schema v3 binds the complete OpenAI Responses construction
contract: credential ceiling, HTTP and codec work bounds, prompt-template
segments, model alias, response identifier prefix, cache scope, output/context
ceilings, seed, and deadline. The actual inherited credential length must fit
that authenticated ceiling. V1 and native v2 remain native-framed and accept no
credential; v2 OpenAI intent remains fail-closed because it lacks this complete
construction contract.

The singular runtime owner selects native-framed or loopback OpenAI Responses
from authenticated policy, binds the existing observational health/readiness
listener, and still requires the independent inherited drain capability before
opaque-CLI progress. Its protocol projection carries no secret or routing
authority. LunaFlux does not read secrets from argv, environment, policy files,
or implicit filesystem locations and does not claim TLS or public reachability.
The deployment environment owns descriptor creation and exclusive transfer,
external TLS/authentication, routing, generation fencing, audit, and restart.

## ADR-0017 — Model-family expansion uses typed workload plans

**Status:** accepted; implementation in progress

DeepSeek V4, GLM 5.3, GLM 5.3 Flash, and MiniMax H3 cannot be made correct by
adding aliases to the dense Llama plan. They require three distinct workload
topologies: decoder-only text generation, multimodal conditional generation,
and audio/video diffusion. LunaFlux therefore expands model support through
immutable typed workload plans rather than family switches in the scheduler,
KV owner, device worker, or kernel catalog.

The decoder workload may grow reusable operations for routed and shared
experts, multi-token prediction, manifold-constrained hyper-connections,
compressed or dynamic sparse attention, and linear recurrent attention. Those
operations carry exact shapes, persistent-state contracts, numeric schemas,
workspace bounds, and positive kernel capabilities. A family builder composes
them into a plan and then disappears from the request path. Existing dense
plans remain valid and do not acquire optional family fields.

Multimodal conditional generation owns a typed preprocessing plan that maps
bounded image or video inputs into embeddings before entering an admitted text
decoder plan. Audio/video diffusion owns a separate pipeline plan for text and
reference conditioning, latent preparation, denoising timesteps, transformer
execution, and video/audio VAE decoding. It does not pass through the token
scheduler or pretend that diffusion steps are decode tokens. Each workload has
its own bounded request and output protocol, while sharing architecture-neutral
device allocation, tensor materialization, AOT artifact admission, execution
telemetry, cancellation, and explicit resource release.

Numeric storage and packed-row order are plan inputs when they affect kernel
ABI. MiniMax H3 therefore records separate F32-input/BF16-output latent
projection and F32-input/F32-output head contracts, frame-major
channel/patch-vector ordering, and channel-major audio rows in the
family-neutral diffusion plan and requirement digest. Decoder families share a
no-bias BF16 projection renderer, while fixed-row diffusion projections share
F32+bias renderers and their adapters retain exact profile geometry and
artifact evidence.

Normalization and sparse-index preprocessing remain explicit operations rather
than incidental model-family branches. Advanced decoders may use a paired BF16
RMSNorm source with independent query and key widths; diffusion uses a
request-specialized fixed-row BF16 RMSNorm source. A separate family-neutral
affine source consumes already row-aligned BF16 shift and scale, performs staged
BF16 AdaLN arithmetic, and widens to F32. A second generic affine source owns
the preceding one-row F32-SiLU/BF16-dense production of shift and scale;
request admission separately owns digest-bound packed timestep and modality
row selections plus finite F32 distinct timestep values. One generic source
renders exact flipped cosine/sine frequency rows and another performs the
bit-exact modality gather. A third generic source renders the exact ordered-F32
`256→5376→2688` dense-SiLU-dense timestep MLP. A fourth renders one BF16
shift/scale table row for every authenticated distinct F32 conditioning row;
an inert typed composition checks model, requirements, timestep values, row
count, ordinal order, and adjacent widths across all three stages. Live buffer
construction and handoff into packed modality gathering remain separate unmet
capabilities. A separate generic gated-MLP source owns bias-free packed gate/up
SwiGLU and dense-down arithmetic. The MiniMax adapter only authenticates its 50
denoiser and 2 refiner scopes, exact weight layouts, and request geometry; it
cannot absorb residual, normalization, attention, or AdaLN authority. GLM hybrid
DSA pooling owns a generic standalone source contract for stable complete-pool
compression. Projected scoring, stable top-k, Flash visible-tail selection, and
Full-profile reuse likewise have separate correctness-only source contracts. A
generic prefix-RMSNorm source normalizes only an authenticated prefix and may
copy a disjoint suffix bit-exactly; the GLM adapter fixes the 512-wide prefix,
Full 64-wide suffix, Flash zero-suffix ABI, epsilon, and downstream KV-B join. A
serial selected-index sparse-attention source owns only projected Q/K/V,
selected-index validation, causal stable softmax, and PV; it cannot absorb DSA
projection, paged-KV, or index-production authority. DeepSeek token-hash lookup
is a generic fail-closed I32 operation. A separate family-neutral startup owner
performs checked I64-to-I32 narrowing with row uniqueness and explicit release,
while the thin DeepSeek adapter selects the exact authenticated checkpoint
tables and projects their three checked sidecars into a generic segmented
device upload. A separate family-neutral operand-region planner binds exact
roles, byte counts, alignments, allocation identities, offsets, capacities, and
non-overlap without retaining pointers or allocation ownership; the thin
DeepSeek join uses it to complete all four token-hash operand descriptors for a
standalone inert plan while remaining non-runnable. This evidence cannot
complete a model artifact when an earlier required operation is unsupported.
Stable biased top-k selection and selected-weight finalization are also
separate generic operations so hash and non-hash routing share finalization
without sharing selection semantics.
A generic scaled-RoPE source likewise owns only adjacent-pair BF16 rotation;
the DeepSeek adapter fixes the official compressed YaRN parameters and
RoPE-only query/KV geometry. Plain-theta sliding-layer selection remains a
separate requirement instead of being inferred inside that source.
DeepSeek query low-rank projection is likewise split into ordered A and B
requirements around query RMSNorm. The checkpoint's E4M3 payload and UE8M0
scale grids are authenticated, but no existing BF16 or scalar-F32-scale
renderer may impersonate UE8M0 exponent decoding, special values, or
dequantization; both projections therefore remain exact typed gaps.
DeepSeek mHC is represented by four versioned, ordered operations: block-control
construction, pre reduction, post residual combine, and head reduction. This
keeps control projection, sigmoid gates, Sinkhorn normalization, stream mixing,
and attention/FFN placement explicit in plan and numeric identity. The former
generic Sinkhorn/mix requirement is not used to infer a CUDA ABI. Block control
now has its own exact family-neutral BF16/F32 renderer and eight-operand ABI,
including distinct F32 pre/post/combination outputs. Pre-reduce has a separate
four-operand renderer and ABI for counts, BF16 stream state, F32 pre-mix rows,
and BF16 reduced rows, with ordered F32 accumulation and one final round.
Post-combine has a separate
six-operand renderer for counts, BF16 branch/state inputs, F32
destination-major/source-minor controls, and BF16 combined state. Admission
authenticates those three phases. Head-reduce has an independent six-operand
BF16/F32 renderer, including function/base/scale inputs and one BF16 head
output. Admission authenticates all four phases, then accepts the replicated
Query-A and separate replicated pre-normalization KeyValue block-FP8
candidates before rejecting Query-B. Separate inert joins authenticate the
official layouts without granting interpretation, materialization, upload, or
launch authority; later phases cannot inherit or reinterpret any earlier ABI.
A separate family-neutral I32 renderer owns ratio-based deterministic
compressed-index counts and dense sentinel-padded slots. The DeepSeek adapter
binds official ratio 128 and maximum position geometry without absorbing
compressed attention, KV allocation, or execution authority. The downstream
path is represented by four truthful phases rather than a fused placeholder:
window/compressed index join, non-overlapping shared-KV preparation, selected
sparse attention, and output inverse-RoPE. A family-neutral selected-attention
source owns only the shared BF16 K/V lookup and stable ordered-F32 softmax; its
F32 per-head sink contributes to the denominator but never the value numerator.
A separate family-neutral I32 join source preserves the official causal
prefill padding, decode circular-ring ordering, compressed sentinel padding,
and literal window-then-compressed append without sorting or deduplication. A
separate seven-operand borrowed-buffer renderer owns non-overlapping ratio-128
preparation: full ordinary BF16 K/V followed by completed compressed rows in
prefill, and the physical circular-window slots followed by the completed
compressed prefix in decode. A separate three-operand source applies the exact
in-place inverse scaled-YaRN transform to the final 64 components of each
512-wide attention head while retaining the 448-wide prefix bit-exact.
Learned compression and cache-state mutation remain typed gaps. Output
projection is split at the official low-rank boundary: a family-neutral grouped
BF16 renderer owns converted `wo_a` with exact `[row,group,4096] ->
[row,group,1024]` geometry. A separate family-neutral `wo_b` renderer owns the
single-rank E4M3/UE8M0 storage boundary: dynamic per-row/block-128 UE8M0
activation scales, E4M3 weight codes, UE8M0 weight scales, ordered F32 block
accumulation, and one BF16-RNE output. The same family-neutral renderer owns the
replicated Query-A projection with profile-specific `4096→1024` or
`7168→1536` geometry. DeepSeek-only startup joins authenticate Query-A,
Output-A conversion, and Output-B source storage/layout against their candidate
operands without acquiring conversion, materialization, upload, tensor-parallel
collective, live KV, or launch authority.
A separate generic interleaved-RoPE source owns bit-exact no-PE query-prefix
copy plus query/key adjacent-pair rotation. The GLM adapter fixes Full-profile
theta, head/suffix geometry, input/output pair layouts, and its six operands;
Flash's no-RoPE requirement stays unlowered rather than becoming a fake no-op.
A generic staged joint-attention source owns only Q/K/V projection, Q/K
normalization, optional authenticated rotary, full noncausal unmasked softmax,
and output projection. The MiniMax adapter fixes packed/refiner row scopes,
epsilon, rotary width, exact weight layouts, and BF16/F32 stage ordering;
residual, pre-attention modulation, and gates remain separate operations.
A second generic gated-MLP renderer owns separate gate, up, and down BF16
weights. GLM uses this form with exact hidden/intermediate geometry and staged
BF16 round points; MiniMax retains the packed `[up;gate]` renderer. Neither
adapter may reinterpret one storage layout as the other.
A separate family-neutral causal short-convolution renderer owns one KDA
stream at a time. The GLM Flash adapter reuses it independently for Q, K, and
V, fixing 8192 channels, kernel width four, CSR sequence boundaries, and an
explicit oldest-first three-token BF16 history handoff. Projection, recurrent
update, residual/norm, and live cache ownership are not folded into this
candidate. A separate projection renderer owns BF16 Q/K/V, low-rank forget
features, raw forget rows, and beta without synthesizing the F32 decay control.
A second family-neutral renderer owns the serial normalized
recurrent-delta transition with BF16 Q/K/V and beta, per-component F32 log
decay, explicit F32 initial/final state, and one BF16 output round. The GLM
adapter fixes 64 heads by 128 components and CSR sequence boundaries. A
bias-free two-stage renderer owns the output-gate projection, and a separate
per-head renderer owns strict F32 RMS normalization followed by the official
sigmoid gate and one BF16 output round. A family-neutral decay renderer owns
only BF16 forget logits plus F32 bias/rate tensors through the safe-lower-bound
sigmoid transform, with F32 output. The existing bias-free dense renderer owns
the final `[8192,4096]` projection. A separate family-neutral router renderer
owns only bias-free F32-input/BF16-weight projection to raw F32 expert logits;
the Full and Flash adapters fix `[256,6144]` and `[288,4096]` checkpoint rows.
A distinct family-neutral renderer applies F32 sigmoid and correction bias into
separate uncorrected-score and corrected-choice-score outputs; duplicate
downstream sigmoid is forbidden. A further family-neutral serial source uses
the corrected scores for top-two group scoring and group/expert choice, then
uses only the uncorrected scores for `sum+1e-20` normalization and scaling.
Lower expert IDs deterministically break ties, without claiming parity with
PyTorch's unsorted `topk`. Three further family-neutral candidates own selected
routed-expert SwiGLU, shared-expert SwiGLU, and the ordered BF16 routed/shared
sum. An exact indexed-parameter-bank layout converts the official per-expert
checkpoint vocabulary into six canonical contiguous banks, and a GLM-only
startup join authenticates those banks against the candidate operands without
materializing or uploading them. Full then applies a separate non-aliasing BF16
outer residual add. Flash applies its exact four-stream mHC composition,
`post * branch + combineᵀ * residual`, with controls rounded to BF16, ordered
F32 matrix accumulation, and one BF16 output round. Flash hyper-control now has
two separate family-neutral correctness candidates: learned function/base/scale
projection with collapse, and positive-matrix Sinkhorn with one initial column
normalization followed by 19 row/column passes. Their intermediate/output
buffers are explicitly non-aliasing. Live cache ownership, physical artifacts,
launch authority, and throughput CUDA remain separate.
A separate family-neutral BF16 K/V assembly source consumes the existing KV-B
producer. Full broadcasts the rotated shared 64-wide suffix across 64 heads and
appends it after each 192-wide no-PE key while splitting 256-wide values; Flash
uses a distinct no-suffix ABI to split its 256+256 packed rows. Both produce the
contiguous K/V operands already required by sparse attention, but grant no live
cache handoff, allocation, or execution authority.
A complete MiniMax VideoVae schema binds the official config, index, three
shards, exact 703 F32 names/shapes/dtypes, and 10,415,475,936 payload bytes.
The existing streaming host materializer may admit that component without a
full payload copy; a metadata-only fixture exercises its bounded arena plan.
The AudioVae schema likewise binds its official config and single authenticated
safetensors header to exactly 1,087 F32 tensors and 605,306,340 payload bytes;
bounded host materialization grants no VAE encode/decode authority. Separate
family-neutral F32 channel-major affine renderers now own only the official
inverse latent normalization prefixes: VideoVAE `[1,24,37,48,84]` and AudioVAE
`[2,32,207]`, with explicit decoder batch two for stereo and ordered
multiply-then-add arithmetic. A separate family-neutral channel-major renderer
owns the exact F32 AudioVAE `dec_in_proj` pointwise Conv1d from 32 to 2,048
channels, including its bias. Another generic renderer implements the released
weight-normalized kernel-7 `decoder.conv_pre` as ordered F32 normalization into
an explicit `[1024,2048,7]` workspace followed by padded channel-major Conv1d
to `[2,1024,207]`. The thin MiniMax adapter authenticates `weight_g`,
`weight_v`, bias, per-output-channel norm axes, and stage order. A further
generic renderer owns the immediately following `decoder.ups.0.0` kernel-9,
stride-5 ConvTranspose1d, using PyTorch weight normalization's per-input-channel
axes to produce `[2,512,1035]` through an explicit normalized workspace. A
separate alias-free activation renderer owns the immediately following
`decoder.resblocks.0.activations.0`: replicate-padded depthwise transposed
convolution to 2,070 time cells, F32 SnakeBeta, then replicate-padded stride-2
depthwise convolution back to 1,035. Its upsample and downsample filters remain
distinct operands. The next exact AudioVAE boundary is one complete three-way
AMP stage: 126 ordered F32 parameter regions and 13 workspace regions feed 97
typed launches covering all three residual blocks, including weight-normalized
dilated same-length convolutions, ordered residual additions, and the final
divide by three. The original upsampler output and precomputed first activation
remain distinct inputs. The following `decoder.ups.1.0` is now an exact
per-input-channel weight-normalized kernel-9, stride-5 ConvTranspose1d
transition from `[2,512,1035]` to `[2,256,5175]` with an explicit normalized
workspace. The following `resblocks.3–5` stage is also exact: a generalized
family-neutral AMP contract binds the precomputed first activation, 126 ordered
parameter regions, kernels 3/7/11 with dilations 1/3/5, 97 launches, and a
119,465,984-byte disjoint workspace before ordered division by three.
The following `decoder.ups.2.0` kernel-4/stride-2 transition to
`[2,128,10350]` and its first `resblocks.6.activations.0` alias-free activation
are now exact, with per-input-channel weight normalization, distinct length-12
up/down filters, and explicit intermediate workspaces. VideoVAE decode,
complete `resblocks.6–8`, `decoder.ups.3.0`, later AudioVAE stages, and the final
output path stay typed gaps, so conditioning records
missing execution rather than missing VAE architecture.
A family-neutral conditioning reference assessment records why MiniMax media
conditioning is not yet an AOT candidate. Its digest binds request and
requirement identities, derived geometry, workflow, and ordered missing VAE,
noise, row-span, and reference-association prerequisites. It deliberately owns
no source, compiler, artifact, device, or execution API, and its require
operation always fails typed.
Every adapter binds exact operands and geometry while all source candidates stay
non-bindable until offline compilation and physical qualification.

Candidate completeness is audited by a family-neutral startup-only structural
owner. It canonically binds one exact requirement digest to unique candidate
ordinals and reports exact missing and non-bindable ordinals. Even a complete
set of caller-declared bindable claims is not artifact admission or execution
authority; those remain separate opaque capabilities.
Typed candidate-to-evidence adapters are split by workload vocabulary so an
advanced-decoder consumer does not acquire GLM-hybrid or joint-diffusion
dependencies. Integration packages may join family-owned authenticated storage
to generic candidate and artifact evidence, but must expose missing compiler,
loader, launch, and physical authority as explicit failures.
Family-neutral offline compile evidence may additionally prove that a canonical
receipt and two module snapshots are byte-identical and agree on source, recipe,
compiler policy, driver identity, target, symbol, and module digest. That
self-consistency is not builder authentication or proof that the named compiler
executed, so producer, compiler-execution, and execution authority stay typed
failures.

An engine instance selects exactly one authenticated workload plan at startup.
The selected workload determines which request codec and execution owner are
constructed; it cannot change per request. Core lifecycle code consumes an
opaque prepared-workload capability and never imports a model-family package.
Model-family packages import only public plan vocabularies and weight-schema
owners. A family may own a thin startup upload adapter that imports its host
arena owner, the family-neutral segmented upload owner, and the public device
context. The segmented upload owner and scheduler, KV, API, device, and kernel
implementation packages never import DeepSeek, MiniMax, or GLM packages.

Every new operation remains inert until an exact numeric contract, weight
materializer, reference implementation, AOT artifact and launch contract,
device executor, deterministic correctness corpus, resource-balance campaign,
and benchmark gate all agree. Semantic recognition, a typed plan, or a family
weight table alone does not grant executable readiness. Unsupported profiles,
missing workload protocols, missing kernels, and partial capability sets fail
at startup without dense-model fallback.

This decision expands the post-v1 architecture; it does not retroactively
change the supported-v1 capability or authorize arbitrary graph execution,
runtime Python, remote model code, runtime JIT, hidden family branches, or a
global mutable runtime context.

## ADR-0018 — Two-node execution preserves the functional planning boundary

**Status:** accepted for the `twospark` workstream; live transport and physical
qualification pending.

An explicitly assigned pair of DGX Spark nodes connected through ConnectX-7
is one inference instance with one scheduler and one worker per accelerator.
The full contract and implementation gates are in [TWO_SPARK.md](TWO_SPARK.md).
This is a new post-v1 network capability, not an exception to ADR-0010's
same-host peer-access admission.

Model-family builders continue to produce immutable tensor-parallel plans.
A separate pure compiler joins those plans with node-qualified device
identities and per-node memory budgets, reusing canonical aligned KV layout
planning. It binds its complete result to one startup-only digest. Node-local
ordinal zero on two different nodes never becomes a fictitious local GPU set.
No model, topology, or network branch enters scheduler policy or LunaTile IR.

Root-free startup envelopes compose existing worker contracts under one
both-rank generation identity. Filesystem authority remains node-local.
Transport authentication, live admission, and explicit native ownership are
separate from these inert values. A pure local-clock lease determines when a
remote resource owner must fence itself; timeout cannot resurrect a generation
and an unreachable peer is not proof of cleanup. Runtime integration reuses
the existing group failure/retirement contract, rank child execution machine,
worker service, and online API owners. Separate authenticated release receipts
gate recovery; failed replacement resources retain their own cleanup obligation.
Artifact verification and deployment digests stay at startup. The scheduler
uses fixed remote mailboxes; TLS and heartbeat tasks stay outside it. Physical
qualification and the pinned TLS server provider remain release gates.
