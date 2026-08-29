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
