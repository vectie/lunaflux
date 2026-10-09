# LunaFlux detailed implementation plan

## Working policy

Current priority (2026-10-10): executable GLM-5.3 Flash, DeepSeek-V4 Flash DSpark
and MiniMax-H3 model features. Additional TLS/admission expansion and unrelated
warning migrations are paused. Retain existing ownership guarantees and focused
tests, but do not make new hardening work a prerequisite for model execution.

Activation transfer now has a plain framed-TCP/pinned-DMA execution owner and a
GLM stage-bound lease. Producer retirement precedes download; complete receiver
upload precedes consumer eligibility. Both composed 45-layer CUDA source halves
compile on GB10; loopback device-double transfer and GLM lease regressions pass.
Rank-local request metadata, remote prepare/run/commit/release and terminal token
delivery now connect through a model-neutral two-rank coordinator. It waits for
both stage retirements and activation upload before committing global history.
The expanded final affected native aggregate passes 106/106. Async fixtures
serialize only their shared native test-double device
lane, not production execution. The checkpoint-backed native diagnostic now
connects actual-footprint placement/export and both prepared ranks; next run it
against actual checkpoint weights under measured host budgets. It consumes
canonical pretokenized frames, not yet an autoregressive generation frontend.
The installed real ModelOpt NVFP4 config now passes the config-only diagnostic;
its storage descriptor and explicit RMS epsilon are validated. Reader/upload
and config regressions pass 23/23. Inventory can live outside the read-only
checkpoint mount. Next verify actual tensor shapes and placed bytes before any
GPU weight allocation; config acceptance alone is not a numerical result.
This correctness transport is host staged,
not GPUDirect/RDMA, and its async polling is not claimed heap-free. CUDA source
compilation is not a whole-model numerical or performance test.

Prepared GLM stages now assemble text boundaries + contiguous decoder + residual
owners with aggregate pre-upload memory accounting. First/last stages load only
their own global weights. A generic pure partitioner checks exact layer costs
against heterogeneous host budgets and preserves complete ordered coverage.
Partial decoder slices no longer bind whole-model worker delivery; complete
stages bind only the request ports they were prepared against. The official-
shaped first-stage native fixture passes 32 steps / 864 launches without warm
heap allocation or blocking synchronization. It is a device-double/zero-weight
fixture, not actual checkpoint GPU execution. Rank-local request ports and
residual transfer/completion are connected in loopback tests; next start these
owners against actual checkpoint stages on the two hosts and validate full
checkpoint prompt-to-token generation. DeepSeek/MiniMax completion remains open.

Contiguous GLM decoder slices now compose the official KDA/DSA and dense/MoE
schedule into one startup-bound queue. Layer frames expose borrowed launches
instead of forcing layer-owned completions; at most two residual buffers are
reused across layers. Aggregate checkpoint/bank/state/workspace accounting occurs
before upload. The current DSA history is single-request, so slices containing
DSA explicitly retain that restriction. Next wire embedding, final normalization,
head/sampling and worker request execution; cross-device handoff and real-weight
GPU equivalence remain separate unfinished features. Do not put TLS/signing work
ahead of those integrations.

GLM text ingress/egress now binds global embedding/norm/head weights to a generic
immutable text precision plan and prepared AOT frame. Its prefix/suffix can join
the decoder slice queue with no additional per-step completion. Selected head
rows bound prefill logits memory independently of input length. This is the
unweighted-mean/BF16 reference path, not DeepSeek's learned head or a throughput
claim. A small GB10 text numerical/deterministic/sanitizer fixture now passes;
real-weight and full-dimension text validation remain. The single-request worker
bridge now delivers output/error completions through the existing wire protocol
with zero measured warm heap allocations. Next assemble text + all model layers
in the actual worker startup path and compose model-wide cross-host stage
placement. The slice binding alone does not run an entire checkpoint. DeepSeek's
learned head and MiniMax's bounded offload/execution composition remain separate
feature work; no new TLS/admission work is needed.

The compact MoE Program now owns startup buffers and binds routed/shared experts
to one reusable execution queue. GLM shared BF16 and DeepSeek shared FP8 adapters
use the same precision/execution layers. Next connect the programs into complete
decoder blocks and whole-model step execution, respecting the per-host memory
budget before loading model banks. Module tests do not close whole-model serving
or two-host execution tasks; those require actual execution rather than labels.

Automatic router projection/score/grouped selection is now an optional prefix
of that same queue. GLM checkpoint router weights and correction vectors bind
without rank/dtype reinterpretation. Remaining wiring includes GLM hybrid
attention state and dense early layers, DeepSeek compressed attention,
whole-model layer ownership, and the existing worker's
non-Llama/Qwen execution dispatch. Do not add transport-hardening prerequisites.

DeepSeek token-hash selection is now another prefix of the same MoE Program:
project BF16 router weights, compute sqrt-softplus, preserve token-table IDs,
gather/normalize learned scores, then execute experts. The existing startup I64
to I32 table conversion and device sidecar arena remain the source of tables;
the Program accepts per-layer offsets and does not scan or copy tables per step.
Whole-model execution still needs to connect this prepared program to layer
state and actual model loading; passing module tests does not finish that task.

The request-owned recurrent delta Program now executes KDA state transitions
through a generic precision plan and AOT CUDA lowering, with independent request
slots and no per-step heap allocations. Numerical, memory/race/synchronization
and bounded component timing checks pass; the serial reference is not a serving
benchmark. Request-owned Q/K/V short convolution now joins that same queue;
the three histories and intermediate buffers have explicit aggregate budgets.
Chunked prefill and single-token decode produce the same output/state bitwise
on three tested GPU geometries, including GLM's actual channel count. The complete
hidden-to-hidden KDA branch is now prepared as one 16-launch queue, including
parallel projections, decay/beta controls, output gating and normalization.
Its small GPU full-chain oracle and chunked execution checks pass, as do 47
focused native tests. The checkpoint-backed KDA layer owner now binds all fifteen
BF16/F32 parameters, preserves rank-three convolution layout and streams into
owned device allocations through one reusable host chunk. An official-sized
native fixture exercises that owner and its complete queue. Its attention mHC
envelope is now executable too: function/collapse, Sinkhorn and input RMSNorm
precede KDA; residual publication follows it in the same 20-launch queue with
one completion boundary. The GLM adapter streams all nineteen physical weights,
and a bounded small GPU full-chain oracle passes. The early dense decoder block
now adds its second mHC envelope and separate-weight staged SwiGLU to that same
queue: 26 launches, one completion, 26 checkpoint weights. Its small GPU oracle
passes, with bitwise prefill/decode agreement and clean memory/race/sync checks.
The recurrent MoE block now composes the same envelopes with automatic routing,
all 288 packed NVFP4 experts and the BF16 shared expert in one 34-launch queue.
The official-sized native checkpoint fixture streams all physical planes and
executes 32 steps without measured step heap allocations or blocking waits.
This is a native device test double, not a GPU checkpoint numerical test. Next
execute the composed MoE layer on GPU with real weights, finish DSA numerics,
and connect layer/worker execution.
Projected sparse attention now has a generic immutable precision plan and one
prepared pooling/index/attention launch frame. Query capacity is independent of
history capacity; IndexShare consumers borrow the device selection and execute
only attention, with no owned pool/index workspace. Its bounded GPU fixture
passes exact prefill/decode output checks and memory/race/sync checks. This does
not yet include GLM projections or complete DSA blocks. Request-owned projected
history now prepends reserve/reset, parallel BF16 append and publication to the
same queue. Chunked prefill/decode, independent owners, reuse, idle preservation
and full-capacity/error behavior pass a bounded GPU fixture. Next connect GLM
complete DSA blocks and model/worker scheduling; these are
feature integration work, not new deployment-hardening prerequisites.
The rope-free low-rank Q/K/V and index projection chain is now executable from
hidden rows: eight projections, two RMSNorm operations, affine index LayerNorm
and packed-KV splitting. One startup-bound frame joins it with retained history
and attention in eighteen launches, with one caller completion. The generic
precision layer owns dimensions/rounds/budgets; the GLM Flash adapter binds the
twelve exact checkpoint parameters and selects its zero-rotary geometry. Small
GPU tests cover all thirteen intermediate buffers and the complete projected
history/attention chain. This does not yet execute a real GLM checkpoint or its
whole-model worker. Output projection and mHC/FFN composition now exist:
`IndexedSparseDecoderPrecision` computes the complete attention envelope and
intermediate budget; its prepared frame contributes twenty-three launches.
`GlmSparseLayer` adds the shared routed/shared FFN implementation and second
envelope in one thirty-seven-launch queue. It binds learned output and pool APE
weights instead of supplying synthetic values. Model-specific index numerics,
real-checkpoint GPU validation and whole-model/worker scheduling remain. Do not
insert transport tasks ahead of those actual feature integrations.
Correctness-first ordered projections still need a throughput schedule. DSA and DeepSeek
compressed attention, MiniMax execution and two-host collectives remain actual
implementation tasks, not transport-hardening prerequisites.

Active compiler work: [counter-driven repair, 2026-09-29](COMPILER_COUNTER_REPAIR_2026-09-29.md).
The three-kernel diagnosis is recorded there with implemented changes,
remaining algorithm work and the bounded physical regression procedure.

LunaFlux is built as vertical, measurable phases. A phase is complete only when
its named behavior and failure gates pass. Later feature work must not bypass
an earlier correctness or ownership gate.

Every phase preserves:

- MoonBit ownership of the serving control path;
- no LunaNexa, MoonGate, Python, PyTorch, or TVM runtime dependency;
- deterministic scheduler and KV ownership;
- private CUDA ABI with explicit native resource release;
- typed startup capability validation;
- raw-payload-free default logs;
- checked native build and black-box tests;
- benchmark evidence for hot-path changes.

The ordinary edit loop is `scripts/check-local.sh` with the affected packages:
format check, warning-denied native type check, and affected tests only.
Boundary-specific sanitizer, CUDA, soak, benchmark, and release campaigns run
only when that boundary changes. The standard completed-phase gate is:

~~~sh
moon info
moon fmt
moon check --target native --deny-warn
moon test --target native --deny-warn
~~~

Native and kernel phases add sanitizer, leak, correctness, and benchmark gates.

Current implementation progress is recorded in [STATUS.md](STATUS.md). A
completed workstream foundation does not promote its enclosing phase before
the phase outcome and every named gate pass.

## Bounded streaming workstream

The branch-local [bounded streaming plan](STREAMING.md) tracks the separate
host-memory workstream. It uses functional MoonBit plans and transitions,
preallocated transfer ownership, and staged qualification. Worker protocol,
logical prefix ownership, serving lifecycle, and bounded policy selection are
integrated under explicit versioned BF16, I8, FP8, Qwen3, Mistral and
tensor-parallel descriptor capabilities. Physical model
parity and performance remain named gates before production qualification.

## Latest sealed physical gate position — 2026-08-28

The latest sealed portable source snapshot
`1b951694414dbd9fc9d796eb9532580d82e0bb6a101ad289194ec58cd8d12eaa`
passes `moon info`, `moon fmt --check`, the 506-task warning-denied native
check, 2,297/2,297 tests on Linux, and all pinned-executable sanitizer,
allocation, process-ABI, and worker-authority gates. A fresh complete `sm120`
campaign rebuilt the
child, compiled and authenticated the offline 21-operation AOT set, and passed
spawned execution, native readiness, broad bounded BF16 serving,
caller-authorized non-routable Responses, one real-timestamp measurement, and
physical eight-token prefix reuse. Broad serving covered concurrent requests,
active-plus-waiting saturation, cancellation, foreign and malformed rejection,
same-owner recovery, restart, drain, exact network/KV/event balance, and
deterministic closure. Every entry in evidence manifest
`94852d96c82390ffa427d2ce630c89086bec4f64b8395c431ed31d2fccd9a8cc`
was independently verified; GPU process/memory balance returned exactly to its
initial state.

The same snapshot independently passes the 128-cycle CUDA primitive, all eight
BF16 kernel families, paged attention, a 128-cycle capture-required complete
graph, the five-case/160-launch shape matrix, and two byte-identical approved
model runs. The approved model produced `1031,2185,688,2844`, closed resources,
and left no compute process. Phase 7 correctly rejects the heterogeneous
`sm120`/`sm75`, no-peer, no-NCCL node before resource authority. That sealed
source also rejected I8 on `sm120` before context or allocation authority. The
current software policy now admits exact `sm120` for I8 and FP8, so the older
rejection is historical evidence rather than the current target policy; it is
still not positive quantized execution evidence.

The current Phase 2/3 soak policy is digest-pinned v3. Its real-worker smoke
and both 2,300-wave fast and timer diagnostics pass on final29. The earlier
final7 24-hour attempt is infrastructure-interrupted after host loss and has no
terminal result. Its exact-final29 replacement passes with 86,400,012
milliseconds of wrapper time and 86,400,002 milliseconds measured by the
worker, positive required wave counters, 833 balanced accepts/disconnects, and
exact queue/active/KV/pending/child resource balance. Evidence manifest
`a49609d819e143adf1e63d99f0ee97014a58d095baaffd0f9a3c2eb43ff8788f`
rehashes completely; the local archive SHA-256 is
`3a105bb7ef9665c86e8dabec1808ef61da838c3579bcb4929e05948e623d4fb0`.
The two historical v2 24-hour passes remain authoritative only for their
frozen v2 source. This physical qualification does not waive the remaining
public routability/TLS/control approval, full baseline comparison,
OCI/SBOM/provenance/signing, LunaNexa integration, positive I8, or positive
homogeneous tensor-parallel/NCCL gates.

Post-final7 additions are carried by the sealed final30 source, full Linux
suite, and sealed final30 physical campaign; the final29 soak also passes. They
add exact loopback `/healthz` and `/readyz` observation,
one inherited descriptor-5 Unix-stream drain capability for the opaque CLI,
one separate descriptor-6 read-once inference-credential capability with
digest-bound OpenAI policy-v3 activation and deterministic wipe propagation,
context-churn plus actual 8-token/9-token spawned-owner qualification harnesses,
a Responses-native 81-capture offline benchmark admission boundary, and a
bounded Mistral semantic-plan foundation. Final30 additionally pins the live
trial-driver identity during independent comparison replay, adds typed
fused-kernel candidate observations, exposes a non-authoritative structured
operator doctor, validates rank-failure semantics, binds a fixture-only Mistral
correctness corpus, and carries exact first-party license evidence through OCI
assembly. Their focused native checks, tests,
sanitizer/allocation gates, aggregate structural boundaries, and bounded
sealed-final30 physical campaign pass. They do not prove public routing, TLS,
external control authorization or generation fencing, full physical
context/leak behavior,
live comparative performance, or Mistral execution. Historical physical
evidence below remains scoped to its exact sealed source.

## Current software-validated source state — 2026-08-29

The exact current tree passes the complete 2,448/2,448 native matrix, aggregate
dependency/debt boundary, and focused authority/allocation/performance gates
locally. A non-overwriting portable handoff is sealed only after these checks;
its digest belongs to the external handoff record so sealing does not create a
self-referential source hash. This is source and software-validation state
only: it is not part of sealed final30 physical evidence, does not change the
exact-source scope of the completed final29 soak, and carries no new NVIDIA,
sanitizer, benchmark, or release evidence until a named post-soak
current-source campaign verifies.

After final30, local-only debt hardening centralizes FP8/I8 numeric target
matching in the kernel catalog, makes the external-protocol instance join
private and fail-closed, splits authenticated bootstrap encoding from untrusted
decode validation, permanently enables MoonBit warning 73 in the aggregate
boundary, removes unused public descriptor and qualification-owner accessors,
and aligns authority validators with the private credential-policy join and
opaque snapshot-pinned worker-executable graph. The next source-only pass
removes the unauthenticated raw-Bytes device-loader surface, drops dead
scheduler retirement accounting with hostile generation-exhaustion coverage,
eliminates optimizer-dependent Result/defer allocations from request/output
progress and blocking inherited-frame I/O, and retains event/graph/graph-exec
cleanup authority across combined native create/destroy failures. The strict
warning-denied native check, 2,364/2,364 tests, sanitizer/allocation gates, and the
complete dependency/debt sweep pass. Worker-service shutdown and close now use
explicit lifecycle catches rather than optimizer-dependent deferred cleanup,
and the root-bound process supervisor no longer exposes its recovery startup
contract outside the owning package. The fused candidates now also have a
source-only physical campaign composition: deterministic double compilation,
typed artifact binding, public-device execution, scalar BF16 comparison,
memcheck plus racecheck observation, exact evidence admission, and a
non-circular three-level immutable seal. Its manifest-covered outcome remains
qualification-only, and admission binds the asserted inner manifest digests
without claiming filesystem-verification authority. The hostile fake-tool
campaign and fixed-shape benchmark-qualification evidence gates pass locally.
The generic LunaTile layer now validates and deterministically lowers affine
copy, aligned asynchronous copy, barriers, pipeline stages, MMA, reductions,
and elementwise operations into an inert identity-bound CUDA translation unit;
its fixed one-thread implementation is a semantic reference, not the later
parallel/tensor-core optimizer. A separate deterministic, authority-free
parallel SIMT specializer now binds exact block/warp/lane mapping, a uniform
2--4 stage pipeline, lifetime-planned shared storage, tensor-core eligibility,
and conservative cross-instruction output-disjointness to that serial oracle.
An offline exporter and isolated campaign runner now bind that exact SIMT
candidate and serial oracle, compile each twice, compare independent numerics,
run memcheck/racecheck, verify resource closure, and seal the result without
granting manifest or promotion authority. The source and fake-tool transaction
gates pass locally, but this exact campaign has not run on NVIDIA hardware and
therefore supplies no physical or performance evidence.

A separate typed tensor-core boundary accepts only exact `sm120`, BF16
row-major `m16n16k16` WMMA with F32 accumulation/output, 32-byte global
alignment, one warp per tile, an identity epilogue, and the authenticated
LunaTile/serial/SIMT identities. Its qualification-only probe and campaign
compose deterministic CUBIN pairs, independent ordered-F32 and serial-CUDA
oracles, dual `cuobjdump`/`nvdisasm` instruction observation, bounded resources,
memcheck/racecheck/initcheck, and a non-circular `FILES`/`RESULT`/outer seal.
The evidence admission package positionally binds the exact 67-field result
and critical sealed files, and its evidence-aware lowering join retains both
fallback identities. The evidence-free `RequireExternallyQualifiedTensorCore`
path still rejects, and both admitted evidence and its opaque qualified wrapper
remain `manifest_bindable=false` with promotion authority absent. These source,
hostile fake-tool, and admission gates pass locally; no tensor-core campaign
has run on NVIDIA hardware, no instruction or numeric result is claimed, and
no production consumer exists. The device-step owner also derives graph
hit/miss counts from the actual captured/eager executor mode and retains the
authenticated selected shape in allocation-free scalar telemetry. A fixed
80-byte checksummed private
child sidecar now carries those scalars through the rooted worker process;
completion publication waits for the matching report, worker replacement
preserves saturating totals, and native/OpenAI instance metrics consume only
monotonic deltas. Tensor-parallel transport explicitly reports this telemetry
unavailable rather than fabricating a miss. The opaque CLI also has a bounded
`bench` command that performs one digest-pinned live deployment admission, one
fixed token-0/two-output greedy request, real monotonic measurement, and full
owner drain. Its evidence grants neither comparison nor promotion authority.
Fused qualification and promotion records remain inert evidence, with no public
arbitrary-key verifier constructor. Separately, production-fast-path V2 APIs
now admit and execute residual-plus-RMSNorm and positioned-QKV/RoPE/KV-write
followed by read-only paged attention. They startup-authenticate the exact
catalog, artifacts, plan adjacency, fallback identities, raw-pointer ABIs, and
live device, then replace two standalone launches with one or three standalone
launches with two. Legacy Llama and generic numeric-BF16/Mistral worker APIs
both accept these optional spans; absence preserves standalone execution.
Production token steps contain no canary, cryptography, filesystem validation,
diagnostic transfer, or qualification scan. Residual V2 preserves the
authenticated CUDA graph policy; only its qualification ABI remains eager-only.
Descriptor-pinned optional artifacts now reach deployed children through the
exact runtime descriptor/bootstrap for Llama and numeric BF16. The QKV
aggregate binds only after child-side live-device identity authentication;
absence preserves standalone execution. Current-source physical CUDA
correctness, sanitizer, performance, and reviewer qualification remain open.
The lower production-V2 and literal spawned `device-greedy`/
`device-greedy-fused-v2` campaign harnesses now reach those exact boundaries
and compare fixed device results with an independent host full-logits referee;
only their software/static composition is validated in this worktree.

Kernel approval verification is now reachable only from startup-owned bounded
FD-7 key authority. The parent consumes and wipes that deployment key, then
sends each child a fresh one-shot attestation bound to the exact manifest,
approved source, pinned executable launch, generation, and ordinal. The child
never receives the deployment key, and stale/substituted/replayed records fail
closed before `Ready`. Missing external approval preserves standalone
inference.

The next exact-source hardware campaign was prepared behind the Phase 2/3
soak, which has now passed. Its runner includes the qualification-only
context-churn and actual 8-token/9-token modes, binds both to the freshly built
authenticated launch/worker, verifies canonical evidence digests, and retains final
GPU/process balance as the terminal gate. Immutable directory manifest/sealing
mechanics are shared without moving campaign schema or promotion decisions out
of their owners. OCI verification rejects every rootfs/metadata regular file
with a hard-link alias, including recovery-time verification views. The
external-process comparison handoff seals 326 canonical invocation records and
326 matching cleanup receipts, allowing offline audit of exact argument hashes,
timeout/grace, and credential-FD scope. Local native, hostile evidence, OCI
recovery, current-source, and fake 81-trial gates pass; physical context runs,
live vLLM/SGLang comparison, approved OCI inputs, and promotion remain open.

The following snapshots remain historical evidence for their exact sources.

The latest exact portable source snapshot
`847c8493a4d43faf8517969be8780eace00380b69fa9c6b7894d3fd9e45b36ac`
passes 2,038/2,038 Linux native tests and a fresh `sm120` campaign that rebuilds
the child and passes authenticated native serving, non-routable OpenAI
Responses loopback qualification, one real-timestamp LunaFlux measurement, and
physical eight-token prefix reuse. The final GPU memory/process balance is
exact. The first campaign exposed and preserved a same-address restart failure
from server-side `TIME_WAIT`; the centralized `SO_REUSEADDR` fix passed
independent live-bind review and the full rerun. This supersedes the older
prefix/benchmark/child-rebuild gaps below without converting qualification-only
evidence into production promotion.

The exact portable source snapshot
`99887e5f4687889fd30f3927508a4adc49ff0b1f052117f87bca9b8c069d9e83`
passes 2,010/2,010 native tests on Linux and the supported `sm120` BF16
primitive, kernel-family, paged-attention, complete-graph, shape-matrix,
approved-model, spawned-parent, native-listener, concurrent-framed, finite
balance, and sanitizer/ABI campaigns. No GPU process or memory imbalance
remains. This establishes the executable BF16 path on the available target; it
does not waive any phase gate.

The remaining hardware/evidence gates are explicit:

- positive I8 and FP8 execution remains unproven; current software admission is
  exact `sm89`, `sm90`, or `sm120`, and physical correctness/performance must be
  established on the selected target;
- positive tensor parallel/NCCL requires a homogeneous peer-capable node with
  NCCL (the available `sm120`/`sm75` pair fails closed before authority);
- physical prefix reuse now passes for the pinned eight-token reusable prefix
  and independently fixed two-request referee output;
- one physical LunaFlux measurement now passes, and a Responses-observing
  81-capture replay boundary exists locally, but benchmark promotion still
  requires the actual pinned, counterbalanced LunaFlux/vLLM/SGLang nine-profile
  trial matrix plus independent correctness authority;
- the existing two 24-hour Phase 2/3 v2 soak passes remain authoritative for
  their frozen source; current v3 smoke plus fast/timer evidence and the
  exact-final29 24-hour run pass for their exact source.

Detailed command outcomes, artifact identities, and evidence hashes are in
[PHYSICAL_VALIDATION_2026-08-27.md](PHYSICAL_VALIDATION_2026-08-27.md).

## Phase 0 — Contract and repository bootstrap

### Outcome

A buildable documentation-first repository with frozen product boundaries and
the first public lifecycle vocabulary.

### Deliverables

- moon.mod using the native target;
- product contract, architecture, debt policy, and benchmark contract;
- public contracts package;
- EngineState and RequestState lifecycle types;
- checked request transition function;
- lunaflux bootstrap executable;
- generated package interfaces;
- dependency-boundary script design;
- Apache-2.0 licensing and repository metadata.

### Gate

- standard phase gate passes;
- generated interfaces contain only intended public lifecycle types;
- LunaNexa worktree remains unmodified by LunaFlux creation;
- no source dependency points at a sibling MoonSuite repository;
- repository status is clean after the bootstrap commit.

## Phase 1 — Reference model correctness

### Outcome

One Llama-style BF16 model produces validated offline logits and greedy tokens
without Python in the LunaFlux runtime.

### Workstream 1: configuration

Create focused packages and types:

- config/schema: versioned document and unknown-field rejection;
- config/service: process and diagnostic settings;
- config/model: model path, expected digest, architecture allowlist;
- config/device: explicit visible-device set and memory ceiling;
- config/resolved: immutable startup explanation.

Implement diagnostics and planning before serve. The original host/model-root
forms report driver, device, ABI, model-file, memory, shape, and capability
problems and now remain explicitly named `legacy-*` compatibility tools. The
canonical doctor and plan instead authenticate one complete digest-pinned
deployment and remain inert.

### Workstream 2: tokenizer

- parse bounded tokenizer.json;
- implement the selected raw ByteLevel-BPE contract and the exact closed
  SentencePiece-derived BPE pipeline used by the pinned reference tokenizer;
- define TokenizerSpec and TokenizerDigest;
- test Unicode, invalid UTF-8 input policy, special tokens, truncation, and
  deterministic encode/decode;
- compare token IDs against a pinned reference corpus.

General SentencePiece model/protobuf loading and arbitrary normalizer,
template, or decoder plugins are excluded. The exact pinned
`tokenizer.json` profile is admitted without widening that boundary.

### Workstream 3: model metadata and safetensors

- bounded JSON parsing for the selected model configuration;
- safetensors header parsing with overflow-safe byte ranges;
- duplicate tensor, overlap, dtype, shape, and total-size rejection;
- validated LlamaModelSpec with layer/head/hidden-size invariants;
- content and plan digests;
- streaming or mapped weight materialization to final buffers;
- no executable metadata behavior.

### Workstream 4: native device ABI

Implement only the ABI required by the reference runner:

- driver initialization and device enumeration;
- context, stream, event, device allocation, and host-transfer operations;
- module/function loading for AOT kernels;
- cuBLASLt handle and GEMM invocation;
- bounded error translation;
- explicit close order and failure-safe partial construction.

Add C stub unit probes, AddressSanitizer builds, repeated open/close tests, and
forced-failure cleanup tests.

### Workstream 5: immutable model plan

The Llama builder emits ordered operations for:

- embedding;
- RMSNorm;
- QKV projection;
- RoPE;
- causal attention;
- output projection;
- residual paths;
- gated MLP;
- final norm and language-model head.

The plan records workspace bounds and required kernel capabilities. Execution
contains no architecture-name switch.

### Workstream 6: reference executor

- single request;
- batch size one;
- full prefill recomputation;
- greedy generation;
- synchronous worker acceptable only in this phase;
- simple correctness kernels or vendor kernels selected explicitly.

The current device-preparation contract is intentionally stateless. Exact
profiles may represent full prefill or full-sequence recomputation, but not a
one-token decode backed by implied cache state. A constructible decode profile
requires the Phase 3 semantic plan, KV arena, cache-position, and page/block-
table inputs; maximum-envelope activation storage is not a substitute.

### Gate

- tokenizer corpus matches the pinned reference;
- malformed model files fail before device allocation;
- tensor metadata and materialization checksums match;
- logits match the recorded tolerance across prompt fixtures;
- greedy token sequences match;
- repeated load/run/unload has balanced host and device resources;
- no performance claim is made.

Checked streaming file-to-device loading, exact static device and
activation/workspace plans, stateless execution profiles, AOT family/entry
launch contracts, digest-verified artifact admission, and the private
module/function launch seam are implemented foundations. Offline reference
admission is now synchronous and capability-relative: three immutable snapshots
close before publication under one caller-owned approved root. A pinned
synthetic ByteLevel-BPE compatibility artifact now composes supported text
encoding with the selected model's independently recorded logits and tokens.
The byte-BPE implementation also has one authoritative reusable
`LunaTokenizerWorker`: fixed startup storage, opaque epoch-bound work,
resumable CSR merge lookup, bounded progress/copy calls, frozen-reference
equivalence, and a positive-controlled release-C allocation gate. The separate
fixed-lane request-preparation proof composes canonical UTF-8, that worker,
bounded token copying, and reusable incremental-output setup; it does not
broaden the worker-only allocation claim or cover framed ingress.
`lunaflux legacy-config-plan` authenticates bounded configuration and explains the derived
semantic plan, required capabilities, and KV capacity, while deliberately
leaving weight identity and kernel-manifest resolution open. These foundations
do not promote
Phase 1 without physical-CUDA numerical, ownership, sanitizer/leak, soak,
benchmark, and resource-balance evidence. See
[STATUS.md](STATUS.md) for the current boundary.

## Phase 2 — Online streaming service

### Outcome

The reference engine becomes a bounded, cancellable service with startup-fixed
request lanes, deterministic global event order, a native protocol, and a
compatibility adapter.

### Deliverables

- async native HTTP or framed protocol server;
- canonical GenerateRequest and StreamEvent contracts;
- bounded request/body/text/token/stop limits;
- API authentication hook without tenant semantics;
- tokenizer task pool;
- one scheduler-owner task and bounded mailbox;
- one isolated device worker process;
- typed worker protocol with sequence and generation;
- OpenAI chat-completions and responses adapters;
- server-sent event streaming;
- graceful drain and readiness;
- payload-safe logs and bounded metric labels.

### Failure scenarios

- client disconnect during tokenization, prefill, decode, and blocked output;
- worker exit and invalid completion sequence;
- malformed compatibility request;
- deadline before admission and during execution;
- output consumer slower than configured buffer;
- shutdown while a request is active.

### Gate

- one request streams the same greedy tokens as offline execution;
- cancellation publishes no post-cancel token;
- worker restart makes readiness truthful and leaks no request state;
- slow output is bounded and eventually cancelled or backpressured;
- public errors contain no paths, pointers, vendor traces, or raw payloads;
- 24-hour low-rate soak has balanced request and device resources.

## Phase 3 — Continuous batching and paged KV

### Outcome

Multiple requests share the GPU through continuous batching and a deterministic
fixed-page KV arena.

### Workstream 1: physical page allocator

- PageId(index, generation);
- preallocated metadata arrays;
- intrusive free queue;
- per-request block-table arena;
- active and cached reference counts;
- deterministic allocation/release;
- invariant checker enabled in debug/test builds;
- allocation snapshot fixtures.

The host page allocator, fixed-capacity per-request block-table arena, inline
page/table identity storage, and randomized ownership fixtures are now
implemented. The scheduler constructs both owners from resolved startup
capacity and uses authenticated checkpoints for exact cross-owner allocation
and rollback during plan construction. Completion-driven terminal release and
device KV storage remain part of this phase gate.

### Workstream 2: device KV arena

- preallocate KV storage during startup;
- one page size selected in ResolvedPlan;
- layer and head layout encoded in ModelPlan;
- block-table upload using reusable staging memory;
- paged KV write and attention kernels;
- no device allocation during a token step.

The versioned paged semantic graph, fixed host page/table ownership, canonical
split K/V device layout, reusable counts/position/page-table staging, exact
all-AOT catalog and artifact admission, full physical launch blueprint, and
owner-mediated ordered graph executor are implemented. The executor
preallocates persistent KV and activation/workspace storage and reuses loaded
functions and argument lists, enqueues exact operations, and waits once on a
reusable completion event. Its owner-mediated completion path now binds the
exact retained BF16 vocabulary-row geometry, reuses fixed readback/sampling
scratch, selects by canonical request seed/output index, and appends an exact
completion lease that the aggregate owner publishes only after executor
finish. A positive-controlled native release gate now covers a production-
reachable schema-v2 four-row batch identity and exact warm mixed/full-batch
ordinary-prefill/final-prefill/decode lifecycle through public execute,
conditional fixed readback, canonical completion authentication, and plan
retirement/reset. The test-only fake device emits nonuniform, row-dependent
BF16 logits; an independent scalar oracle covers greedy and stochastic
temperature/top-k/top-p selection across varied row token counts, page CSR,
and completion slots. Phase promotion still requires a production paged-kernel
bundle and physical-CUDA numerical correctness, sanitizer, leak, soak, and
benchmark evidence for logits and sampled tokens.

### Workstream 3: scheduler

- unified token budget;
- decode-first scheduling;
- bounded waiting-time aging;
- chunked prefill;
- explicit emergency decode-page reserve;
- recompute-only preemption;
- exact capacity rejection before request activation;
- double-buffered SchedulePlan descriptors.

Startup capacity resolution and the bounded single-owner request registry are
implemented foundations. Resolution checks the worker, model, page,
block-table, row, and generated-token publication envelopes. Admission then
authenticates model/recipe provenance and checks context, physical-page,
request, waiting, and deadline bounds; cancellation and deadline expiry are
generation-safe and transactional with respect to terminal-notice capacity.
Transactional plan construction now activates FIFO requests only after submit,
selects decode resources before prefill, applies bounded eligible-prefill
aging, chunks intermediate/final prefill, preserves the emergency page reserve,
and submits into distinct A/B owners with exact plan/table/page rollback.
Paired completion owners issue exclusive exact-plan leases; ordered full-batch
retirement preflights output, terminal, and release capacity, enforces token
stops/output limits, publishes generated token IDs, and resets exact owners.
Eligible aged-prefill fairness, persistent round-robin decode selection, and
recompute-only preemption are implemented. Preemption releases device-derived
KV state, preserves dense generated-token history, replays the prompt plus all
generated tokens except the retained latest decode input, and returns the
request to FIFO admission without changing its public processed-input count.
Prefix reuse remains Phase 4. Generated-text decoding and exact one-credit
session publication are implemented above the root-bound service. While one
physical plan is executing, the service may build and retain exact scheduler
plan N+1 in the other A/B owner; physical child exchange remains serialized.
The trusted request-receipt prerequisite is implemented:
one fixed-capacity incremental canonical reader authenticates declared length
before payload acceptance, while request admission samples monotonic time before
the first copied byte and binds a single immutable request/absolute deadline.
The live fixed-lane preparation boundary consumes that receipt into a revocable
`LunaPreparedRequest` shell around one exact-generation claim carrying model,
tokenizer-digest, inference-envelope, token, deadline, and incremental-output
authority. A central FIFO quantum owner advances retained UTF-8 byte BPE,
bounded token copying, and incremental-output setup; each preallocated lane
remains leased through scheduler retirement and explicit claim release. The
persistent online instance consumes only that claim and owns no tokenizer or
pool state. The synchronous object-form compatibility facade remains. The live
fixed-lane pool captures trusted receipt time before byte one, drives canonical
raw-frame scanning, and imports the validated scalar/byte view into its token,
semantic, and output owners under the same work ceiling without constructing a
`GenerateRequest`. It deliberately adds no socket, async task, or listener;
the one-shot endpoint, reusable native max-one and pipeline Servers, and the
serialized OpenAI HTTP Server now own network dispatch and exact partial-write
response confirmation above it. TLS and concurrent-client arbitration remain
outside this boundary.
The authenticated OpenAI route performs the equivalent reduction directly:
one receipt remains bound from byte-one capture through JSON parsing, template
expansion, semantic validation, and admission into fixed typed storage. It
constructs no JSON tree, prompt `String`/`Bytes`, canonical request frame,
`GenerateRequest`, or `TextInput`.

### Workstream 4: worker overlap

- scheduler constructs plan N+1 during plan N execution;
- events distinguish submission, execution, and completion;
- steady-state host buffers and rings are preallocated;
- stale plan and completion generations are rejected.

The immutable fixture vocabulary, reusable fixed-capacity plan/completion
buffers, authenticated lifecycle epochs, scalar row drafts, provenance-bound
capability recipes, whole-build checkpoints, final-prefill output semantics,
and stale-generation checks are implemented. Scheduler-owned distinct A/B
buffer pairing, transactional build/submission, exclusive completion leases,
ordered full-batch retirement, generated-token publication, and exact owner
reset are implemented. The scheduler retains its exact resolved worker-protocol
limits for outer-owner binding. Canonical fixed-capacity plan/completion wire
frames preserve full page generations, capability order, sampling replay identity,
and typed outcomes across a flat little-endian boundary; untrusted receive is
transactional, semantically validated, and epoch-authenticated. Received
completion frames are additionally matched to the retained exact plan before
the service populates its paired completion owner; retryable full-batch
scheduler retirement is implemented and returns every authenticated owner to
its reusable state. The isolated side can write exact canonical
completion frames directly from authenticated received-plan rows without a
scheduler heap-owner capability. The device descriptor stage consumes the
validated received plan frame directly as well, and its post-execution path
authenticates that exact frame epoch before scalar sampling and direct
completion-frame publication. Plan construction fills one scalar row draft
and its retained token/page/capability tables in a single pass. Startup-sized
open-addressed identity tables provide `O(1)` duplicate and
completion-slot lookup; the isolated worker performs one validation scan,
stages the retained frame directly, and publishes completion against that same
epoch. Independent
positive-controlled native release gates now cover the scheduler token step,
the public device-step staging/fixed-H2D path, and a production-reachable
schema-v2 four-row public device-worker execute lifecycle across ordinary and
final prefill, decode, and greedy/stochastic selection through conditional
logits readback, completion publication, validation, and plan retirement/reset
after warm-up. The mixed/full-batch proof uses nonuniform row-dependent fake
BF16 logits and an independent scalar stochastic oracle; it does not establish
physical-CUDA correctness or promotion. A
private POSIX primitive now supplies exact-path shell-free spawn, an inherited
socketpair, bounded fixed-buffer I/O, monotonic timeouts, and deterministic
reap. The legacy protocol supervisor proves monotonic A/B submission,
oldest-first receive, retained response epochs, and fail-stop malformed-session
handling through three-plan A/B/A echo transport. The root-bound production
facade instead grants one outstanding credit to the serialized device child.
An exact checksummed
Configure/BootstrapSource/ParentApprovalAttestation/Ready handshake binds model identity, an
admitted-bootstrap SHA-256 derived from graph/artifact evidence, a
bootstrap-source SHA-256 derived from canonical `EncodedBootstrapSource`
bytes, exact process-visible device ordinal, generation, predecessor, and
worker/inference limits before protocol readiness;
the child canonically decodes and verifies the source plus a fresh one-shot
parent attestation before Ready; incompatible
children fail closed and startup double failures retain cleanup
authority. The full-graph blueprint and artifact bundle derive the
admitted-bootstrap identity
from a bounded canonical schema including device-step limits and the exact
assignment. The supervisor now
requires close/reap before ordered abandonment of exact outstanding
submissions and derives a non-reusing replacement predecessor only after every
obligation is retired. The real-child gate proves replacement through sequence
5. A thread-confined scheduler/worker service records plans before writes,
retries received frames and
synthesized worker failures under scheduler backpressure, retires scheduler
state before process abandonment, and starts the exact replacement contract
only after its single production obligation clears. The legacy echo gate covers
two-slot transport; the root-bound service gate covers one outstanding plan,
publication pressure, worker death, recovery, replacement, and balanced KV
resources. Before replacement it transactionally fails every surviving active
request and releases its host page/table identities so no page ID can refer to
the replacement's fresh device arena. An independent backend-neutral service
identity retains model identity, generation, bootstrap-source digest, worker
limits, and inference limits. The legacy single-worker constructor binding
supplies only its private bootstrap digest and ordinal; tensor-parallel rank
bootstrap and topology remain in the private group transport. Construction and
replacement additionally authenticate the scheduler's exact predecessor. The production constructor now
accepts an immutable scheduler blueprint plus ordinary approved roots and
constructs both mutable owners without exposing aliases; the alias-taking
constructor is retained only for deterministic fixtures. Deterministic worker
buffers, child ownership, descriptor-pinned executable admission, and Configure/source/expected-
Ready frames are allocated before root acquisition. Native spawn, scalar
handshake I/O, exact Ready comparison, scalar cleanup, and owner publication
allocate no managed objects while rooted child authority is live. A restricted
epoch-authenticated online worker lease now enforces the permanent ownership
family, exact generation/position/publication order, monotonic time, recovery,
and close prerequisites. The alias-free `LunaOnlineInstance` now prepares the
production-owned service once, claims a startup-bounded set of off-reactor
prepared requests under fresh opaque tickets, retains only the expected
tokenizer digest and preparation envelope, preflights exact semantic-event
epoch headroom, and publishes through one deterministic typed event credit.
Busy and draining outcomes do not consume prepared authority; foreign binding
and replay fail before lower admission. Final acknowledgement and exact lower
retirement recycle only the authenticated lane while preserving other live
requests, the worker, lease epoch, scheduler history, and plan predecessor.
Worker/device failure uses cooperative replacement and publishes the recovered
failed terminal before reuse; explicit instance drain owns healthy shutdown.
Replacement is guarded by an explicitly configured lease-owned bounded
backoff. It starts only after old-child cleanup, flight retirement, and device
invalidation, exposes a cached monotonic wake without blocking, caps delay, and
deterministically abandons restart after clock/timestamp/generation drift or
finite attempt exhaustion. Attempt history resets only after a plan commit
whose sequence is later than the current replacement's authenticated startup
predecessor, is the scheduler's exact retired sequence, and crosses its
configured stability deadline.
Steady allocation-free token stepping, physical stop-token suppression, and
natural Maximum/StopToken Usage+Completed-v2 publication are now present.
Incremental string-stop matching now performs an exact cancellation cut (or
authenticates final-token natural precedence), withholds stop/post-stop bytes,
and publishes Usage+Completed(StopSequence). Caller cancellation now defers
behind pinned credit and publishes Usage+Completed(Cancelled); automatic
credit-free deadline enforcement publishes Usage+Failed(deadline_exceeded)
after natural-terminal preflight. Decoder/output rejection and worker/device
loss now terminalize through exact authenticated drains to Usage plus fixed
payload-safe Failed(output_invalid) or Failed(worker_unavailable). Normal
progress only latches that state; explicit off-reactor terminalization owns
recovery/reap and preserves the first failure cause across retry. The
same-child two-request gate authenticates monotonic plan history and request
position reset without respawning. A canonical framed adapter now borrows only
the credit's semantic view and owns no ACK authority. Native one-shot,
reusable max-one, fixed-lane pipeline, and serialized OpenAI HTTP servers
compose bounded ingress and partial writes around that boundary. The production
owner now selects the existing native/OpenAI fixed-capacity connection pool
when authenticated preparation capacity is greater than one, retaining the
singleton only for exact capacity one. TLS, keep-alive HTTP, and a host
off-reactor executor remain future work.

The `engine/device_worker` aggregate now implements the bounded readiness-owner
foundation anticipated by this workstream. It admits independently expected
model metadata and startup plus a precomputed immutable weight-file inspection
and one opaque aggregate paged execution admission. Exact model, weight-layout,
bootstrap, and startup evidence is checked before resources open. Preparation
then verifies the received startup contract, opens the assigned visible device,
checks its capability, completely re-admits the retained inspected file without
a duplicate inspection, and prepares the complete paged executor from the
aggregate admission. It publishes the contract only while context, weights,
and executor remain live, with retryable dependency-ordered cleanup for
compound failures. The spawned device child now imports fixed roots, performs
real source reconstruction before `Ready`, and forwards each bounded plan
through this owner before publishing its exact completion. Clean idle waits are
unbounded until the first prefix byte; partial frames use bounded deadlines. A
positive-controlled release generated-C gate proves repeated serialized-loop
success does not call a MoonBit allocator. One approved spawned-child physical
prefill/decode exchange now passes; concurrent physical exchanges, broader
shapes, and sanitizer/leak promotion remain open. Host plan
N+1 construction already overlaps the serialized physical flight N.

The source-reconstruction workstream now also owns an opaque approved-
filesystem foundation. A neutral internal capability representation supports
both descriptor-relative traversal and the fixed-FD spawn lease, while
the public facade exposes only pinned root/file owners, bounded positional
reads, opaque same-handle stamps, deterministic lifecycle, and a bounded
startup-only immutable snapshot. Snapshot creation holds one operation lease
across before/after size+mtime+ctime stamps, exact positional reads, and a
trailing-growth probe; it has one accepted payload-allocation site and publishes
nothing after detected truncation or mutation. Canonical path validation is
duplicated at the MoonBit and native boundaries; atomic leases prevent
`openat`, `pread`, or `fstat` from racing descriptor close/reuse.
Fixed-FD inheritance now adds reusable pinned model/kernel leases, sanitized
rooted spawn, and immediate child-side import. The worker supervisor retains
the same pair across every replacement and closes it only when the instance is
retired; initial admission also binds each canonical encoded root label to the
exact caller-owned capability before pair activation, without restart-time
ambient revalidation. Remaining work is production transport and deployment
evidence beyond this root-ownership boundary. Weight and
kernel-artifact loaders use this authority; weight inspection retains its
validated locator privately, completely re-admits before allocation, and
transfers from reusable fixed host storage. The bounded
`engine/execution_manifest_file` aggregate now digest-pins one canonical
paged-Llama v2 manifest and derives catalog v3, static/memory plans, full-graph
contracts, artifact loading, and the inert blueprint from typed evidence.
Remaining promotion work is transport/deployment evidence for the inherited
authority path, not another ambient model/kernel root migration.
Readiness additionally requires successful terminal source-file close. A close
failure consumes the file authority and closes any ready allocation; if that
allocation close also fails, retryable cleanup authority is retained at the
payload-safe `SourceClose` stage.

### Gate

- deterministic scheduler fixtures cover mixed prefill/decode queues;
- randomized allocator model checking preserves ownership invariants;
- cancellation and failure at every plan boundary balance all pages;
- no steady-state device allocation;
- hot-path allocation instrumentation shows no general heap allocation after
  warm-up;
- saturation produces bounded queueing or typed rejection, never device OOM;
- supported profiles are within ten percent of the best pinned baseline or
  have a documented, measured blocker before phase promotion.

## Phase 4 — Radix prefix reuse

### Outcome

Token-prefix discovery reuses physical page runs without coupling logical cache structure to GPU objects.

### Deliverables

- compressed token radix tree;
- longest-prefix lookup;
- page-aligned shared runs;
- request-private partial tail;
- explicit cache security scope;
- model/tokenizer/layout salted cache identity;
- active locks/reference counts;
- deterministic LRU plus optional priority;
- zero-reference eviction only;
- prefix hit/miss/eviction metrics;
- cache-disable request policy;
- prefix-aware waiting-order policy bounded by fairness.

A compressed, startup-preallocated token radix implements exact salted identity,
longest-full-page lookup, active references, request-private tails, deterministic
priority/LRU ordering, and zero-reference eviction. Scheduler/KV integration is
transactional and preserves absolute aged fairness. Bounded telemetry and host
gates pass. Full salted roots and `(root,parent,first-token)` children use
balanced indexes; all arena free paths are `O(1)`, zero-reference victim lookup
is `O(1)` with `O(log E)` updates, exact duplicate-page rejection is bounded
`O(P log P)`, and compaction visits only the changed ancestry. Waiting requests
retain logical evidence only; activation exact-revalidates and acquires page and
entry references transactionally, with rollback at every build checkpoint.
Physical-CUDA cached-versus-uncached correctness now passes for the pinned
eight-token reusable prefix with independently fixed two-request outputs.
Prefix-rich parity and prefix-cold regression against the pinned baselines
remain open.

### Test matrix

- exact, partial, and absent prefix;
- page-boundary and one-token-short matches;
- duplicate prompts;
- cancellation while sharing pages;
- eviction pressure with active descendants;
- tokenizer/model/layout/security-scope mismatch;
- repeated insert/remove compaction;
- cache-disabled requests;
- deterministic replay of cache/scheduler snapshots.

### Gate

- cached and uncached logits/tokens match;
- no cross-scope or cross-plan reuse occurs;
- active pages are never evicted;
- prefix-rich benchmark reaches parity or better than the best pinned baseline;
- prefix-cold regression remains within the declared budget;
- the radix package imports no device implementation.

## Phase 5 — Graph capture and LunaTile kernel program

### Outcome

Measured hotspots move to an AOT tile-oriented catalog while preserving engine architecture.

### Workstream 1: capability manifest

Define a versioned manifest entry whose digest is bound by an external
deployment signature-approval receipt, containing:

- operation and semantic version;
- GPU architecture range;
- dtype and accumulator;
- tensor/KV layout;
- shape class and alignment;
- workspace requirement;
- graph-capture safety;
- artifact digest;
- specialization-record digest and every baked semantic input identity;
- compiler/toolchain identity;
- declared floating-point equality or tolerance policy;
- correctness and benchmark evidence reference.

### Workstream 2: LunaTile IR

Implement the smallest IR needed for selected kernels:

- typed TensorView and address space;
- affine tile loops;
- copy and asynchronous copy;
- barriers and pipeline stages;
- MMA;
- reductions and elementwise expressions;
- compile-time layout/shape constraints.

Add validation, canonicalization, memory planning, vectorization, pipelining, CUDA lowering, and deterministic serialization.

### Workstream 3: typed offline specialization

Implement a deterministic specializer that consumes only authenticated `ModelPlan`,
static device-plan, execution-profile, and catalog capabilities. It may bake checked
dimensions/offsets, layout, RoPE, codebooks, expert strides, and entry points. It must:

- keep those values typed rather than using option maps or source concatenation;
- reject unsupported shape, head, dtype, layout, quantization, alignment, and target before artifact emission;
- serialize a content-addressed record binding inputs, constants, compiler identity, flags, and numerical contract;
- run only during offline artifact production, never because of a live request;
- expose no generated source, compiler, model, or device authority through a production public API.

The first evidence harness compares each generated family with an independent
scalar referee on adversarial blocks, real tensor rows, and boundary shapes;
proves dispatch with a canary; compares logits/tokens; and ends with a mixed-workload
benchmark. Strict and reassociating builds use separately named numerical standards.

### Workstream 4: initial custom kernels

Prioritize by profiler evidence:

1. paged decode attention;
2. paged prefill attention;
3. fused QKV plus RoPE and KV write;
4. RMSNorm/residual fusion;
5. sampling reductions.

GEMM remains cuBLASLt/CUTLASS until evidence justifies replacement.

### Workstream 5: CUDA graphs

- capture declared shape classes during warm-up;
- maintain explicit eager fallback capability only when validated;
- no graph construction on live request data;
- report graph hit/miss and selected shape class;

The implementation target is now named an execution-graph bucket. CUDA Graph
is the first backend. Prepare distinct `1`, `2-8`, `9-16`, and `17-32` row
buckets where the model envelope permits, so decode does not launch a maximum-
capacity graph whose kernels immediately return. Bucket selection is resolved
from prevalidated scalar counts and indexes prebuilt graph executors without
allocation.
- bound graph memory in the startup plan.

### Gate

- IR serialization and lowering are deterministic;
- invalid layouts fail before compilation;
- specialization records reproduce byte-identically and change with any bound input;
- generated artifacts reject mismatched model, layout, target, compiler policy, or launch contract;
- dispatch canaries prove tests exercised the specialized entry point, not fallback;
- each custom kernel passes differential, boundary, sanitizer, and race tests;
- each replacement wins its declared microbenchmark shape set;
- end-to-end mixed workload does not regress;
- production mode executes no compiler or JIT path.

The software foundation now implements semantically neutral LunaTile
validation/serialization and a generic deterministic serial-reference CUDA
translation for copy/async-copy, barrier/pipeline, MMA, reduction, and
elementwise operations with alignment-aware vector copies and canonical
program/planning/source identities; deterministic family reference CUDA lowering for embedding,
RMSNorm, positioned RoPE, residual add, QKV/output/LM-head projection, gated
MLP, and mixed prefill/decode paged attention; a closed offline CUBIN builder;
strict post-compile pointwise binding; and deterministic full-graph manifest
production without runtime geometry synthesis. Sampling placement is an
authenticated startup choice: greedy requests may use the AOT device reducer
embedded with the admitted LM-head artifacts and return one eight-byte result
per producing row, while temperature/top-k/top-p use the preallocated host
readback path. No fake sampling graph operation or token-step placement decision
is introduced. The BF16/I8 runtime enqueues an
authenticated graph in exact order and waits on one completion event. The BF16
production worker selects capture with eager fallback only from authenticated
capture-safe graph metadata and its validated fallback disposition; otherwise
it remains ordered eager. A
startup-only CUDA graph lifecycle seam now admits optional driver symbols,
captures only that fixed prebuilt sequence, reuses one exact graph exec per
owner/stream, and exposes no node-update API. Required capture fails closed;
eager fallback exists only under its explicit startup policy. Fake-driver and
sanitizer coverage proves lifecycle and retry behavior. A capture-required
sm120 add-one probe also passed 128 exact launch/poll/close cycles without
selecting eager fallback. Runtime graph telemetry now derives captured hits or
ordered-eager misses from that private executor mode, saturates without wrap,
and reports the authenticated selected shape through opaque allocation-free
snapshots after successful completion. A typed BF16 release producer now constructs actual-
CUBIN catalog-v3 families, complete launch contracts, strict family bindings,
the canonical schema-v2 manifest, and an exact compiler/JIT-free kernel-root
plan. Its two-layer fixture proves 21 operations reuse nine physical modules
without catalog ambiguity. The authenticated model-plan-to-candidate command
and strict offline compilation join are now complete. Graph-memory accounting
now binds authenticated capture metadata to an explicit descriptor-v2 ceiling,
adds the declared upper bound once to startup capacity, and leaves v1
eager-only releases explicitly absent. A bounded offline/startup
profile-priority admission now binds sorted stateless full-context observations
to the immutable model plan, target, workload, and profiler identities and
deterministically selects the operation with the greatest attributed self
time. Its separate paged path consumes an admitted paged launch-contract set,
derives the exact target/profile/device-KV layout, binds mixed prefill/decode
row, context, cache-position, and page-slice geometry, and requires every
observation to name an operation covered by that profile. The page-table trace
digest is explicitly a raw profiler claim, not page ownership or execution
authority. The paged capture owner retains bounded sorted per-operation
counters without adding execution authority. Neither path performs profiling
or authorizes kernel promotion. A separate promotion owner can consume sealed
paired evidence plus external approval but remains inert and non-bindable: the
current generic fixture ABI has no authenticated mapping to a real operation's
exact operands, shape, and numerical contract. Broad production-shape coverage
and performance evidence remain open.

Physical sm120 evidence now covers all eight generated BF16 reference families,
the residual specialization through the ordered executor, one paged-attention
mixed-row fixture, and the capture-required primitive graph lifecycle. A later
complete tiny graph passed 128 required-capture cycles across twelve launches,
with mixed prefill/decode KV writes, fourteen CPU-refereed boundaries, exact
fixture logits/token, closed resources, and empty stderr. A bounded five-case
shape matrix then passed 40 captured graph launches. Finally, the pinned
upstream tiny BF16 model was compiled from its canonical 21-operation plan and
executed physically on sm120: eleven selected logits matched the independent
corpus with maximum absolute error `0.0005459413`, the greedy continuation was
`1031,2185,688,2844`, same-page KV persistence passed, and all resources
closed. This is approved-model numerical evidence; a later r14 campaign also
crosses the spawned worker/service boundary for one pre-listener two-token
request. It is not broad serving, shape, leak/soak, or performance promotion.
The r9 2026-08-28 integrated snapshot repeated that approved-model campaign
from its warning-denied 1,957-test Linux source, passed the release packaging,
assembly, materialization, and deployment-boundary gates, and left no GPU
compute process. Its concurrent OpenAI pool separately passed slow/fast-client
progress, reuse, retirement, and drain. The later r14 campaign materialized the
target-specific descriptor, policy, model/tokenizer, AOT modules, canonical
child, and 256-token KV geometry, then passed physical spawned execution for
the pinned two-token fixture.
A later r17 campaign reused that exact launch and child, bound the production
native loopback listener, and passed the exact
`Accepted,Token(1031),Token(2185),Usage,Completed` stream with one network
accept/disconnect, restored 32-page KV balance, empty stderr, and complete
listener/child cleanup. This is a bounded traffic-readiness fixture, not TLS,
concurrency, public-network, latency, throughput, or release promotion.
An authenticated native campaign harness now pins the checked-in upstream tiny
BF16 model content, tokenizer, canonical single-row plan, launch recipe, and an
independently supplied child-executable digest. It crosses the production
one-argument runtime path, consumes the exact post-child-Ready service owner,
and submits one predeclared two-token framed request without binding a
listener. Its opaque operator-only validator checks Accepted, the pinned
`1031,2185` token continuation, Usage, Completed(TokenLimit), consecutive
one-row/one-token prefill/decode plans, and one-page KV residency across both
steps before disconnecting, draining, closing, and reaping. No service,
scheduler, worker, device, or logits owner escapes. The physical r14 result is
limited to this token-level fixture: exact tokens `1031,2185`, plan sequences
`1,2`, single-page geometry with live-or-retired balanced telemetry, empty
stderr, and complete device/child cleanup. It is not selected-logit,
traffic-readiness, general serving, or performance evidence.
The product-owned offline BF16 candidate exporter now derives a canonical
source/recipe set solely from an authenticated model plan, exact KV layout,
profile, sm120 target, and inert compiler policy. Its no-overwrite filesystem
owner emits the declaration and independent inventory accepted by the existing
offline CUDA builder. A narrow native command authenticates the checked-in tiny
model configuration, closes its approved root, and emits that envelope without
compiler, process, device, runtime, or serving authority. Exact compiler
major/minor/patch identity is a bounded typed input and remains strictly bound
through the recipe and two-build verifier. The approved-model campaign has now
crossed this bridge through offline CUDA compilation and physical numerics;
the exporter itself still makes no readiness claim.
Pointwise, projection, and paged-attention now share the non-circular candidate
-> offline compile -> final-contract binding sequence, and kernel-bundle
admission accepts only the opaque bound forms. Complete optimized kernels and
full comparative benchmarks remain open. The latest final7 campaign adds
bounded broad spawned-worker serving validation; broader shapes and contexts
remain open.
The two block-128 fused families retain their deterministic qualification
sources, scalar/page-boundary referees, double-build binders, hostile-input
gates, sanitizer campaign composition, and inert benchmark/promotion records.
Production V2 removes qualification canaries and supplies narrow runtime
artifacts: residual plus RMSNorm is one prepared launch, while positioned
QKV/RoPE/KV-write plus read-only paged attention is two. Exact startup admission
checks target, plan adjacency, operands, layouts, fallback identities, artifact
bytes, ABI, and live-device identity once. Residual V2 preserves the
authenticated graph policy, while its diagnostic qualification ABI remains
eager-only. Both legacy Llama and generic
numeric-BF16/Mistral worker APIs can prepare and execute the optional spans.
Missing optional artifacts preserve standalone execution. The exact deployed
child descriptor/bootstrap now carries these optional artifacts for Llama and
numeric BF16, and admits the QKV aggregate only after live-device identity is
available. No current-source physical fused CUDA result,
sanitizer/race result, microbenchmark win, mixed-workload comparison, or release
promotion is claimed.
Parallel LunaTile mapping, multi-stage overlap, lifetime-based shared-memory
reuse, the exact BF16 tensor-core source candidate, isolated physical campaign
composition, and sealed evidence admission are source-complete for their
current narrow contracts. Their NVIDIA correctness/SASS/sanitizer/resource
campaigns, paired performance evidence, external review, production manifest
consumer, and promotion gates remain separately open.
External signature approval remains deployment-owned.

## Phase 6 — Sampling, usability, and operational hardening

### Outcome

LunaFlux reaches a vLLM-like operator experience for its smaller capability set.

### Deliverables

- temperature, top-k, top-p, seed, stop tokens, and stop strings;
- deterministic RNG stream ownership per request;
- digest-pinned `lunaflux run DEPLOYMENT`;
- digest-pinned, inert `lunaflux doctor DEPLOYMENT`;
- digest-pinned, inert `lunaflux plan DEPLOYMENT` with authenticated decisions;
- digest-pinned `lunaflux bench DEPLOYMENT`;
- digest-pinned, inert `lunaflux inspect-kernels DEPLOYMENT`;
- concise versioned configuration;
- health, readiness, metrics, and structured diagnostics;
- startup memory/capacity report;
- install, upgrade, drain, rollback, and troubleshooting guides;
- OCI build with exact runtime dependencies and kernel manifest.

A bounded sampler implements temperature/top-k/top-p, deterministic RNG,
finite-logit validation, and fixed scratch. Restrictive top-k is `O(V log K)`;
the unrestricted/top-p path deliberately retains canonical `O(V log V)` order
to preserve IEEE accumulation and draw mapping. Native/OpenAI ingress shares
sampling, seed, and incremental stop semantics; host/fake-seam fixed-seed gates
pass. Strict bounded configuration, inspection records, explanations, and
operator preflights form the foundation and report false readiness when
incomplete. The canonical diagnostic forms consume the same digest-suffixed
deployment admission as run/bench, publish only root-free evidence after every
root closes, and never probe a device, spawn a child, or bind a listener. The
older model-root and separate-root forms are isolated under explicit
`legacy-*` compatibility names and cannot receive malformed canonical
operands. A strict independently digest-pinned runtime descriptor now
composes separately approved model and kernel roots into inert model, weight,
KV, execution-manifest, bootstrap, startup, and device-worker admissions. It
opens no device; host plan/inspection evidence can succeed while run/bench and
readiness remain false. Pinned doctor now renders an admitted, checked memory
and execution/inference-capacity report while preserving that false readiness;
descriptor-v1 scheduler/service capacity is labelled unavailable. Deployment/
OCI now has a declarative compiler/JIT-free Containerfile, a mandatory exact
context/base verifier, a Linux-only build wrapper, hostile static gates, and an
install/upgrade/drain/rollback runbook. A deterministic no-overwrite deployment
bundle assembler additionally separates launch/model/policy roots from the OCI
rootfs, requires exact inventories and digests, and rejects substitution,
ambient files, PTX/JIT material, and partial-transfer output. This macOS host
did not build or prove a Linux/CUDA image; approved base and builder provenance,
final-rootfs/SBOM scan, final digest, physical readiness, and benchmarks remain
open.

`lunaflux validate-release
ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>` now performs the same
recipe-specific descriptor, policy, model, kernel, tokenizer, bootstrap, and
worker-executable semantic join as live startup without opening a device,
spawning a process, binding a listener, or retaining filesystem authority. It
returns canonical root-free digest evidence. The atomic no-overwrite
materializer now supplies the separately typed source-capability/target-label
contract: no-follow staged roots are identity-bound to the final absolute
labels carried by launch/bootstrap evidence, the exact recipe join runs while
the output remains under a private cleanup claim, and only root-free semantic,
tool, and transaction digests survive publication. Hostile target substitution,
semantic failure, and partial-publication injection leave no output. This is
release-host semantic evidence, not OCI construction, reviewer approval, or
physical readiness.

Interrupted publication is now explicitly resumable rather than silently
cleaned or overwritten. The v2 claim binds the canonical output, current uid,
input/tool/materializer digests, and fixed publication order; the prepared
record binds the exact bundle, deployment, and semantic inventories. Recovery
admits only exact prefix states and either releases an otherwise empty CLAIMED
directory with regular-file unlink plus `rmdir`, or completes and verifies the
publication. Ambiguous, linked, symlinked, special-file, non-prefix, and digest-
substituted states are refusal-only.

The 2026-08-28 Linux/CUDA rerun passed these OCI, bundle-assembly, atomic
materialization, and deployment-boundary gates from the final integrated
source. These results validate release wiring and hostile host behavior; they
do not substitute for an approved base image, final-rootfs/SBOM scan,
reproducible image digest, or target-specific deployment approval.

The native one-argument run path is pinned as
`lunaflux run ABSOLUTE_DEPLOYMENT_ROOT#sha256=<64-lowercase-hex>`. It admits a
fixed strict launch envelope, independently loads descriptor, policy, and
tokenizer evidence, verifies the assigned CUDA target and worker executable,
then transfers roots only to the existing worker/service/server composition.
Health and readiness are separate; only the bound ready server is Ready, and
drain/failure clears readiness. The macOS gate remains truthfully unready. The
2026-08-27 NVIDIA campaign passed the CUDA primitive, ABI, sanitizer, full
native check, and full native test subsets recorded in
[PHYSICAL_VALIDATION_2026-08-27.md](PHYSICAL_VALIDATION_2026-08-27.md). It could
not run production paged/I8 inference in that earlier campaign. Later r14 and
r17 campaigns supplied one approved tiny-BF16 deployment, spawned execution,
and bounded native-listener proof; general readiness remains false. The final integrated
`8351d804...41ae8` snapshot additionally passed 1,902/1,902 Linux native tests,
the portable 10,000-request balance proof, all BF16 release and I8 software
gates, both warmed allocation gates against the host compiler's raw-allocation
ABI, and the 128-cycle capture-required twelve-kernel graph with exact logits,
KV writes, sampled token, and closure. Its physical I8 probe failed closed
before resource creation because that sealed source still excluded `sm120`.
Current software admission includes exact `sm120`; the historical rejection
remains valid only for that source and is not positive I8 evidence.

Post-final7 local work adds exact `/healthz`, `/readyz`, and `/metrics`
observation, a fixed inherited Unix-stream drain
capability for the opaque CLI, and an independent descriptor-6 read-once
inference-credential channel. The drain channel validates descriptor 5 before
model/device/listener construction, admits one bounded command, drives the
singular runtime owner, closes listeners before the channel, and has native ABI,
ASan/UBSan, allocation, malformed/replay, peer-loss, CLI-activation, and
lifecycle coverage. OpenAI Responses serves observation on the same HTTP
origin as inference and creates no second control listener; native-framed mode
retains a separate loopback-only HTTP control listener. Authenticated policy v3
admits either exact loopback plaintext or wildcard-only
`private_network_plaintext` for an isolated container network, and enforces its
credential ceiling. LunaNexa maps a wildcard origin to the inspected private
container address before health/readiness probes and invocation. Native v1/v2
accept no credential and OpenAI v2 remains fail-closed. The credential
source channel and scratch are closed/wiped before runtime resource creation.
The opaque API-auth policy owns an idempotent full-buffer wipe; verifier, HTTP,
OpenAI server/pool, terminal drain, and startup-rejection paths propagate that
singular close. HTTP operation reuse wipes used head/body cells and terminal
close wipes its complete fixed request storage. It deliberately has no public
drain route, TLS, external authorization, or generation fencing. Those
deployment contracts and physical restart/routability tests remain release
gates.

### Gate

- common model starts with one command;
- every automatic choice has an explanation;
- unsupported options fail before readiness;
- fixed-seed sampling is replayable;
- CLI, native API, and OpenAI adapter use one canonical request vocabulary;
- cold-start and readiness measurements are published.

## Phase 7 — Tensor parallelism

The separately authorized [`parallel` IR workstream](PARALLEL_IR.md) implements
immutable memory effects, dependency/lifetime analysis, cross-rank collective
agreement and pure event/queue lowering from admitted TP execution plans.
The default ordered execution path is preserved. An opt-in backend now binds
compute/NCCL queues to separate streams and preallocated events, and requires
physical numerical, resource-balance and performance evidence.

The separately authorized [`twospark` workstream](TWO_SPARK.md) develops an
explicit two-node network capability. It preserves this phase's local-only
admission. Remote rank execution, session supervision, service composition and
an operator-launched rank entry point have host implementations and tests;
two-node physical qualification and production TLS-provider admission remain
separate release gates.

### Outcome

One model can execute across multiple local GPUs without changing request,
scheduler, KV, or public API semantics.

### Deliverables

- topology probe and explicit supported topology declaration;
- NCCL private ABI and lifecycle tests;
- row/column tensor-parallel model plan;
- sharded weight materialization directly to destination workers;
- per-rank KV arenas and collective sequence;
- rank-coordinated plan generation and failure generation;
- collective timeout and worker-loss handling;
- one production worker-service/online path over rooted single-worker or
  generation-scoped whole-group transport, including exact predecessor
  retirement and deterministic replacement/close;
- multi-device capacity and communication explanation.

Pipeline parallelism and cross-node execution remain excluded.

The 2026-08-27 two-GPU host is intentionally not promotion evidence: its RTX
2080 lacks BF16 and peer access is unavailable in both directions. Static
topology/admission/collective/rank gates pass, while this unsupported physical
topology is correctly rejected. Physical one-versus-multi-GPU numerical and
performance gates require a homogeneous supported node.

The product-owned `phase7_topology_diagnostic` command now replaces ad-hoc
GPU inventory evidence for this gate. It emits immutable schema-v1 evidence
for exact device identities and capabilities, every directed peer query, the
dynamically admitted collective runtime, and the existing typed local topology
admission. Unsupported topology and missing collective libraries are successful
diagnostic rejections; they never enable a degraded tensor-parallel path or
open contexts, allocations, communicators, or rank processes.

The 2026-08-28 current-source capability rerun repeated this exact rejection on
the physical sm120/sm75 host, confirmed no peer access in either direction and
no installed NCCL runtime, and left no context, communicator, rank process, or
GPU compute process. Its focused admission suite passed 176/176. This closes
the unsupported-topology fail-closed test only; the positive homogeneous
tensor-parallel numerical and performance gate still requires eligible
hardware and NCCL.

The production software route is now complete for an exactly admitted,
homogeneous peer-capable topology. A distinct descriptor/launch/source/child
path binds ordered ranks and device ordinals, targets, shard and KV plans,
collective policy and versions, executable identities, approved roots,
external kernel approval, and service ceilings. Preflight derives a nonzero
canonical group-template digest from those immutable claims; the live group
bootstrap digest commits that template as its parent before adding generation,
rendezvous, and rank-process state. Every child independently re-admits the
same source and rank contract, and the rooted service receives only the
complete group owner. Materialized-bundle substitution, zero-template,
restart, cleanup, ordering, and fake multi-rank gates pass. These are software
and fake-transport results only, not positive NCCL, numerical, capacity, or
performance evidence.

### Gate

- one- and multi-GPU outputs match within tolerance;
- no worker loads a complete copy of a sharded model;
- collective order is deterministic;
- rank failure fails the generation without deadlock;
- supported topology improves capacity or performance against one GPU;
- unsupported topology fails startup.

## Phase 8 — Quantization and model-family expansion

### Outcome

Capability breadth grows through immutable model plans, not scheduler branches.

### Sequence

1. FP8 on hardware with validated support;
2. one weight-only quantization format;
3. additional dense decoder family expressed by existing operations;
4. only then evaluate MoE, LoRA, speculative decoding, and multimodal plans.

Each addition supplies:

- format/schema validator;
- plan capability;
- kernels and manifest entries;
- accuracy fixtures;
- memory and performance benchmark;
- invalid-combination matrix;
- no new branch in core scheduler policy.

The weight-only software foundation now implements one closed symmetric-I8
per-output-channel format for dense Llama: canonical numeric schema and plan
construction, authenticated bounded file inspection and direct device-layout
materialization, exact catalog-v4/artifact/launch admission, manifest-v4 and
bootstrap-v3 joins, a serialized executor, strict runtime descriptor v2, common
worker recipe v6, independently admitted instance policy/tokenizer/scheduler
authority, digest-authenticated launch-v2 routing through the existing native
service owner, verified worker-executable release wiring, and deterministic
reverse-order cleanup with retry ownership. Software target admission is closed
over exact `sm89`, `sm90`, and `sm120`; adjacent architectures fail before
authority. The older `sm120` rejection campaign belongs to a prior sealed
policy and is not current target truth. Physical numerical correctness,
sanitizer/leak balance, device soak, accuracy, memory and performance
measurements, and the production hardware readiness decision remain open.

The additional dense-family foundation now strictly admits one bounded
`MistralForCausalLM` semantic profile. Content-bound metadata and exact
configuration semantics feed the existing dense stateless and paged
21-operation plan; when `sliding_window` is present, admitted context is capped
so the existing full causal attention remains exact. Exact BF16 weight
binding/materialization and strict regular or materialized descriptor routes
now reuse the common BF16 worker/service owner without scheduler or KV
branching. Focused family, weight, descriptor, worker-wire, materialized-route,
and socket-backed runtime gates pass. A digest-authenticated fixture-only corpus
now proves exact dense-plan and 21-weight equivalence and replays logits,
argmax, and continuations without production authority. A production-approved
artifact and correctness corpus, physical numerics, accuracy, memory,
performance, and readiness remain open.

FP8 remains earlier in the same sequence. Its local schema, device capability,
AOT-manifest, inert startup joins, and authenticated finite-E4M3 numeric-weight
materialization pass their software gates. The loader accepts only canonical
`F8_E4M3`, rejects both E4M3FN NaN encodings and invalid scalar-F32 scales in a
bounded first pass, and reauthenticates before copying into the final arena.
The production dense-Llama builder now emits the complete all-or-nothing FP8
schema for full-context and paged plans, and a family facade delegates exact
file authentication to the canonical numeric loader with typed plan/file
failures and identity anti-replay. A host-only dynamic-scale ABI joins that
plan to admitted manifest/launch authority, exact profile/shape, scalar scale
regions, and one whole caller-owned Workspace with a fenced lifecycle. Version
2 accepts only exact numeric PagedV4 launch authority and digest-binds complete
dimensions plus ordered raw operands. It derives one external-input F32 scale
cell for simple projections and adds an independent post-SiLU product cell for
compound gated MLP. A separate deterministic CUDA 12 AOT lowerer and closed
compile/artifact/symbol binder cover QKV, output, head, and gated MLP while
preserving the pinned v1 source path. Exact-shape v2 remains inert and
single-shot. The additive reusable PagedV4 v3 route reconstructs a digest-pinned
schema-v5 execution manifest, accepts only opaque externally approved CUBIN
release authority, re-admits the model/plan/profile/target/raw ABI/workspace/KV
envelope in the child, and prepares the existing mixed graph executor before
the common rooted service can publish readiness. Unique four- or eight-byte
scale evidence is sentinel-initialized and every cell must be finite and
positive before output publication. Regular and materialized descriptor,
launch, worker, service, and one-argument CLI routes pass focused hostile,
cleanup/restart, and socket-backed software gates. Exact `sm120` is admitted by
the device, descriptor, bootstrap, and CUDA-v2 software routes alongside
`sm89`/`sm90`; adjacent architectures fail closed. No current-source physical
FP8 numerical campaign has qualified this route, and physical
numerical correctness, sanitizer/leak balance, soak, accuracy, memory,
performance, and release promotion remain open.

### Gate

- accuracy and task-quality budget is declared and passed;
- memory improvement is measured;
- unsupported hardware fails startup;
- existing BF16 and prior model-family benchmarks do not materially regress;
- scheduler and KV public APIs remain architecture-neutral.

## Phase 8b — Advanced model workloads

H3 execution follows [MINIMAX_H3_EXECUTION.md](MINIMAX_H3_EXECUTION.md): share
the compiler/device foundation but keep the joint-diffusion iteration owner
separate from autoregressive scheduling. Complete a bounded numerical path
before expanding serving or optimizing individual candidate counts.

The [2026-09-26 code inventory](MINIMAX_H3_CODE_INTEGRATION_2026-09-26.md)
records the text/vision/reference encoders, full output decoders, prepared joint
request, component weight startup and offline source export. Older phase notes
below describe their earlier implementation boundary, not current module absence.
Full-checkpoint GPU correctness, memory and performance qualification is pending.

### Outcome

DeepSeek V4, GLM 5.3 full and Flash, and MiniMax H3 become executable through
the typed workload boundaries accepted by ADR-0017. This phase does not weaken
the original dense-model gates or reinterpret diffusion work as token decode.

### Sequence

1. construct exact, content-bound family profiles from official configuration;
2. emit reusable advanced-decoder, multimodal-conditioning, and joint-diffusion
   plans with canonical identities and complete tensor vocabularies;
3. materialize every declared BF16, FP8, FP4, scale, index, recurrent-state,
   vision, audio, and video tensor directly into its final bounded layout;
4. implement correctness-only reference operations and deterministic family
   corpora before granting any production execution authority;
5. add content-addressed AOT kernels, launch contracts, catalog admission, and
   device executors for every operation used by each exact profile;
6. add workload-specific bounded request/output codecs and compose them with
   the common lifecycle owner without importing family types into scheduler,
   KV, device, or public lifecycle packages;
7. pass deterministic correctness, sanitizer, leak, cancellation, cleanup,
   memory, throughput, latency, and task-quality campaigns on named hardware.

Each family is promoted independently. A green semantic plan, weight schema,
or reference interpreter is progress evidence but is not executable readiness.
The current startup path includes family-neutral streaming safetensors,
bounded final host arenas, a canonical segmented device layout, opaque
synchronous arena borrowing, and thin DeepSeek/GLM/MiniMax raw-upload
projections. These raw allocations carry no kernel or execution authority.
MiniMax now authenticates and streams the complete official VideoVae
vocabulary as 703 F32 tensors with 10,415,475,936 payload bytes across three
digest-bound shards and the complete AudioVae vocabulary as 1,087 F32 tensors
with 605,306,340 payload bytes in one authenticated file. Metadata-only
fixtures prove both bounded segmented arena plans without allocating their
payloads. Denoiser, Qwen3-VL, VideoVae, and AudioVae component materialization
are therefore positive; whole-model execution remains fail-closed, and no VAE
encode/decode execution authority is implied.
The first offline CUDA foundation candidates use family-neutral source
renderers for exact BF16 embedding, dynamic/paired/fixed-row RMSNorm, no-bias
dense projection, and fixed-row F32+bias projection with BF16 or F32 output.
Thin advanced-decoder,
GLM, and joint-diffusion adapters bind family requirement identity, full
geometry, compiler policy, launch shape, operands, and content-digested
recipes. Advanced Decoder and GLM both retain their exact normalization epsilon
in final-RMS requirements; GLM also retains it in the per-layer decoder-RMS
requirement.
DeepSeek V4's five profiles and both GLM 5.3 profiles now lower their terminal
language-model heads through that shared renderer with exact hidden/vocabulary
geometry and a four-operand step-count/hidden/weight/logits ABI.
DeepSeek additionally models query low-rank A and B projections as separate
ordered requirements around its independent low-rank query/key RMSNorm, rather
than treating the path as one fused projection. Exact E4M3/UE8M0 scalar
semantics support the replicated Query-A candidate; Query-B remains open. It
also lowers its
routed-expert-width-aware F32 piecewise sqrt-softplus score transform. Its
manifold-constrained hyper-connection contract is now split into four
versioned operations rather than a generic mix: block-control construction,
pre reduction, post residual combine, and head reduction. The bounded scalar
reference follows the official normalized control projection, sigmoid gates,
stable row softmax, 20 alternating Sinkhorn normalizations, and separate
attention/FFN sites with F32 controls and BF16 activation boundaries. The
four phases now have separate family-neutral serial correctness sources.
Block control has an exact eight-operand ABI: counts, BF16 state, separate F32
function/base/scale inputs, and F32 pre/post/combination outputs. Pre-reduce
has an exact four-operand ABI: counts, BF16 stream state, F32 pre-mix rows, and
BF16 reduced rows, with ordered F32 accumulation and one output round.
Post-combine has an exact six-operand ABI: counts, BF16 branch/state inputs,
F32 post/combination controls in destination-major/source-minor order, and a
BF16 combined-state output. It computes the branch term followed by ordered
ascending-source accumulation and one output round. Head-reduce has a separate
six-operand ABI for counts, BF16 stream state, F32 function/base/scale inputs,
and one BF16 reduced head output. Complete artifact admission authenticates all
four candidates, accepts the exact replicated Query-A and pre-normalization
KeyValue block-FP8 projections, and then remains fail-closed at Query-B; no mHC
phase inherits a neighboring ABI.
Its
first hash-routed layers also have a fail-closed I32 lookup candidate. A
family-neutral startup owner now narrows the official I64 tables directly from
the final authenticated host arena into bounded, per-row-unique I32 sidecars;
the DeepSeek adapter binds exact layer/source identities and zeroizing release.
The same adapter projects the three exact Flash-0731 sidecars into a canonical
segmented device layout and validates the borrowed source before upload. A
focused integration owner joins each uploaded sidecar region to token-hash
operand 2 and authenticates that standalone inert plan against an admitted
module/entrypoint/launch ABI.
It additionally joins a canonical offline compile receipt and two byte-identical
module snapshots to the exact source, recipe, compiler policy, driver identity,
target, symbol, and admitted module digest. This evidence is self-consistent but
does not authenticate its producer or prove compiler execution, so it still
refuses compiler provenance, loader, launch, and physical authority;
a family-neutral operand-region planner now adds the exact step-count, token-ID,
table, and routing-output allocation-relative regions, validates capacity,
alignment, and non-overlap, and canonically joins all four to the upstream
evidence. It does not complete a full-model artifact because admission stops at
Query-B. Live allocation ownership and executor
composition remain open.
Non-hash layers now lower deterministic lower-expert-ID biased top-k, while a
separate selected-weight candidate gathers the original unbiased scores,
normalizes, and scales both score-selected and hash-selected rows. Official
PyTorch tied-top-k parity remains unqualified because its tie indices are not
stable. A separate RoPE-only candidate covers the compressed/scaled YaRN path
for all five DeepSeek profiles with exact 64-wide adjacent-pair BF16 geometry,
unit-magnitude official interpolation, and independent query/KV operands.
Plain-theta sliding layers remain a distinct gap because the current
requirement has no authenticated layer-mode selector. A separate deterministic
ratio-128 compressed-index candidate consumes bounded I32 query positions and
cache offsets and produces per-row counts plus dense `-1`-padded I32 slots;
the obsolete fused compressed-attention placeholder is now decomposed into
window/compressed index join, non-overlapping shared-KV preparation, selected
sparse attention, and output inverse-RoPE. The selected-attention phase has an
exact eight-operand correctness candidate with shared BF16 K/V, I32 indices,
an F32 per-head denominator-only sink, stable ordered-F32 softmax, and one
BF16 output round. The join phase now has its own exact eight-operand I32
candidate preserving causal prefill padding, decode circular-ring ordering,
compressed `-1` padding, and literal window-then-compressed concatenation.
A seven-operand borrowed-buffer preparation candidate now materializes ordinary
and completed compressed BF16 K/V in the exact prefill layout, or the physical
128-slot circular window followed by the completed compressed prefix in decode.
The following inverse-RoPE phase now has an exact in-place three-operand
candidate: the first 448 BF16 components of every 512-wide head remain
bit-exact, while the final 64 adjacent-pair components receive conjugated F32
scaled-YaRN rotation and one BF16-RNE write. The following output projection is
now split at the official storage boundary. Group-local Output-A consumes the
converter's authenticated BF16 `[groups,1024,4096]` layout and has a separate
ordered-F32/BF16-RNE candidate plus an inert layout-to-operand join. Output-B
has a separate single-rank candidate with dynamic per-row/block-128 UE8M0
activation scaling, E4M3 weights, UE8M0 block scales, ordered F32 accumulation,
and one BF16-RNE output. Its inert join authenticates official source storage
and the candidate operand without granting materialization, upload,
tensor-parallel collective, or launch authority. Learned compression, actual
Output-A/Output-B materialization, cache-state mutation, KV ownership,
collective execution, and launch remain separate typed gaps.
The preceding replicated Query-A projection now uses the same exact dynamic
per-row/block-128 E4M3/UE8M0 arithmetic through the family-neutral renderer.
Profile-owned storage/layout binding fixes `4096→1024` for Flash profiles and
`7168→1536` for Pro profiles without granting interpretation fallback,
materialization, upload, compilation, or launch authority. A separate exact
replicated KeyValue candidate applies the same block-128 arithmetic to
`hidden→512`, preserves the official pre-KV-normalization boundary, and binds
an inert storage/layout join. Query-B remains the next typed projection gap.
GLM
lowers the exact bias-free DSA output projection for both profiles and the
standalone first-valid, complete-pool key compression stage for Flash. Separate
correctness-only sources now cover ordered projected F32 scoring, stable top-k,
raw-pool expansion, Flash visible-tail append/pad semantics, and Full-profile
bit-exact adjacent-layer index reuse. A separate selected-index sparse-attention
oracle binds the exact Full width 2048 and Flash width 2051, derives the unique
KV-head geometry, rejects invalid/duplicate/future indices by zeroing the whole
query row, and performs ordered F32 QK, stable softmax with BF16 probabilities,
and ordered F32 PV. The score/select and attention sources are deliberately
serial and consume already projected tensors. Exact query-A and key/value-A
BF16 dense projection candidates now cover the first input-projection stage for
both GLM profiles while keeping the compressed and RoPE suffix geometry
explicit. An exact query-A RMSNorm candidate covers rank 2048 for Full and 1536
for Flash with epsilon 1e-5. Exact query-B projection candidates map those
normalized ranks through `[16384, rank]` BF16 weights into zero-copy
`[token,64,256]` head views, keeping no-PE components before RoPE components.
Exact KV-B projection consumes normalized 512-wide compressed rows and emits
zero-copy `[token,64,key-no-PE then value]` views: `[28672,512]` Full weights
for 192+256 components and `[32768,512]` Flash weights for 256+256 components.
An exact KV-A split/RMSNorm candidate now normalizes the 512-wide BF16 prefix
with epsilon 1e-5. Full emits the 64-wide RoPE suffix through a separate
bit-exact operand, while Flash authenticates a zero-suffix four-operand ABI.
Full additionally lowers one interleaved-RoPE candidate with theta 8,000,000:
each 256-wide query head preserves its 192-wide no-PE prefix bit-exactly and
rotates the 64-wide suffix, while one shared 64-wide key suffix is rotated in
the same six-operand ABI using adjacent-pair input and split-half output. A
family-neutral bit-exact assembly candidate now splits packed KV-B into
contiguous values and no-PE keys, broadcasting Full's rotated shared 64-wide
suffix across 64 heads. Flash uses a distinct no-suffix ABI rather than a fake
zero-byte operand. Live cache handoff, paged-KV throughput implementation, and
physical qualification remain open; the assembly cannot be substituted for
neighboring projections.
Both GLM profiles also lower exact dense SwiGLU through a separate-weight
family-neutral renderer. Full binds 6144 hidden and 12288 intermediate width;
Flash binds 4096 and 12288. Six operands keep counts, hidden rows, gate, up,
down, and output distinct. Ordered F32 projection sums and explicit BF16
gate/up, SiLU, product, and final-output rounds cannot be confused with the
packed MiniMax weight layout.
Both GLM profiles additionally lower the official bias-free MoE router linear
projection through a family-neutral F32-input/BF16-weight renderer. Full fixes
6144 hidden features and 256 expert rows; Flash fixes 4096 and 288. Ordered F32
accumulation produces raw F32 logits. A distinct five-operand F32 stage emits
both uncorrected sigmoid scores and correction-biased choice scores, rejects
aliased outputs, and replaces the redundant downstream sigmoid transform.
A following five-operand grouped-routing stage uses corrected scores for
top-two group scoring, group selection, and expert top-eight choice, then
gathers uncorrected scores, normalizes by the ordered sum plus `1e-20`, and
scales by 2.5. Lower expert IDs break exact ties deterministically, while
PyTorch unsorted-`topk` tie parity remains unqualified. Downstream routed
expert, shared expert, and routed-plus-shared composition now have three
separate correctness candidates. Full uses unclamped SwiGLU at hidden 6144;
Flash uses hidden 4096 and clamps gate/up at 10. A generic indexed bank layout
and thin GLM join bind every per-expert checkpoint tensor name, ordinal, shape,
offset, and digest to the candidates without claiming packing or upload
authority. Full then lowers the exact conventional BF16 outer residual add;
Flash lowers the exact four-stream mHC composition
`post * branch + combineᵀ * residual` with staged BF16 controls, ordered F32
matrix accumulation, and one BF16 output round. Two separate correctness
candidates now produce the upstream controls: learned function/base/scale
projection plus collapse, then positive-matrix Sinkhorn with one initial column
normalization and 19 row/column passes. Distinct intermediate buffers and
serial launch geometry are explicit; live-state and execution authority remain
unmet.
GLM Flash additionally lowers an exact BF16 Q/K/V, low-rank forget-control,
and beta input-projection candidate while retaining the per-component F32 decay
control as its own unmet operation. It lowers one family-neutral KDA causal
short-convolution candidate reused independently for Q, K, and V. Its ABI binds CSR sequence
offsets, BF16 `[8192,1,4]` depthwise weights, and explicit oldest-first
three-token initial/final history. Ordered F32 convolution and SiLU end in one
BF16 output round. A separate family-neutral serial recurrent-delta candidate
binds BF16 Q/K/V and beta, per-key-component F32 log decay, explicit F32
`[sequence,64,128,128]` initial/final state, CSR boundaries, normalized delta
update, and one BF16 output round. A bias-free two-stage BF16 projection
produces the KDA output gate, and a separate exact sigmoid-gated per-head
RMSNorm performs ordered F32 normalization and one final BF16 round. The final
output projection, decay control, cache/state ownership, and throughput CUDA
remain separate unmet operations.
All candidates remain deliberately non-bindable until compilation and physical
qualification.

The compact routed-expert path added on 2026-10-09 is a separate executable
integration, not a promotion of those inert candidate bundles. A pure precision
plan shares bank offsets between bounded checkpoint upload and AOT lowering.
Three stages compute gate/up, down, and rank-local F32 combination directly from
NVFP4/MXFP4/BF16 weights with BF16 activations. It binds through the existing
ordered executor. Small deterministic GB10 execution and memory/race/sync tests
cover this implementation; full checkpoint execution, tensor-core expert GEMM,
cross-rank reduction, and complete model serving are still unfinished. DeepSeek's
dynamic-FP8 route now has separate five-stage expert execution: BF16 input
quantization, gate/up, product quantization, down, and local combination.
Pure precision IR explicitly selects amax floor, scale blocks, product rounding
and routing-score placement. The family adapter supplies checkpoint names and
geometry; bounded upload and ordered execution remain shared. Shared-expert
binding and the prepared cross-rank MoE execution queue are now implemented.
Communication preserves F32 routed accumulation; replicated shared
experts are added once after reduction, followed by the explicit BF16 output
round. Shared experts reuse the same packed MLP plan/upload/lowering rather than
a family-specific runtime. The generic combination plan records whether routed
results round before shared addition. A prepared ordered queue inserts the
collective between routed and final combination, without filesystem checks,
crypto, request-path JIT, or new TLS requirements. Native queue tests verify
allocation-free repeated execution and ordering. Bounded GB10 shared-expert
and finalizer tests pass correctness, memory, race and synchronization checks.
Actual two-host reduction, full checkpoint serving and tensor-core expert
performance remain separate unfinished work.
Joint diffusion now also has deterministic, request-shape-specialized mixed-
precision video/audio latent-input projection, BF16 video/audio final-RMSNorm,
row-aligned staged-BF16 final AdaLN-to-F32 boundaries, F32 video/audio
output-head, and BF16 shifted-flow Euler candidates.
Projection recipes fix F32 packed inputs, separate row-major F32 output/input
weights and bias, ordered F32 accumulation, one BF16 output rounding, and exact
video-patch/audio-row order.
Final-normalization recipes authenticate exact selected-row geometry, BF16
weights and activations, rational epsilon, and ordered F32 reduction with one
BF16 output round. A distinct final-AdaLN parameter-row candidate consumes one
F32 conditioning row, applies F32 SiLU with one BF16 round, then ordered F32
dense projection from BF16 weight/bias to one BF16 shift/scale row. Boundary
recipes consume already row-aligned BF16 normalized, shift, and scale buffers,
round each affine stage to BF16, then widen exactly to F32 head input. Upstream
Request admission now owns canonical finite F32 distinct timestep values,
packed row-to-timestep maps, and video/audio modality selections. A
family-neutral candidate renders the exact 256-wide flipped cosine/sine
frequency rows, an ordered-F32 dense-SiLU-dense candidate maps each distinct
row through `256→5376→2688`, and a multirow candidate maps every authenticated
conditioning row through shared BF16 `[10752,2688]` weight and `[10752]` bias
into BF16 `[R,5376]` shift/scale tables. A typed inert composition checks the
three stages' identities, digests, rows, ordinal order, and adjacent widths,
while a
bit-exact eight-operand BF16 gather candidate materializes the selected shift
and scale rows without device-side diagnostics. Runtime construction and
binding of those request-owned buffers into the downstream gather remain open.
The exact 50 denoiser and 2 token-refiner transformer MLPs now share a
family-neutral bias-free packed gate/up SwiGLU plus dense-down source. Its
request-bound candidates authenticate BF16 `[28672,5376]` up/gate weights,
BF16 `[5376,14336]` down weights, separate packed/refiner row scopes, and the
official first-half-up/second-half-gate order without fusing residual,
normalization, attention, or AdaLN.
The same 50+2 scopes now lower a staged joint-attention candidate. Packed
attention performs BF16 Q/K/V projection, Q/K RMSNorm with epsilon 1e-5,
96-wide rotary from authenticated F32 cosine/sine rows, full noncausal unmasked
attention with ordered F32 accumulation and BF16 stage rounds, then the BF16
output projection. Refiner attention uses the same staged contract without
rotary. Residual, pre-attention AdaLN/RMSNorm, and gates remain separate.
VideoVAE and AudioVAE decode now begin with separate exact F32 latent
destandardization candidates. They retain the official 24- and 32-channel
means and standard deviations, use ordered multiply then add without FMA, and
bind request-derived `[1,24,37,48,84]` and decoder `[2,32,207]` geometry. The
stereo dimension is explicitly decoder batch two. AudioVAE additionally lowers
its exact F32 `dec_in_proj` pointwise Conv1d from `[2,32,207]` to
`[2,2048,207]`, then an exact two-stage F32 weight-normalized kernel-7
`decoder.conv_pre` to `[2,1024,207]`. The first stage writes the explicit
58,720,256-byte normalized-weight workspace; the second performs padded
channel-major convolution. The immediately following `decoder.ups.0.0` is an
exact two-stage, per-input-channel normalized kernel-9, stride-5
ConvTranspose1d candidate producing `[2,512,1035]` through an explicit
18,874,368-byte workspace. The first following AMP activation is an exact
three-launch F32 candidate: replicate-padded depthwise upsampling to 2,070 time
cells, SnakeBeta, and replicate-padded stride-2 downsampling to 1,035, using
distinct checkpoint filters and two 8,478,720-byte workspaces. Here AMP means
anti-aliased multi-periodicity, not autocast. The complete first three-way AMP
residual stage now follows as one 97-launch inert candidate. It authenticates
126 ordered F32 parameter regions, keeps the original upsampler output
distinct from the precomputed block-0 activation, applies all weight-normalized
dilated convolution pairs and residual additions in `30+33+33` stages, then
performs one ordered three-way sum/divide-by-three launch through a
58,167,296-byte disjoint workspace plan. The following `decoder.ups.1.0`
transition has its own per-input-channel weight-normalized kernel-9, stride-5
ConvTranspose1d candidate from `[2,512,1035]` to `[2,256,5175]` with an
explicit normalized-weight workspace. The complete second three-way AMP stage
then binds `resblocks.3–5`, 126 ordered parameter regions, kernels 3/7/11,
dilations 1/3/5, a 119,465,984-byte disjoint workspace, and 97 exact launches
over `[2,256,5175]`, preserving the precomputed first activation and ordered
final average. The next `decoder.ups.2.0` kernel-4/stride-2 transition to
`[2,128,10350]` and its following `resblocks.6.activations.0` alias-free
activation are exact, with per-input-channel normalization, distinct length-12
filters, and explicit intermediate workspaces. The VideoVAE decoder body,
complete `resblocks.6–8`, `decoder.ups.3.0`, later AudioVAE stages, and final
output remain typed gaps.
Conditioning execution remains a deliberately negative boundary. A family-neutral
reference assessment binds the exact plan, requirement, request digest,
operation ordinal, workflow, request-derived geometry, and ordered missing
prerequisites. The authenticated VideoVae and AudioVae architectures, weights,
and inverse-normalization prefixes are no longer missing prerequisites. FL2VA derives two 1008-row keyframe latents of
width 96 but records missing payload, VAE encode execution,
posterior/normalization, noise/scale arithmetic,
and row-span authority. Ref2VA records per-reference encoded geometry and
video-soundtrack association among eleven blockers. The assessment has no
source, recipe, compiler, target, entry-point, artifact, or execution API and
its require operation always fails typed.
Output-head recipes separately authenticate F32 normalized hidden rows,
row-major F32 weights, F32 bias and F32 predictions; their storage contract is
distinct from the input projection's final BF16 round. Flow recipes bind
rational scheduler parameters. All bind the exact plan
and requirements identity, resolved latent geometry, compiler
policy, target, launch, and artifact-compatible operands; they likewise remain
uncompiled, unqualified, and non-executable.
A family-neutral startup-only coverage audit and separate workload-specific
candidate-evidence adapters report exact missing and non-bindable requirement
ordinals without granting artifact or execution authority. Flash-0731 proves
22 inert candidates across 40 requirements with 18 gaps, GLM Full proves 24
inert candidates across 28 requirements with 4 gaps, GLM Flash proves 32
inert candidates across 47 requirements with 15 gaps, and each MiniMax profile
proves 29 inert candidates across 33 requirements with 4 gaps. Completion
remains a fail-closed promotion prerequisite.
DeepSeek V4 and GLM 5.3 full retain decoder scheduling semantics while their
MoE and sparse-attention operations are execution capabilities. GLM 5.3 Flash
adds a bounded vision-conditioning frontend. MiniMax H3 owns a separate
audio/video diffusion owner and never enters the token-step scheduler.

### Gate

- every official registered profile loads from an exact bounded configuration
  and rejects neighboring architectures or partial metadata;
- every plan tensor is authenticated, uniquely bound, and materialized without
  an intermediate full-model copy;
- every semantic operation has matching reference, numeric, AOT artifact,
  launch, catalog, and device-executor evidence;
- family outputs match independently accepted fixtures within declared numeric
  and task-quality budgets;
- cancellation and every startup/runtime failure release all host, device,
  stream, event, graph, and file resources exactly once;
- unsupported profiles, formats, targets, and partial capability sets fail at
  startup without dense-family fallback;
- scheduler, KV, lifecycle, and device public APIs remain family-neutral.

## Phase 9 — LunaNexa runtime integration and release

### Outcome

LunaNexa can deploy, observe, drain, and invoke LunaFlux as an opaque approved
runtime.

### LunaFlux deliverables

- digest-pinned OCI image;
- read-only model-mount contract;
- health/readiness endpoints;
- native and OpenAI compatibility contract versions;
- bounded metrics and shutdown/drain behavior;
- SBOM, license, kernel manifest, and build provenance;
- deployment guide containing no LunaNexa credential assumption.

### Integration ownership

The LunaNexa repository owns the LunaFlux adapter, image allowlist, deployment
template, fleet policy, and artifact materialization. LunaFlux accepts the
generic runtime inputs and never imports the adapter.

The sibling LunaNexa repository has an inert digest-only
`LunaFluxCandidateManifest` plus a strict `LunaFluxPromotionSubmission` for the
serving target, listener, health/readiness, drain, OpenAI Responses invocation,
benchmark evidence, and eight independent approval subjects. Structural
inspection still adds `PromotionVerifierUnavailable` and reports
`routable=false`. A separate authenticated ten-subject verifier now owns a
deployment-injected HMAC-SHA256 key, checks domain-separated signed receipts
for candidate, OCI, SBOM, provenance, license, kernel, physical-serving,
benchmark, control, and invocation authority, and privately constructs the
catalog that can produce opaque adapter authority. The public package exposes
no catalog or approval-input constructor. Production key acquisition and
policy, issued receipt sets, and an approved catalog entry remain external; no
such production inputs are committed and no rollout has been promoted. The
authority-only opaque adapter projection now
retains exact candidate, image, model, launch, control, invocation, and catalog
identity; it exposes approved health/readiness and inherited-drain facts while
keeping inference credentials call-scoped. Approved OCI and
target evidence, deployment translation from authenticated external drain to
the inherited local capability, inference credentials, physical serving,
benchmark comparison, catalog/template publication, and reviewer acceptance
remain external gates.

LunaFlux now owns a separate inert `lunaflux.final-release-inventory.v1`
assembler/verifier for the final output join. It binds eight distinct,
externally authenticated artifacts for the OCI digest, SBOM, license inventory,
build provenance, kernel manifest, complete-rootfs scan, runtime contracts, and
exact source identity. The deterministic read-only output preserves the exact
authenticator and release-tool bytes, rejects overwrite, substitution,
symlinks, hard links, cross-subject replay, and partial publication, and replays
every approval during verification. The authenticator identity and its trust
policy remain externally allowlisted; fixture authenticators and the inventory
wiring itself grant no release, reviewer, deployment, or readiness authority.
An actual approved image and authentic artifacts have not yet been supplied.

The live benchmark-campaign software boundary is also implemented without
granting measurement or comparison authority. A deterministic external-process
orchestrator consumes one authenticated 81-trial declaration and separately
digest-pinned campaign replay, trial-driver, correctness, live-engine-identity,
and process-supervisor executables. It fixes the three-engine/nine-profile/
three-trial matrix, Latin-square order, identical protocol/tokenizer/input
constraints, bounded timeout/cancellation, credential-descriptor ingress, raw
capture sealing, per-trial correctness, and process-group cleanup. The trial
driver cannot assert engine identity: each trial requires a separate sealed
observation binding the exact declared revision, image, configuration, and
executable digest. Independent verification replays preflight, correctness,
raw Responses framing, cleanup receipts, and the existing comparison-admission
tool. Hostile fake-engine, identity substitution, malformed framing, timeout,
partial-output, and no-overwrite gates pass locally. Pinned live engines,
credentials, externally approved identity/correctness policies, physical
measurements, and comparison/reviewer acceptance have not been supplied.

### Final campaign

1. deploy one approved model on one target GPU node;
2. run correctness corpus through direct and LunaNexa paths;
3. execute latency, chat, long-prefill, prefix, saturation, and churn profiles;
4. cancel, drain, restart worker, and restart container;
5. verify readiness and bounded failover behavior;
6. scan logs, responses, environment, and image for secrets and payload leaks;
7. export raw benchmark and release evidence;
8. compare against pinned vLLM and SGLang under the same workload.

### Release gate

- all supported features pass on physical target hardware;
- reference correctness remains accepted;
- no native leak or structural boundary violation is found;
- performance results are published, including losing workloads;
- LunaFlux and LunaNexa build independently;
- named security, operations, and performance reviewers accept the evidence.

## Deferred capabilities

These require separate architecture decisions after the first release:

- speculative decoding;
- LoRA multiplexing;
- MoE expert parallelism;
- prefill/decode disaggregation;
- cross-node tensor or pipeline parallelism;
- ROCm or other device backends;
- production runtime kernel JIT.

Deferral is deliberate. None may add hidden branches or placeholder flags
before its phase is approved.
