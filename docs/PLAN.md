# LunaFlux detailed implementation plan

## Working policy

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

The standard phase gate is:

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

Implement lunaflux doctor and lunaflux plan before serve. Doctor reports driver,
device, ABI, model-file, and kernel-manifest problems. Plan reports memory,
shape, and capability decisions without materializing weights.

### Workstream 2: tokenizer

- parse bounded tokenizer.json;
- implement byte-level BPE normalization and merge behavior needed by the
  selected reference tokenizer;
- define TokenizerSpec and TokenizerDigest;
- test Unicode, invalid UTF-8 input policy, special tokens, truncation, and
  deterministic encode/decode;
- compare token IDs against a pinned reference corpus.

SentencePiece and arbitrary normalizer plugins are excluded.

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
module/function launch seam are implemented foundations. They do not promote
Phase 1 without physical-CUDA numerical, ownership, sanitizer/leak, soak,
benchmark, and resource-balance evidence. See
[STATUS.md](STATUS.md) for the current boundary.

## Phase 2 — Online streaming service

### Outcome

The reference engine becomes a bounded, cancellable single-request service with
a native protocol and compatibility adapter.

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
Completion retirement, stop-token/output enforcement, generated-token
publication, global fairness/preemption, and live worker integration remain
open.

### Workstream 4: worker overlap

- scheduler constructs plan N+1 during plan N execution;
- events distinguish submission, execution, and completion;
- steady-state host buffers and rings are preallocated;
- stale plan and completion generations are rejected.

The immutable fixture vocabulary, reusable fixed-capacity plan/completion
buffers, authenticated lifecycle epochs, scalar row drafts, provenance-bound
capability recipes, whole-build checkpoints, final-prefill output semantics,
and stale-generation checks are implemented. Scheduler-owned distinct A/B
buffer pairing and build/submission integration are implemented. Authenticated
completion retirement, live worker overlap, and the runtime
allocation-instrumentation gate remain open.

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

SGLang-style token-prefix discovery reuses vLLM-style physical page runs
without coupling logical cache structure to GPU objects.

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

A conservative fixed-capacity, uncompressed token trie now implements the
identity, longest-full-page lookup, active-reference, and deterministic
zero-reference eviction semantics. It deliberately does not claim the
compressed-radix performance target; scheduler integration, metrics, and
benchmark evidence remain open.

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

## Phase 5 — Graph capture and MoonTile kernel program

### Outcome

Measured hotspots move from reference/vendor execution to an AOT,
tile-oriented kernel catalog while preserving a stable engine architecture.

### Workstream 1: capability manifest

Define a signed/versioned manifest entry containing:

- operation and semantic version;
- GPU architecture range;
- dtype and accumulator;
- tensor/KV layout;
- shape class and alignment;
- workspace requirement;
- graph-capture safety;
- artifact digest;
- compiler/toolchain identity;
- correctness and benchmark evidence reference.

### Workstream 2: MoonTile IR

Implement the smallest IR needed for selected kernels:

- typed TensorView and address space;
- affine tile loops;
- copy and asynchronous copy;
- barriers and pipeline stages;
- MMA;
- reductions and elementwise expressions;
- compile-time layout/shape constraints.

Add validation, canonicalization, memory planning, vectorization, pipelining,
CUDA lowering, and deterministic serialization.

### Workstream 3: initial custom kernels

Prioritize by profiler evidence:

1. paged decode attention;
2. paged prefill attention;
3. fused QKV plus RoPE and KV write;
4. RMSNorm/residual fusion;
5. sampling reductions.

GEMM remains cuBLASLt/CUTLASS until evidence justifies replacement.

### Workstream 4: CUDA graphs

- capture declared shape classes during warm-up;
- maintain explicit eager fallback capability only when validated;
- no graph construction on live request data;
- report graph hit/miss and selected shape class;
- bound graph memory in the startup plan.

### Gate

- IR serialization and lowering are deterministic;
- invalid layouts fail before compilation;
- each custom kernel passes differential, boundary, sanitizer, and race tests;
- each replacement wins its declared microbenchmark shape set;
- end-to-end mixed workload does not regress;
- production mode executes no compiler or JIT path.

## Phase 6 — Sampling, usability, and operational hardening

### Outcome

LunaFlux reaches a vLLM-like operator experience for its deliberately smaller
supported capability set.

### Deliverables

- temperature, top-k, top-p, seed, stop tokens, and stop strings;
- deterministic RNG stream ownership per request;
- lunaflux run MODEL;
- lunaflux doctor;
- lunaflux plan MODEL with explainable automatic decisions;
- lunaflux bench MODEL;
- lunaflux inspect-kernels MODEL;
- concise versioned configuration;
- health, readiness, metrics, and structured diagnostics;
- startup memory/capacity report;
- install, upgrade, drain, rollback, and troubleshooting guides;
- OCI build with exact runtime dependencies and kernel manifest.

A bounded host sampler now implements temperature, top-k, top-p, a specified
deterministic RNG stream, strict finite-logit validation, and fixed scratch
storage.
Online/device execution integration, stop-string matching, whole-path native
allocation instrumentation, operator ergonomics, and production benchmark
evidence remain part of this phase.

### Gate

- common model starts with one command;
- every automatic choice has an explanation;
- unsupported options fail before readiness;
- fixed-seed sampling is replayable;
- CLI, native API, and OpenAI adapter use one canonical request vocabulary;
- cold-start and readiness measurements are published.

## Phase 7 — Tensor parallelism

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
- multi-device capacity and communication explanation.

Pipeline parallelism and cross-node execution remain excluded.

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

### Gate

- accuracy and task-quality budget is declared and passed;
- memory improvement is measured;
- unsupported hardware fails startup;
- existing BF16 and prior model-family benchmarks do not materially regress;
- scheduler and KV public APIs remain architecture-neutral.

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
- multimodal encoder/decoder graphs;
- prefill/decode disaggregation;
- cross-node tensor or pipeline parallelism;
- ROCm or other device backends;
- production runtime kernel JIT.

Deferral is deliberate. None may add hidden branches or placeholder flags
before its phase is approved.
