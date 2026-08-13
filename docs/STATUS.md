# LunaFlux implementation status

This file records implemented behavior, not intended architecture. A phase is
not complete until every gate in [PLAN.md](PLAN.md) passes.

## Completed foundation

- Phase 0 product, architecture, lifecycle, benchmark, and debt contracts.
- Checked engine and request lifecycle vocabulary.
- Focused validated configuration records and deterministic startup decisions.
- A byte-bounded configuration JSON adapter that rejects duplicate, unknown,
  missing, mistyped, and out-of-policy fields before constructing focused
  records.
- A bounded extracted byte-BPE tokenizer foundation with exact byte decoding,
  explicit special-token policy, limits, and deterministic merge ranking.
- A bounded `tokenizer.json` adapter for the selected byte-level BPE contract,
  with duplicate-key rejection and explicit rejection of unsupported tokenizer
  semantics.
- Validated dense Llama-style BF16 dimensions and content identities, with the
  plan identity derived canonically from the complete validated semantic spec
  rather than trusted as caller input.
- A bounded selected-model architecture JSON adapter that rejects remote code,
  tied embeddings, and unsupported dense-Llama variants.
- Bounded decoded safetensors metadata validation, including checked tensor
  sizes, ranges, duplicates, overlaps, precision, and materialization budget.
- A bounded safetensors byte reader with checked little-endian header lengths,
  strict JSON schema, exact integer conversion, duplicate-key rejection, and
  complete payload accounting.
- An immutable architecture-neutral operation plan and dense Llama family
  builder with explicit shape, workspace, RoPE, activation, tensor-reference,
  and semantic kernel-capability contracts.
- Exact dense-Llama weight-manifest binding that validates safetensors names,
  shapes, completeness, and plan tensor-reference order before allocation.
- A single selected-Llama admission boundary that proves configuration,
  safetensors metadata, semantic plan, and exact tensor vocabulary agree before
  materialization.
- Bounded reference-bundle file admission for three explicit regular files,
  with per-file and aggregate pre-allocation limits, exact-snapshot SHA-256,
  tokenizer/model vocabulary agreement, payload-safe errors, and deterministic
  handle closure. Whole-file ownership is explicitly reference-only.
- Two-pass bounded weight materialization with identity preservation, complete
  preflight validation, zero-copy tensor visitation, exact complete-file
  source offsets, and an explicit host-copy reference representation.
- Deterministic aligned single-arena device-weight layouts plus a synchronous,
  bounded approved-root loader. Component-wise no-follow traversal pins a
  strict relative descendant; complete digest, metadata, and dense-Llama
  vocabulary admission precede device allocation. Inspection privately retains
  that locator, and later loading completely re-admits it before allocation.
  Transfer reuses fixed host storage, copies source-ordered chunks into final
  device regions, re-hashes exact copied bytes, rechecks the same-handle stamp,
  and retains no model-sized snapshot. A terminal source-file close failure
  closes any otherwise-ready allocation before publication. If that allocation
  close also fails, opaque cleanup authority is returned at `SourceClose` with
  idempotent retry rather than losing the partial owner.
- Deterministic Float reference kernels for dense-Llama embedding, RMSNorm,
  projections, split-half RoPE, causal grouped-query attention, residuals,
  SiLU-gated MLP, and the language-model head.
- An architecture-neutral offline plan interpreter that prepares immutable BF16
  weights once, executes validated operation plans, and performs bounded greedy
  generation. Pinned tiny-model fixtures cover non-degenerate attention and MLP
  paths with independently derived logits.
- An exact Apache-2.0 upstream tiny BF16 Llama artifact pinned by immutable
  revision and SHA-256. End-to-end tests match 33 independently generated
  sampled logits, three argmax tokens, and two four-token greedy continuations.
- A compatible Apache-2.0 external ByteLevel-BPE corpus derived from pinned
  Hugging Face Tokenizers sources. Five independently generated cases cover
  ranked merges, spaces, multibyte Unicode, a literal zero byte, newlines, and
  byte-exact decoding.
- Bounded deterministic greedy sampling with stable lowest-token tie breaking,
  plus temperature/top-k/top-p stochastic sampling over fixed scratch storage.
  The stochastic path rejects non-finite logits before advancing its specified
  xorshift64 stream and preserves deterministic lowest-token tie ordering.
  Online/device integration, whole-path allocation instrumentation, and
  production performance evidence remain open.
- A private dynamically loaded CUDA Driver/cuBLASLt ABI with opaque handles,
  explicit retryable resource release, parent-child ownership, and a semantic
  public device inventory. Unsupported hosts report a typed unavailability
  reason without attempting device allocation. Opened contexts pin their
  ordinal and discovered capability.
- Checked offset host/device transfers and reusable row-major BF16 cuBLASLt
  GEMM plans with FP32 accumulation, explicit caller-owned storage, retryable
  descriptor cleanup, and injected-dispatch ownership/failure probes.
- Atomic native operation guards that interlock use, child construction, and
  explicit close across contexts and every owned child resource. Phase 1 GEMM
  synchronizes before returning so borrowed operands cannot outlive device
  work.
- Private bounded CUDA module and function loading plus a synchronous AOT
  launch seam. Launch dimensions, argument count, allocation ownership,
  offsets, byte counts, alignments, context identity, and close ordering are
  checked before or around the driver call.
- An immutable exact startup kernel catalog that selects by model operation,
  shape, capability, CUDA target, and workspace, using either the narrow
  implicit vendor BF16 GEMM semantic or a content-addressed AOT artifact.
  Missing, incompatible, or ambiguous support fails closed without JIT or
  fallback. Content-addressed bindings distinguish a module digest, semantic
  kernel family, and profile-specific entry-point identity. The vendor GEMM
  contract is limited to operation layouts representable by its narrow ABI.
- An immutable static device plan that proves the semantic plan, exact weight
  layout, and resolved kernel catalog share identity and target, then resolves
  every operation input, output, workspace, and kernel binding before
  execution preparation.
- A deterministic exact BF16 activation/workspace memory plan with liveness-
  checked slot reuse, retained terminal output, bounded aligned arena size,
  one startup allocation, and explicit close.
- Exact stateless device execution profiles for `FullPrefill` and
  `FullRecompute`. They precompute token-staging, weight, activation, workspace,
  operation, complete terminal-output, and final-row views for an exact batch
  and sequence shape without claiming cached state.
- Bounded AOT launch-contract admission for every exact profile and AOT-backed
  operation. Contracts pin exact dimensions, ordered semantic operand roles,
  byte/alignment claims, workspace, family, and entry point; they cannot carry
  arbitrary payloads, paths, or a JIT channel.
- Content-addressed artifact admission that verifies and owns each required
  module once per digest and maps all required stable entry points to bounded
  CUDA function symbols. Missing, duplicate, unreferenced, or digest-mismatched
  declarations fail before module import.
- Bounded, versioned AOT manifest and module-file admission under an approved
  read-only root. It pins the exact model, target, catalog, module digests,
  stable entry points, and function symbols; rejects unsafe path forms and
  unsupported file types; proves aggregate bounds before module-sized reads;
  and admits immutable snapshots without an extra full copy.
- Opaque approved-filesystem capabilities backed by a private native ABI.
  Component-relative `openat`, no-follow directory traversal, final regular-
  file checks, pinned positional reads, same-handle stamps, atomic operation/
  close exclusion, deterministic close, and payload-safe failures are
  implemented. A startup-only bounded snapshot holds one lease across its two
  stamps, exact positional read, and trailing-growth probe, with exactly one
  accepted payload allocation. It rejects truncation or size/mtime/ctime change
  before publication. Native and public capability representations are
  private. The streaming weight loader has adopted this authority. A reusable
  ordered model/kernel lease pair, sanitized fixed-FD rooted spawn, and child
  import are implemented. The root-bound supervisor now retains the exact pair,
  executable, limits, and canonical source across zero-argument replacement,
  closing the pair only at instance/service retirement. The startup-only child
  reconstructs admission and readiness; its steady-state production execution
  loop remains open.
- Synchronous model-configuration file admission now consumes a caller-owned
  approved root plus typed relative locator, closes the immutable bounded
  snapshot source before publication, verifies an independent lowercase
  SHA-256 file identity, and parses metadata under an independently supplied
  model content identity. It exposes no ambient path or async filesystem API.
- The internal device-worker bootstrap composition now authenticates a decoded
  source before fixed-role root acquisition, derives the bounded paged-Llama
  recipe, closes kernel authority before device preparation and model authority
  before exact readiness publication, and retains independent retryable worker
  and root cleanup authority after compound failure. Transport publication
  remains outside this owner.
- The startup-only device child consumes Configure then canonical source from
  the inherited private channel, calls the real bootstrap composition, and
  writes Ready only after an exact readiness query. It accepts only clean EOF
  afterward and deterministically closes the owner; decode/bootstrap/channel
  failures exit silently nonzero. CPU ordering, invalid-config no-Ready, native
  child-control ABI, and AddressSanitizer gates pass; physical CUDA readiness
  and the steady-state execution loop remain deferred.
- An exact prepared Phase-1 device executor that cross-checks model, target,
  catalog, profile, artifact, allocation ownership, physical range, and actual
  pointer-alignment evidence before importing code. It reuses token and
  activation storage, launch arguments, loaded functions, and vendor
  descriptors across synchronous full-sequence dispatches. Invalid tokens and
  unavailable or stale outputs fail closed.
- Explicit executor cleanup ownership across partial construction and partial
  close. A double construction/cleanup failure returns an opaque retryable
  cleanup guard; a close failure blocks further dispatch while preserving
  retry authority over still-open children.
- A distinct versioned paged model graph whose every operation consumes exact
  live-step counts, with positioned rotary input, page-table descriptors, and
  persistent layer-indexed split K/V state. Its catalog is all-AOT and binds
  exact runtime-input order, operation shape, layout version, tokens per page,
  entry point, dimensions, and physical operands before code is imported.
- Fixed-capacity device-step staging uploads counts last after complete host
  validation, authenticates scheduler-retained page generations before plan
  submission, and permanently poisons a staged owner after any partial transfer
  failure. A positive-controlled native release gate proves the warmed public
  descriptor `stage`/`finish` path performs no generated/runtime allocation.
- An owner-mediated synchronous full-paged-graph executor that leases the
  caller-owned weight allocation and privately owns descriptor buffers,
  activation/workspace and persistent-KV allocations, stream, modules,
  functions, dimensions, and argument lists. Exact staged/executed capabilities
  enforce lifecycle order; any partial launch failure poisons the executor, and
  partial construction or close retains explicit retryable cleanup authority.
- Owner-mediated BF16 terminal-logit readback and completion construction for
  that executor. Startup binds the retained LM-head row geometry and allocates
  fixed readback/logit/selection scratch; successful execution reads only
  producing rows, rejects non-finite logits, applies greedy or
  counter-addressed stochastic selection from exact request seed/output index,
  and appends to the exact-plan completion writer while leaving publication to
  the aggregate owner after executor finish. Physical CUDA
  numerical correctness and a positive-controlled full lifecycle allocation
  gate remain open.
- A canonical transport-independent inference request and streaming-event
  contract with immutable token/text inputs, exact model identity, bounded
  sampling/stops/deadlines/cache scope, monotonic usage, and payload-safe public
  failure vocabulary.
- A backend-neutral worker protocol for exact prefill/decode rows, flattened
  token/page/capability tables, plan and model generations, completion slots,
  and typed completion records. Its reusable fixed-capacity plan and completion
  buffers authenticate lifecycle epochs; scalar row drafts avoid temporary
  table construction; final prefill explicitly returns the first generated
  token; provenance-bound row capability recipes preserve exact model-plan
  operation order; and whole-build checkpoints can roll back several committed
  rows. Submitted-work validation remains separate from the scheduler's
  current-generation publication check. The scheduler owns distinct paired A/B
  plan and completion buffers, issues exact-epoch completion writers, and
  retires completed owners strictly in plan-sequence order. Canonical bounded
  little-endian plan and completion frames now detach that protocol from heap
  owner capabilities: each receive validates checksum, exact counts/ranges,
  identities, sampling fields, outcomes, and canonical table coverage before
  replacing an epoch. Startup scratch provides O(n log n) uniqueness checks
  and O(1) completion-slot lookup without steady-state allocation. The service
  authenticates each received completion frame against the retained exact plan
  and converts it into the paired scheduler completion owner without retiring
  work, preserving publication-backpressure retry. An exclusive worker-side
  writer now produces canonical completion frames directly from exact received
  plan rows with abort/stale-epoch safety, so no scheduler completion owner is
  needed across this boundary. Reusable device-step staging also consumes the
  validated plan frame directly without a scheduler plan owner in the worker.
  Its post-execution path binds logits to the exact retained frame owner and
  epoch, applies validated scalar sampling fields, and freezes a canonical
  completion frame directly; no scheduler plan or completion owner is
  reconstructed in the isolated side.
  The private native layer now provides shell-free exact-executable spawn, an
  inherited socketpair endpoint authenticated by construction, fixed-buffer
  exact I/O with monotonic deadlines, bounded wait/terminate, and deterministic
  kill/reap close. A protocol-aware supervisor now owns distinct A/B plan and
  completion frame buffers, enforces monotonic submission and oldest-first
  receive, pins a validated response until explicit post-publication
  retirement, and permanently fails malformed sessions. The production worker
  channel now has a separately linked child-side frame runtime and gates
  covering three monotonic exchanges, A/B/A reuse, retained
  response inspection, EOF, zero exit, and reap. That child intentionally
  performs an exact checksummed Configure/BootstrapSource/Ready handshake that binds model
  identity, an admitted-bootstrap SHA-256 derived from graph/artifact evidence,
  a bootstrap-source SHA-256 derived from canonical `EncodedBootstrapSource`
  bytes, model generation,
  exact process-visible device ordinal, predecessor, worker limits, and
  inference limits. It canonically decodes and checks the source before the
  supervisor publishes protocol readiness. Startup
  cleanup retains explicit authority after a double failure, and submission
  rejects a foreign model generation before transport mutation. The child intentionally returns
  deterministic protocol completions rather than opening CUDA. Closed-child
  recovery now retains validated responses, retires exact unreturned
  submissions in sequence order, and derives a replacement predecessor only
  after all obligations are discharged. The real-process gate proves a clean
  replacement, an abandoned sequence 4, and successful continuation at
  sequence 5. A new thread-confined worker service encapsulates scheduler and
  process owners, records scheduler plans before transport, retries pinned
  completion frames without reopening completion epochs, commits retryable
  bounded worker failures before abandonment, and replaces the child only
  after both scalar flight obligations retire. An independent retained
  `WorkerServiceBinding` pins the expected bootstrap and bootstrap-source digests, device ordinal,
  and inference envelope; each join/restart also verifies exact scheduler-owned
  model identity, generation, predecessor, and worker-protocol limits. Its
  real-process gate proves output-publication backpressure, worker death, failure retirement,
  non-reusing restart, post-restart completion, and balanced KV ownership.
  The full-graph physical blueprint and artifact bundle now derive the
  admitted-bootstrap digest
  from bounded canonical module, symbol, launch, layout, operand, device-step
  envelope, and exact device-assignment evidence.
- An aggregate `engine/device_worker` readiness owner. Its inert admission
  retains the independently expected startup contract and exact bounded model
  path, weight, device, memory, kernel, artifact, and bootstrap evidence.
  Preparation compares the complete received contract before resource opening,
  opens and verifies the exact visible device ordinal/target, loads verified
  weights internally, prepares the complete paged executor, and exposes only
  the exact readiness contract while context, weights, and executor remain
  live. Executor/weights/context cleanup is dependency ordered, retryable, and
  retains authority after compound preparation/cleanup failure. This is the
  readiness-owner foundation, not child-process integration: the current child
  has no model/source locator delivery, does not call this owner or forward
  execution through it, and does not emit `Ready` from it.
- A generational fixed-page KV metadata `PageAllocator` with preallocated
  arrays, an intrusive FIFO free queue, separate active and cached references,
  exact-run rollback, terminal-generation retirement, invariant diagnostics,
  and a randomized ownership model test.
- A fixed-capacity request `BlockTableArena` with generational table ownership,
  preallocated dense page mappings, reusable rollback/release buffers, and
  randomized ownership fixtures. Canonical inline `PageIdStorage` and
  `BlockTableIdStorage` retain optional identities without boxed option cells;
  they do not acquire or release the underlying resources.
- A fixed-capacity logical token-prefix trie with full-page-only longest-prefix
  reuse, complete cache identity salting, generational entry IDs, transactional
  result buffers, active references, deterministic zero-reference eviction,
  and explicit bounded linear-scan complexity. It never owns device memory or
  mutates physical page ownership.
- Focused scheduler and cache configuration records for token/request budgets,
  chunked prefill, waiting-age policy, the two-plan-buffer invariant, bounded
  generated-token publication, physical page/block-table limits, prefix
  metadata arenas, reference bounds, and cache layout identity. Startup-only
  runtime capacity resolution now checks the scheduler, cache, model-shape, and
  worker envelopes and materializes canonical page-allocator and block-table
  limits. The scheduler subsequently authenticates exact intermediate-prefill,
  final-prefill, and decode row recipes against the loaded model generation and
  capability-cell capacity.
- A deterministic single-owner scheduler foundation with fixed request slots,
  an intrusive FIFO waiting queue, globally unique request generations, bounded
  terminal notices, and tokenized admission against the resolved runtime plan.
  It validates model identity, duplicate IDs, context/page envelopes, deadlines,
  and recipe provenance, and it explicitly rejects nonempty stop strings;
  cancellation and deadline sweeps preflight publication and identity capacity
  before mutation. Its bounded `build_next` path activates FIFO requests only
  after submission, reserves decode resources before prefill policy, emits
  prefill rows before decode rows, and uses exact plan, block-table, and page
  checkpoints for rejected-build rollback. Intermediate prefill, final prefill,
  and decode use separate authenticated capability recipes. Distinct A/B plan
  owners permit two outstanding submissions. Paired completion owners use a
  fixed slot index and exclusive writers; full-batch retirement authenticates
  current versus cancelled/expired work, preflights both publication rings and
  KV release, publishes final-prefill/decode tokens in plan order, enforces
  token-stop/maximum-output terminals, and resets the exact plan side. Normal
  idle and two-owner pressure are flat allocation-free outcomes in generated
  native code; the positive-controlled warmed scheduler allocation gate passes.
- A bounded `lunaflux reference` command that validates explicit paths and
  digests, loads an admitted host snapshot, and produces offline greedy tokens.
- A reusable depth-bounded JSON duplicate-key guard for map-backed parsers.
- A repository guard for package direction, sibling-product independence,
  CUDA ABI ownership, Python exclusion, temporary debt markers, and source-size
  review thresholds.

## Explicitly not implemented or not yet proven

- The tokenizer shipped with the pinned upstream model uses SentencePiece
  normalization, template processing, unknown-token fusion, and byte fallback,
  and is correctly rejected rather than approximated by the selected
  ByteLevel-BPE contract. The selected ByteLevel subset is independently corpus
  validated, but this particular model must be driven by token IDs or a
  separately approved compatible tokenizer artifact.
- The streaming device loader now uses pinned approved-root and regular-file
  capabilities rather than ambient string paths. Deployment approval and
  read-only mount policy remain external trust inputs. Low-level fixed
  descriptor-role inheritance is implemented and proven; production supervisor
  retention and child reconstruction from those roles remain open.
- Physical-CUDA validation of driver loading, transfers, modules, function
  launches, events, numerical BF16 GEMM and AOT correctness, implicit-heuristic
  shape support, and repeated resource balance. Local C stubs pass a manually
  instrumented ASan/UBSan run, but the MoonBit runtime was not instrumented and
  macOS leak detection was unavailable, so the full sanitizer/leak gate remains
  open.
- Physical and numerical proof for the new paged path. The semantic graph,
  host page/table ownership, reusable device descriptor, persistent device KV
  allocation, exact all-AOT launch ABI, artifact admission, physical blueprint,
  synchronous owner-mediated full-graph dispatch, and aggregate readiness owner
  are implemented. No
  production paged-kernel bundle has yet passed real-CUDA model correctness,
  sanitizer, leak, soak, or benchmark gates; generated logits are not yet wired
  through the spawned child into the online service path. The stateless reference interpreter
  remains the correctness oracle and deliberately recomputes full sequences.
- Online APIs, generated-text decoding/transport publication, continuous
  batching with global fairness/preemption, live worker overlap,
  physically proven paged execution, online sampling and prefix integration,
  or telemetry. The
  scheduler registry/lifecycle, worker, sampling, page-allocation, block-table,
  prefix-index, and runtime-capacity foundations exist but are not an online
  service. No end-to-end child-to-device execution or bounded-waiting claim is
  made; the device-worker aggregate has not yet been wired into the spawned
  child or service dispatch path.

The `lunaflux doctor` command reports the semantic CUDA inventory, bounded
reference loading, and offline executor status. `lunaflux plan` remains
configuration-only, while `lunaflux reference` reads digest-pinned files and
runs the correctness executor. All retain production readiness as false.

## Next correctness gate

The remaining Phase 1 promotion evidence is a physical-CUDA runner proving
transfers, module/function launches, BF16 GEMM and AOT numerics, balanced
repeated load/run/unload, concurrent resource stress, and the complete
sanitizer and leak gates, with physical concurrency/race evidence still open.
Device KV ownership and true cached decode remain open. Performance and
production-readiness claims remain out of
scope until physical correctness, resource balance, soak, and benchmark
evidence passes.
