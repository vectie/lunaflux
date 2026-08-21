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
- A reusable startup-preallocated `LunaTokenizerWorker`. Its opaque
  generation-bound input writer copies one scalar byte into fixed backing per
  outer preparation unit, then transfers authority to work that advances byte
  BPE by a fixed operation budget, uses resumable CSR merge lookup, preserves
  exact ranked group-merge semantics, and copies results only in budget-capped
  chunks. The proportional `Bytes` encoder is a compatibility facade over that
  same writer and is not a reactor quantum. Frozen-reference and
  positive-controlled release-C gates cover byte-work equivalence and the
  allocation-free warm path. A fixed-lane Luna request-preparation pool
  includes the tokenizer input backing in checked aggregate byte admission and
  drives input copy, tokenization, token transfer, and incremental-output setup
  from one central FIFO quantum owner. Terminal tokenizer errors clean their
  already-stale work alias before the pool releases remaining lane authority.
  Canonical raw-frame scanning and validation are also cooperative;
  object-form request materialization, direct framed-view integration, trusted
  receipt-clock composition, and network ingress remain open.
- A bounded `tokenizer.json` adapter for the selected byte-level BPE contract,
  with duplicate-key rejection and explicit rejection of unsupported tokenizer
  semantics.
- A digest-pinned synthetic compatibility tokenizer for the selected 3,000-row
  reference model. It proves the supported ByteLevel-BPE path from `Luna*c` to
  exact token IDs, independently recorded logits, the greedy token, and a
  four-token continuation. This is functional integration evidence, not a
  claim of upstream SentencePiece compatibility or trained tokenizer quality.
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
- A substantive `lunaflux plan` command that authenticates a bounded model
  configuration under an approved root, builds the exact paged-Llama semantic
  plan, and explains shapes, operation count, workspace, required kernel
  capabilities, and KV page/capacity decisions. It closes filesystem authority
  before publishing. The model-content digest is explicitly caller-declared,
  not weight-verified, and kernel-manifest resolution remains open.
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
  import are implemented. Initial root-bound admission no-follow traverses each
  encoded absolute model/kernel label and requires exact opaque directory
  identity with the corresponding caller-owned capability before retained-pair
  activation or spawn. Failures remain role-specific and payload-safe. The
  root-bound supervisor retains that exact pair, executable, limits, and
  canonical source across zero-argument replacement without ambient label
  revalidation, closing the pair only at instance/service retirement. The child reconstructs
  admission and readiness, then runs a serialized bounded plan/completion loop.
- Synchronous model-configuration file admission now consumes a caller-owned
  approved root plus typed relative locator, closes the immutable bounded
  snapshot source before publication, verifies an independent lowercase
  SHA-256 file identity, and parses metadata under an independently supplied
  model content identity. It exposes no ambient path or async filesystem API.
- The offline reference artifact aggregate now follows the same capability
  boundary: one caller-owned approved root plus one opaque three-locator source
  yields three immutable digest-pinned snapshots synchronously. Same-handle
  per-file/aggregate admission and deterministic file close complete before
  parsing or bundle publication. The CLI alone opens the absolute deployment
  root and closes it before reference execution or output.
- The internal device-worker bootstrap composition now authenticates a decoded
  source before fixed-role root acquisition, derives the bounded paged-Llama
  recipe, closes kernel authority before device preparation and model authority
  before exact readiness publication, and retains independent retryable worker
  and root cleanup authority after compound failure. Transport publication
  remains outside this owner.
- The device child consumes Configure then canonical source from
  the inherited private channel, reconstructs the fixed-root bootstrap into a
  live `DeviceWorkerOwner`, and writes Ready only after an exact readiness
  query and reusable loop storage.
  It then reads one bounded plan, acquires its completion writer before device
  mutation, finishes execution, and publishes the exact completion before the
  next plan. Clean idle waits without a first-byte deadline; partial frames are
  bounded and fail-stop. CPU ordering, positive-controlled release generated-C
  allocation, invalid-config no-Ready, native child-control ABI, and
  AddressSanitizer gates pass; physical CUDA readiness remains deferred.
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
  the aggregate owner after executor finish. A positive-controlled native
  release harness now proves a warm public device-worker owner completes 131
  exact canonical one-row ordinary-prefill/final-prefill/decode and
  greedy/stochastic execute/authenticate/retire/reset cycles without MoonBit
  managed/array/string allocation or native resource creation, with exact fake
  transfer/launch/synchronize/readback counts and balanced page/native close.
  Fake launch/readback/non-finite faults also prove no completion publication,
  owner faulting, writer reuse, and cleanup. Mixed-row/full-batch execution,
  independent stochastic correctness, physical CUDA numerical correctness,
  and promotion evidence remain open.
- A canonical transport-independent inference request and streaming-event
  contract with immutable token/text inputs, exact model identity, bounded
  sampling/stops/deadlines/cache scope, monotonic usage, and payload-safe public
  failure vocabulary. The first trusted receipt boundary can now capture the
  relative deadline exactly once as an opaque absolute monotonic deadline;
  scheduler admission never rebases it after parsing or tokenization delay.
- A canonical fixed-buffer request-v1/event-v2 wire for every `GenerateRequest`
  field and all five `StreamEvent` variants. Completed v2 carries an optional
  bounded UTF-8 terminal tail for unmatched incremental stop prefixes. It
  bounds hostile counts before arithmetic
  or allocation, validates exact lengths/checksums/reserved bytes and canonical
  option encodings, and publishes only epoch-bound opaque frames. A reusable
  `LunaFramedRequestWorkspace` accepts and validates request-v1 through bounded
  work quanta, then lends an epoch-bound scalar/indexed-byte view without
  constructing request payload objects. Its
  fixed-capacity one-frame incremental reader authenticates the 16-byte prefix
  and unsigned declared total before payload acceptance, rejects trailing or
  pipelined input, poisons on prefix/frame failure, and delegates final
  authority to the same scanner plus a synchronous object-form compatibility
  materializer. It is a transport-neutral codec foundation, not a listener or
  ingress claim.
- A bounded incremental decoded-output owner that copies exact tokenizer pieces
  into caller-owned fixed storage, validates UTF-8 across token boundaries,
  and withholds stop strings matched across token or code-point boundaries.
  Pattern tables and scratch are constructed before token stepping. The
  persistent Luna online instance drives this owner through acknowledged typed
  Luna events. `take_event` issues one opaque exact request/event-epoch credit;
  only its ACK advances the semantic owner. Usage ACK atomically creates a
  distinct Completed/Failed epoch, abort invalidates outstanding credit, and a
  canonical framed adapter borrows only the semantic view. Positive-controlled
  release-C gates cover the warm online and adapter paths. The fixed-lane
  preparation pool now feeds this boundary; listener dispatch and network
  transport remain open.
- A startup-preallocated, generation-authenticated
  `LunaRequestSemanticStorage` foundation can now stream structurally bounded
  stop-token, stop-string, cache-scope, permission, and inference-limit data
  into budgeted validation and a sorted stop-token index. Its scalar/byte View
  exposes no collection, string, epoch, or release authority. Incremental
  output is the first real consumer: it copies one authenticated stop byte per
  charged setup unit, retains explicit failed-Work cleanup, and detaches only
  after final View authentication at lease transfer. Its preallocated one-View
  slot is included in request-preparation pool reference-cell admission. The
  coordinated object-request path now streams stop tokens, UTF-16 stop text,
  cache scope, and permission into this storage under the pool quantum. Output
  setup receives only the full semantic View; the scheduler receives a narrow
  token-only projection and holds an independent preallocated retention until
  slot recycle. The online worker holds the second retention through healthy
  request retirement or successful recovery close, so claim release cannot
  invalidate scheduler semantics early. Builder-structural failures precede
  Work and do not claim full multi-error equivalence with
  `StopConditions::new`.
- A request-admission bridge now composes that reader with trusted
  monotonic time: the first valid nonempty append samples exactly once before
  byte mutation, and one successful take retains an immutable request plus its
  once-derived absolute deadline without exposing timestamp/deadline getters or
  a detachable frame/receipt capability. The live fixed-lane preparation pool
  authenticates the selected model, tokenizer digest, and exact inference
  envelope; advances retained UTF-8 tokenization, token copying, and
  incremental-output setup through bounded FIFO quanta; and returns a
  revocable `LunaPreparedRequest` shell around one exact claim. Each lane's
  token, semantic, and output storage remains generation-pinned through claim
  release after lower retirement. Saturated or draining admission does not
  consume the receipt; deadlines, cancellation, work ceilings, stale aliases,
  and explicit discard/release are covered. Ready, failed, prepared, and
  claimed results intentionally pin their lane until the outer owner disposes
  of the capability. The legacy facade performs the same semantic-storage and
  View-based output contract synchronously, including blocking tokenization and
  semantic validation; it is not a reactor quantum. Canonical-frame scanning
  and validation is cooperative, while direct framed-view preparation and
  network ingress orchestration remain open.
- A backend-neutral worker protocol for exact prefill/decode rows, flattened
  token/page/capability tables, plan and model generations, completion slots,
  and typed completion records. Its reusable fixed-capacity plan and completion
  buffers authenticate lifecycle epochs; scalar row drafts avoid temporary
  table construction; final prefill explicitly returns the first generated
  token; provenance-bound row capability recipes preserve exact model-plan
  operation order; and whole-build checkpoints can roll back several committed
  rows. Submitted-work validation remains separate from the scheduler's
  current-generation publication check. The scheduler owns distinct paired A/B
  plan and completion buffers, while the production root-bound service admits
  one outstanding plan for its serialized child. It issues exact-epoch completion writers and
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
  exact I/O with an unbounded idle first-byte wait and bounded partial-frame
  deadlines, bounded wait/terminate, and deterministic kill/reap close. A legacy
  protocol supervisor owns distinct A/B plan and
  completion frame buffers, enforces monotonic submission and oldest-first
  receive, pins a validated response until explicit post-publication
  retirement, and permanently fails malformed sessions. The production worker
  channel now has a separately linked child-side frame runtime and gates
  covering three monotonic echo exchanges, A/B/A reuse, retained
  response inspection, EOF, zero exit, and reap. That legacy echo child
  performs an exact checksummed Configure/BootstrapSource/Ready handshake that binds model
  identity, an admitted-bootstrap SHA-256 derived from graph/artifact evidence,
  a bootstrap-source SHA-256 derived from canonical `EncodedBootstrapSource`
  bytes, model generation,
  exact process-visible device ordinal, predecessor, worker limits, and
  inference limits. It canonically decodes and checks the source before the
  fixture publishes protocol readiness. Startup
  cleanup retains explicit authority after a double failure, and submission
  rejects a foreign model generation before transport mutation. The echo child
  returns deterministic protocol completions rather than opening CUDA; the
  separate device child owns the real serialized execution loop. Closed-child
  recovery now retains validated responses, retires exact unreturned
  submissions in sequence order, and derives a replacement predecessor only
  after all obligations are discharged. The real-process gate proves a clean
  replacement, an abandoned sequence 4, and successful continuation at
  sequence 5. A new thread-confined worker service encapsulates scheduler and
  root-bound process owners, checks its single production credit before
  scheduler mutation, records plans before transport, validates received
  retirement authority before scheduler mutation, and stages each frame once
  into a paired typed completion owner. Publication pressure retains a private
  accepted-flight state and returns a nonallocating scalar backpressure result;
  retry neither rereads the wire frame nor reopens or advances the completion
  owner. The service commits bounded worker failures before abandonment and
  replaces the child only
  after the outstanding obligation retires and every surviving active request
  is terminally invalidated with balanced host KV release. Waiting requests are
  preserved because they own no device bytes. An independent retained
  `WorkerServiceBinding` pins the expected bootstrap and bootstrap-source digests, device ordinal,
  and inference envelope; each join/restart also verifies exact scheduler-owned
  model identity, generation, predecessor, and worker-protocol limits. Its
  owned-instance factory now consumes an immutable scheduler blueprint and
  ordinary approved roots, constructs both mutable owners internally, and
  publishes a one-shot ready-or-cleanup shell. The alias-taking constructor is
  fixture-only. Worker buffers, child ownership, executable snapshot, and
  Configure/source/expected-Ready frames are preallocated before root
  acquisition; native spawn, scalar handshake validation and cleanup, and
  owner publication allocate no managed objects with rooted authority live.
  The owned service now also makes a permanent Raw-versus-Online family
  choice. Its restricted epoch lease authenticates sequential streaming
  request epochs, exact generation/position/publication order, monotonic
  admission/deadline
  time, cancellation cuts, worker-loss recovery, and deterministic close
  without exposing scheduler/process/handle owners. The alias-free
  `LunaOnlineInstance` prepares that worker once, consumes one externally
  prepared `LunaPreparedRequest` claim at a time under a fresh opaque ticket,
  publishes one canonical Accepted credit, and provides exact request
  retirement plus off-reactor abort/recovery/instance-shutdown paths. Healthy retirement
  preserves the worker, lease epoch, publication cursor, and plan history;
  worker/protocol/device failure remains close-only. The instance owns no
  tokenizer or raw receipt state; Busy and Draining leave prepared authority
  retryable, exact preparation mismatches fail before lower mutation, and
  destructive claim transfer rejects retained-alias replay.
  The borrowed ordinary roots remain caller-owned. Normal allocation-free
  Token output and natural Maximum/StopToken Usage+Completed-v2 bundles are
  implemented; stop tokens are counted but suppressed. Incremental string-stop
  matching now withholds stop/post-stop bytes, commits an exact cancellation
  cut when possible, and translates only its authenticated terminal to
  Usage+Completed(StopSequence); final-token natural-terminal precedence is
  covered without cancelling an already-terminal request. Caller cancellation
  now defers behind pinned credit and publishes Usage+Completed(Cancelled), and
  automatic owner-clock expiry publishes Usage plus canonical nonretryable
  Failed(deadline_exceeded), with natural terminal precedence. Decoder/output
  rejection and authenticated worker/device loss now retain exact terminal
  authority and publish Usage plus fixed payload-safe Failed(output_invalid,
  nonretryable) or Failed(worker_unavailable, retryable). Reactor-safe progress
  reports when explicit off-reactor recovery/reap/drain is required. Ingress
  transport remains open;
  this is not online-serving readiness.
  Its
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
  retains authority after compound preparation/cleanup failure. The production
  child imports its fixed approved roots, reconstructs and authenticates the
  encoded source through `engine/device_worker_bootstrap`, derives `Ready` from
  the live `DeviceWorkerOwner`, and forwards bounded plan/completion frames
  through that owner. The real-child gate currently proves hostile startup and
  failure behavior; positive spawned-child execution with physical CUDA and
  numerical output remains unproven.
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
  It validates model identity, duplicate IDs, context/page envelopes, exact
  receipt-relative deadlines, and recipe provenance; the core request type
  still rejects nonempty stop strings until the incremental owner is composed;
  cancellation and deadline sweeps preflight publication and identity capacity
  before mutation. Its bounded `build_next` path activates FIFO requests only
  after submission, reserves decode resources before prefill policy, emits
  prefill rows before decode rows, and uses exact plan, block-table, and page
  checkpoints for rejected-build rollback. Intermediate prefill, final prefill,
  and decode use separate authenticated capability recipes. Distinct A/B plan
  owners permit two outstanding submissions in scheduler and echo fixtures;
  the production root-bound service restricts the serialized device child to
  one. Paired completion owners use a
  fixed slot index and exclusive writers; full-batch retirement authenticates
  current versus cancelled/expired work, preflights both publication rings and
  KV release, publishes final-prefill/decode tokens in plan order, enforces
  token-stop/maximum-output terminals, and resets the exact plan side. A shared
  monotonic publication sequence makes the separate token and terminal rings
  fail closed against out-of-order dequeue, including cancellation cuts. A
  single owner-selected dequeue now returns the exact globally oldest typed
  publication without exposing a ring choice. Normal
  idle and two-owner pressure are flat allocation-free outcomes in generated
  native code; the positive-controlled warmed scheduler allocation gate passes.
- A bounded `lunaflux reference` command that opens one approved absolute
  deployment root, validates three strict relative locators and digests, loads
  an admitted host snapshot, closes the root, and produces offline greedy
  tokens.
- A reusable depth-bounded JSON duplicate-key guard for map-backed parsers.
- A repository guard for package direction, sibling-product independence,
  CUDA ABI ownership, Python exclusion, temporary debt markers, and source-size
  review thresholds.

## Explicitly not implemented or not yet proven

- The typed LunaTile IR and its offline specialization system are Phase 5
  architecture, not current implementation. Existing model, device, execution,
  launch-contract, catalog, and artifact-admission types establish inputs to
  that future boundary, but LunaFlux does not yet partially evaluate them into
  typed specialization records or generated kernel families. No current
  artifact therefore claims baked model/layout/codebook constants, a
  specialization compiler, or specialization-specific differential and
  end-to-end benchmark evidence.
- The tokenizer shipped with the pinned upstream model uses SentencePiece
  normalization, template processing, unknown-token fusion, and byte fallback,
  and is correctly rejected rather than approximated by the selected
  ByteLevel-BPE contract. The selected ByteLevel subset is independently corpus
  validated, and the checked-in synthetic compatibility fixture now proves its
  text-to-model composition against this model. Production use still requires
  an independently approved tokenizer with suitable trained semantics; the
  synthetic fixture is not such an artifact.
- The streaming device loader now uses pinned approved-root and regular-file
  capabilities rather than ambient string paths. Deployment approval and
  read-only mount policy remain external trust inputs. Low-level fixed
  descriptor-role inheritance, production supervisor retention across restart,
  and child reconstruction from those roles are implemented and proven.
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
  sanitizer, leak, soak, or benchmark gates. Generated logits and sampled
  completions are structurally wired through the spawned child and worker
  service, but that path has not passed physical-CUDA numerical validation. The
  stateless reference interpreter remains the correctness oracle and
  deliberately recomputes full sequences.
- Public network ingress/API adapters, continuous batching with global
  fairness/preemption, live worker overlap, physically proven paged execution,
  prefix integration, or telemetry. The public in-process `LunaOnlineInstance`
  retains one worker across sequential healthy requests and implements
  generated-text decoding, sampling-result consumption, exact
  one-credit Token/Usage/Completed/Failed publication, cancellation, deadlines,
  request retirement, explicit drain, and close-only failure terminalization.
  The
  scheduler registry/lifecycle, worker, sampling, page-allocation, block-table,
  prefix-index, and runtime-capacity foundations exist, and the scheduler,
  root-bound service, spawned device child, and device-worker aggregate are now
  connected. There is still no network listener/protocol adapter or
  physical-CUDA serving evidence, so no bounded-latency serving claim is made.

The `lunaflux doctor` command reports the semantic CUDA inventory, bounded
reference loading, and offline executor status. `lunaflux plan` now
authenticates model configuration and reports the derived semantic plan,
required capabilities, and exact KV-capacity decision without materializing
weights or resolving a kernel manifest. `lunaflux reference` reads
digest-pinned files and runs the correctness executor. All retain production
readiness as false.

## Next correctness gate

Phase 1 promotion still requires `doctor` evidence for admitted model files and
the resolved kernel manifest, weight-verified extension of the current semantic
`plan` report, and a physical-CUDA runner proving
transfers, module/function launches, BF16 GEMM and AOT numerics, balanced
repeated load/run/unload, concurrent resource stress, and the complete
sanitizer and leak gates. Physical concurrency/race evidence also remains open.
Physical device-KV correctness and true cached decode evidence remain open.
Performance and production-readiness claims remain out of scope until physical
correctness, resource balance, soak, and benchmark evidence passes.
