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
- Deterministic aligned single-arena device-weight layouts plus an async,
  two-pass, bounded regular-file loader that verifies the complete file digest
  and selected dense-Llama tensor vocabulary before device allocation. Its
  second pass copies source-ordered chunks directly into final device regions,
  re-hashes the exact bytes copied, rechecks source size and modification time,
  retains no model-sized host snapshot, and closes partial allocation on
  failure.
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
  retires completed owners strictly in plan-sequence order. Live worker
  transport/overlap is not yet integrated.
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
  native code; runtime allocation instrumentation remains open.
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
- The streaming device loader relies on the approved read-only regular-file
  mount contract. It performs no-follow path-kind and opened-handle checks plus
  two complete digest passes, but MoonBit's portable file API does not provide
  an atomic `openat(O_NOFOLLOW)`/`fstat` boundary for adversarial writable
  directories. That stronger path-race exclusion remains unproven.
- Physical-CUDA validation of driver loading, transfers, modules, function
  launches, events, numerical BF16 GEMM and AOT correctness, implicit-heuristic
  shape support, and repeated resource balance. Local C stubs pass a manually
  instrumented ASan/UBSan run, but the MoonBit runtime was not instrumented and
  macOS leak detection was unavailable, so the full sanitizer/leak gate remains
  open.
- A device KV arena, block-table upload, paged-attention execution, or true
  incremental decode. The scheduler now transactionally acquires host request
  tables and page identities while building plans, and the host page allocator,
  block-table arena, and logical prefix metadata are implemented. The current
  device semantic graph still has no KV input, cache position, page table, or
  block mapping. Device profiles
  therefore support only stateless full prefill and full-sequence
  recomputation. The reference interpreter deliberately retains intermediate
  activations and recomputes the full sequence during generation; it is a
  correctness oracle, not a fallback.
- Online APIs, generated-text decoding/transport publication, continuous
  batching with global fairness/preemption, live worker overlap,
  device paged KV, online sampling and prefix integration, or telemetry. The
  scheduler registry/lifecycle, worker, sampling, page-allocation, block-table,
  prefix-index, and runtime-capacity foundations exist but are not an online
  service. No whole-token-step allocation or bounded-waiting claim is made.

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
