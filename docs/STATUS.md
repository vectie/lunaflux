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
- Bounded deterministic greedy sampling with stable lowest-token tie breaking.
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
- A prepared static GPU executor that cross-checks exact profiles and launch
  contracts against opened allocations/functions, constructs reusable launch
  arguments, and executes the ordered model graph. The memory plans, activation
  arena, AOT admission, and private launch seam are foundations, not execution
  evidence by themselves.
- A KV arena or true incremental decode. The current semantic graph has no KV
  input, cache position, page table, or block mapping. Device profiles therefore
  support only stateless full prefill and full-sequence recomputation. The
  reference interpreter deliberately retains intermediate activations and
  recomputes the full sequence during generation; it is a correctness oracle,
  not a fallback.
- Online APIs, scheduling, paged KV, prefix reuse, stochastic sampling, or
  telemetry.

The `lunaflux doctor` command reports the semantic CUDA inventory, bounded
reference loading, and offline executor status. `lunaflux plan` remains
configuration-only, while `lunaflux reference` reads digest-pinned files and
runs the correctness executor. All retain production readiness as false.

## Next correctness gate

The remaining Phase 1 promotion evidence is a physical-CUDA runner proving
transfers, module/function launches, BF16 GEMM and AOT numerics, balanced
repeated load/run/unload, concurrent resource stress, and the complete
sanitizer and leak gates, with physical concurrency/race evidence still open.
A prepared static executor remains open, as do KV ownership and true cached
decode. Performance and production-readiness claims remain out of scope until
physical correctness, resource balance, soak, and benchmark evidence passes.
