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
- Validated dense Llama-style BF16 dimensions and immutable artifact/plan
  identities.
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
- Two-pass bounded weight materialization with identity preservation, complete
  preflight validation, zero-copy tensor visitation, and an explicit host-copy
  reference representation.
- Deterministic Float reference kernels for dense-Llama embedding, RMSNorm,
  projections, split-half RoPE, causal grouped-query attention, residuals,
  SiLU-gated MLP, and the language-model head.
- An architecture-neutral offline plan interpreter that prepares immutable BF16
  weights once, executes validated operation plans, and performs bounded greedy
  generation. Pinned tiny-model fixtures cover non-degenerate attention and MLP
  paths with independently derived logits.
- Bounded deterministic greedy sampling with stable lowest-token tie breaking.
- A private dynamically loaded CUDA Driver/cuBLASLt ABI with opaque handles,
  explicit retryable resource release, parent-child ownership, and a semantic
  public device inventory. Unsupported hosts report a typed unavailability
  reason without attempting device allocation.
- A reusable depth-bounded JSON duplicate-key guard for map-backed parsers.
- A repository guard for package direction, sibling-product independence,
  CUDA ABI ownership, Python exclusion, temporary debt markers, and source-size
  review thresholds.

## Explicitly not implemented or not yet proven

- Tokenizer normalizer/pre-tokenizer variants outside the admitted byte-level
  BPE contract, and model file I/O.
- CUDA kernels, kernel selection, cuBLASLt execution, and validation on a real
  CUDA host. The macOS build proves unavailable-host behavior and injected
  lifecycle failures, but the current macOS toolchain invocation omits the
  sanitizer runtime from the generated final link.
- Production CUDA kernels, kernel selection, cuBLASLt execution, or a device
  executor. The reference interpreter deliberately retains intermediate
  activations and recomputes full prefill during generation; it is a
  correctness oracle, not a production fallback.
- A pinned external selected-model/tokenizer/logit corpus and model file I/O.
- Online APIs, scheduling, paged KV, prefix reuse, stochastic sampling, or
  telemetry.

The `lunaflux doctor` command reports the semantic CUDA inventory result and
the availability of the offline reference executor, while `lunaflux plan`
remains configuration-only. Both report production readiness as false; neither
reads model files or implies production CUDA inference support.

## Next correctness gate

The next correctness slice is bounded model file I/O plus a pinned external
selected-model/tokenizer/logit corpus, followed by a real device executor and
kernel catalog. The private CUDA ABI must also pass sanitizer, leak, and
real-device correctness gates on a CUDA CI host before it can be called
production-ready. Performance claims remain out of scope until the external
reference corpus and resource-balance gates pass.
