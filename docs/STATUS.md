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
- Validated dense Llama-style BF16 dimensions and immutable artifact/plan
  identities.
- Bounded decoded safetensors metadata validation, including checked tensor
  sizes, ranges, duplicates, overlaps, precision, and materialization budget.
- A bounded safetensors byte reader with checked little-endian header lengths,
  strict JSON schema, exact integer conversion, duplicate-key rejection, and
  complete payload accounting.
- An immutable architecture-neutral operation plan and dense Llama family
  builder with explicit shape, workspace, RoPE, activation, tensor-reference,
  and semantic kernel-capability contracts.
- A reusable depth-bounded JSON duplicate-key guard for map-backed parsers.
- A repository guard for package direction, sibling-product independence,
  CUDA ABI ownership, Python exclusion, temporary debt markers, and source-size
  review thresholds.

## Explicitly not implemented

- `tokenizer.json` and model-architecture JSON readers, tokenizer normalizer
  and pre-tokenizer compatibility, and model file I/O.
- CUDA ABI, device discovery, GPU resource ownership, cuBLASLt, or AOT kernels.
- Weight materialization, model-plan execution, logits, or token generation.
- Online APIs, scheduling, paged KV, prefix reuse, sampling, or telemetry.

The `lunaflux doctor` and `lunaflux plan` commands therefore inspect only the
checked foundation. They report readiness as false and do not probe hardware,
read model files, allocate weights, or imply inference support.

## Next correctness gate

The next vertical slice is the private native CUDA ABI with explicit lifecycle
tests and sanitizer evidence, followed by the remaining bounded artifact
readers, weight materialization, and an offline single-request BF16 reference
executor. Performance claims remain out of scope until reference logits and
resource-balance gates pass.
