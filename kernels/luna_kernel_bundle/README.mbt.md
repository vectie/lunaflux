# Luna kernel bundle

This native-only package is the inert offline join between typed Luna CUDA AOT
lowerings, independently compiled deterministic CUBIN receipts, the admitted
full-graph launch contract, and the existing schema-v2 execution-manifest
producer.

It never invokes a compiler, reads a file, starts a process, loads CUDA, or
derives launch geometry and operands. Typed admission requires the lowering's
exact dimensions, ordered operands, workspace contract, operation identity,
entry point, source digest, and recipe digest. Two immutable module snapshots
must match byte-for-byte, match both receipt hashes, and hash to the module
identity already present in the final contract.

`produce_luna_kernel_bundle` then proves unique full model-plan coverage,
orders operations by the immutable plan, delegates manifest authentication to
`full_graph_manifest`, and emits the exact deduplicated `sha256/<digest>.cubin`
inventory. No model-plan operation kind is missing a lowering family:
pointwise owns embedding, RMSNorm, RoPE, and residual add; projection owns QKV,
output projection, gated MLP, and language-model head; paged attention owns
causal attention. The residual-only compiler is accepted only for its exact
fixed-shape three-pointer ABI; live paged residual operations use pointwise.

Pointwise, projection, and paged-attention source recipes are all non-bindable
candidates and contain no module digest. Their strict post-compile binders
accept only the final full-graph contract, after the actual CUBIN digest has
become catalog authority. Bundle admission accepts only those opaque bound
forms, yielding the sequence candidate -> compile -> catalog -> contract ->
strict bind -> bundle without a provisional module identity.

FP8 v2 bundle admission accepts only its distinct bound lowering. It verifies
the exact final four- or eight-byte workspace operand with alignment four,
the complete ordered weight/scale raw ABI, the catalog-v4 execution digest,
the artifact's exact stable symbol, and byte-identical dual-build CUBIN
receipt. The v1 admission function and its four-byte contract remain unchanged.

This package is artifact assembly, not physical CUDA validation or evidence of
full-model inference.
