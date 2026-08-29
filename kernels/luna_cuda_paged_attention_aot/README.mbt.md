# Offline paged-attention CUDA AOT lowering

This package lowers one non-bindable catalog-v3 BF16 paged causal-attention
candidate into deterministic CUDA source and a closed offline compilation
recipe. The candidate is derived from the typed operation, profile, canonical
KV layout, stable entry-point ID, operands, target, and compiler policy; it
cannot see a module digest or final catalog family. The emitted ABI preserves
the launch-contract operand order exactly:
step counts, query positions, query-row offsets, sequence lengths, page-table
offsets, physical-page indices, QKV activation, output activation, key cache,
and value cache.

The reference-grade kernel supports LunaFlux's mixed step directly: prefill
rows precede decode rows according to the five-word `StepCounts` descriptor.
Each block computes one query-token/query-head result serially with FP32 dot
products and stable softmax, then rounds the result to BF16. The first query
head also writes every current token's K/V to the canonical split paged cache.
Attention reads K/V belonging to the current chunk directly from the immutable
QKV activation, so it never depends on grid-wide synchronization or
cross-block visibility. Prior-prefix K/V is read from the page table.

This is deliberately a correctness implementation, not a performance claim.
It allocates no workspace and rejects contracts with a workspace, a foreign
operand order, non-v3 semantics, noncanonical KV layout, unsupported BF16
target, reassociating compiler policy, or launch dimensions other than
`grid=(max_query_tokens, query_heads, 1), block=(1,1,1)`. The package invokes
no compiler, process, filesystem, CUDA runtime, or JIT path. Release tooling
must compile the source offline and admit the resulting content-addressed
CUBIN through the existing artifact boundary. Only then may
`bind_manifest_paged_attention_cuda_aot` join the candidate to the exact
full-graph contract and emit the final module-bearing binding record.

`scripts/probe-luna-paged-attention-cuda.sh` is an optional physical numerical
probe for the exact small fixture shape documented by the script. It compiles
the checked generated-source fixture to a CUBIN, loads that CUBIN through
the CUDA driver, executes a mixed prefill/decode batch, compares against an
independent host referee, and verifies fused KV writes. Merely adding this
script is not physical-CUDA evidence. The native test pins the exact generated
source and current non-bindable recipe digests; the validation script hashes both
checked fixtures and rejects any drift before the physical probe can use them.
