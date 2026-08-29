# Full-graph AOT execution-manifest production

This native, offline-only package joins one already admitted catalog-v3
`FullGraph` launch-contract set with exact per-operation AOT artifact metadata.
Every operation must preserve the contract's entry point, CUDA launch
dimensions, ordered semantic operands, operand byte counts and alignments, and
workspace contract. The package never derives or adjusts launch geometry.

Every module locator is fixed to `sha256/<module-digest>.cubin`; alternate or
merely digest-paired paths are rejected. Successful admission emits
deterministic canonical schema-v2 bytes plus their SHA-256 identity. Those
bytes are input to the existing independently
authenticated `engine/execution_manifest_file` loader. Runtime and release
assembly continue to treat the document as untrusted implementation claims and
rederive semantic contracts from typed model, memory, KV, and step evidence.

The package has no filesystem, CUDA, compiler, process, environment, JIT, or
device authority. Artifact files and the emitted manifest are persisted only
by an outer offline build/release tool.

This producer is graph-complete only when every semantic operation has a real
offline artifact. LunaFlux now has deterministic BF16 source/recipe lowering
families for embedding, RMSNorm, positioned RoPE, residual add, QKV and dense
projection, gated MLP, language-model head, and mixed paged attention. Strict
post-compile binders and `kernels/luna_kernel_bundle` can join actual
deterministic CUBIN receipts to the final launch contracts before calling this
producer. Sampling remains host-owned because it is not an authenticated model-
plan operation.

No product-owned outer command currently derives and persists that complete
candidate/contract/bundle sequence from approved model inputs. This package
does not produce the separate catalog-v4 I8 manifest, and neither its typed
output nor scoped physical kernel fixtures establish approved full-model CUDA
execution, serving, sanitizer/leak, soak, or benchmark evidence.
