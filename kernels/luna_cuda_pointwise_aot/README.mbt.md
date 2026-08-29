# BF16 pointwise CUDA AOT lowering

`kernels/luna_cuda_pointwise_aot` is an offline-only, deterministic lowering
boundary for the reference-grade BF16 kernels that do not belong to GEMM or
paged-attention ownership.

The manifest-backed v1 slice covers:

- embedding lookup;
- row-wise RMSNorm with a fixed 256-lane Float32 reduction order;
- positioned split-half Llama RoPE over Q and K, with V copied unchanged;
- residual addition with one ordered Float32 add and BF16 round-to-nearest.

Lowering is deliberately two-stage so content addressing is not circular. A
candidate consumes an existing typed `PlanOperation`, exact paged profile,
target, compiler policy, entry-point ID, and launch operands. It requires the
catalog-v3 positional order—runtime inputs, semantic inputs, then output—and
checks every envelope byte count before emitting source. The candidate is not
manifest-bindable. After compilation has produced a content-addressed module,
the strict binder consumes the admitted `PagedKvLaunchContractSet`, selects the
unique full-graph contract, and requires exact target, catalog version,
profile, operation semantics, dimensions, operand metadata, and entry-point ID
equality before exposing the full module/family entry point. Its binding record
authenticates that join. The profile fixes the maximum launch shape while
`StepCounts[3]` bounds live tokens at execution. The eventual CUBIN may be
stored at the checked relative locator `sha256/<artifact-digest>.cubin`
without introducing a mutable cache key.

The package also emits a standalone canonical KV-cache writer for differential
and dispatch tests. It uses layout-v1
`[page][token][kv-head][head-dimension]` storage and retains the attention
runtime-input order before rotated-QKV, Key, and Value operands. It is
deliberately marked `manifest_bindable=false`: the current semantic plan gives
`KvCacheReadWrite` ownership to paged attention, whose production kernel must
fuse current-chunk KV publication and attention rather than pretend this
auxiliary is a complete `CausalAttention` implementation.

The numerical contracts are explicit. Embedding and KV write are BF16-copy
exact; residual add preserves one stated add and BF16 rounding; RMSNorm and
RoPE use named Float32 tolerance contracts because reduction and
transcendental implementations are device/toolchain relative. Reassociation
is rejected and fast-math is never requested.

This package does not compile source, open files, spawn processes, load CUDA,
or execute on a request path. It does not cover QKV/output/LM-head projection,
gated MLP, paged prefill/decode attention, sampling reductions, CUDA graphs,
artifact admission, or benchmark promotion. Those remain separately owned
families and gates.

`fixtures/physical_sm120` records the exact generated CUDA and recipe bytes for
the four sm120 numerical-probe shapes. `physical_fixture_test.mbt` binds both
SHA-256 values back to fresh typed lowerings, while
`scripts/validate-luna-bf16-family-physical-fixtures.sh` verifies the checked
files. The separate CUDA Driver probe compiles only these authenticated sources
and compares live rows with an independent host referee. These fixtures are a
reproducible test input, not evidence that the probe ran on physical hardware.
