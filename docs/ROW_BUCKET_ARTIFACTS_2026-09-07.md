# Per-row-bucket AOT projection artifacts

## Implementation

Dense and QKV projection export now retains measured small-row distribution
winners alongside the maximum-profile kernel, in the same AOT module. Each
optional variant declares a token-row bound, symbol, and launch dimensions.
The original entry remains the fallback. Without applicable tuning records,
the exporter produces no additional symbols or metadata.

The existing backend-neutral projection strategy and tile compilation pipeline
select and schedule each variant. CUDA lowering emits its device implementation.
Artifact and full-graph manifest serialization carry the typed variants through
startup loading. Startup creates smaller bounded queues using the smallest
covering variant and its bucket-specific geometry. No tuning, symbol lookup,
or filesystem access is added to the token-step path.

Variants retain the parent's operand layout and numerical contract. They are
currently consumed only by BF16 QKV/output projection execution; I8 manifests
reject this extension. Fused ingress does not automatically acquire these
projection variants. MLP, language-model head, attention, and alternative
residency schedules are not covered by this first extension.

## Validation and remaining work

Targeted native tests cover bounded selection, fallback, launch geometry,
metadata serialization, and invalid bounds/symbol aliases. An opt-in physical
test compares two dense schedules at token counts 1, 2, 7, 8 and the primary
fallback at 9, 16, 32, 256; it checks captured execution and untouched output
tails. Its synthetic tuning record is a test fixture, not a performance record.

Physical validation and real per-bucket timing are required before recording
a speedup or installing new tuning winners. This change does not resolve the
C8 token variation investigation or establish faster attention/GEMM schedules.
No new Qwen end-to-end benchmark result is claimed here.
