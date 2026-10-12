# Decoder numerical-program migration

The migration preserves the selected numerical program. It does not select
arithmetic by matching a reference, claim that agreement proves correctness, or
add payload hashing. Model adapters keep their model semantics; scalar rounding,
fold topology and state effects belong to compiler plans; CUDA spells the
private implementation. Quantized encodings and block matrix primitives retain
their existing typed precision laws rather than being flattened into scalar
BF16 arithmetic.

## Work and validation ledger

| Family | Expression/effect migration | Validation |
| --- | --- | --- |
| Text and learned heads | Previous committed work | 103 affected tests; bounded Spark head fixtures |
| Learned compressor | Previous committed work | Typed folds/effects and actual prepared schedule |
| Rotary product | Expression consumed by tensor/cache lowerings | Separate and fused programs remain distinct |
| HC nonlinear controls | Affine, sigmoid, root, softmax publication, collapse and both post-mix expressions consumed | Existing explicit control/tree laws retained; full physical check pending |
| Dense SwiGLU | Explicit projection/activation/product rounding; effect-derived frame order | Native plan/source checks; physical check pending |
| Recurrent delta | Explicit square/dot folds, normalization, decay, state update | Existing exclusive-slot ownership retained; physical check pending |
| Causal convolution | Ordered tap fold and activation expression | Existing raw-history publication retained; physical check pending |
| Query/output/expert low-bit paths | Existing typed quantization/scaled-dot laws plus matrix publication, clamp, activation and route expressions | Physical low-bit primitives stay in private lowering |
| Sparse/window attention, indexing, routing/combine | Probability storage, normalization/division, scores, activation and combine expressions consumed | Sparse pool→index→attention frame order now derives from effect IR |

Estimated division is a distinct primitive from rounded division. The exact
rewrite gate rejects replacing one with the other. This retains the existing
GLM sparse probability law, rather than silently changing it during migration.
Rounding at BF16 product, route weighting, state and output boundaries is explicit.
Matrix fragments and reduction trees remain typed operations, not scalar folds.

Working-tree warning-denied native check passed. Initial full suite: 5,088/5,099;
the failures were source-spelling/count and source-digest snapshots. They are
being replaced by explicit arithmetic/source assertions, not numerical tolerance
changes. `moon info` completed with 3,656 existing migration warnings and zero
errors. Compiler dependency closure: 11 packages, no evaluation dependency.
GLM runner tests: 3/3. Full rerun and Spark results remain pending below.

Comparison and teacher-forced evaluation stay in offline runners. Full-model
validation must include final logits and token/position semantics, not just
hidden-state similarity. No whole-model quality or completed migration claim is
made by this work-in-progress ledger.

The user explicitly requested stopping MiniMax to use both Sparks. Its exact
user unit was stopped; systemd reports timeout during termination, MainPID=0,
and no remaining GPU process. This is an interrupted experiment, not completed
media validation. Logs and partial outputs were preserved. Both Sparks are now
available for serial, bounded two-rank validation.

The GLM runner now exports and compiles fresh AOT from the source-pinned binary.
Reusing the old AOT would not exercise migrated kernels. Existing declared shard
labels are copied as metadata; no payload hashes are computed.
