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

## Physical and integration checks

Migration checkpoint `fc3d4e1c` is pushed. Clean source-pinned native ARM builds
for both checkpoint executables passed, followed by fresh two-stage CUDA AOT
export/compilation. No prior cubins are used as evidence for changed lowering.

`/tmp/lunaflux-decoder-numeric-20261012-v2` exercises freshly generated recurrent,
causal-convolution and F32 hyper-connection kernels on Spark179, serially under
2 GiB/no-swap cgroups. Recurrent small/medium/64-head×128-key×128-value shapes
preserve output/state bits against the retained serial implementation, with a
separate F64 oracle check. Convolution small/tail/8192-channel shapes preserve
history and prefill-versus-decode chunking, with independent numerical checks.
The F32 hyper fixture checks control order, transposed residual orientation,
inactive rows and repeated execution. All seven cases pass memcheck/full leak,
racecheck and synccheck. These are fixtures, not full-model quality results.
The initial v1 export was stopped while waiting on the local full-suite build
lock; no GPU work had started. v2 uses an isolated build directory.

GLM now has an explicit `egress-logits` diagnostic: borrow the existing head
allocation, prepare a bounded transfer before execution and copy once after
EOF. Normal generation adds no observer, transfer or per-token comparison.
The resident runner's `logits` mode captures request 1's final selected row,
predicting generated token index 7 after consuming seven generated tokens.
Offline analysis must use that literal prefix, not call it initial-prefill
logits. Access before preparation is rejected and covered by the stage test.
The observer native check, stage test and eight runner tests passed; targeted
interface generation retains 936 existing migration warnings, zero errors.

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
