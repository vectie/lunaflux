# DeepSeek and GLM numerical-program integration

## Scope and invariant

Use the numerical architecture at the actual prediction boundary of both
models. GLM's stream mean and BF16 normalization/logit publication must not be
replaced with DeepSeek's learned stream collapse and F32 logit publication.
Both bind the same typed scalar-expression and ordered-fold vocabulary, followed
by an explicit effect schedule and private CUDA lowering. Preserve existing
arithmetic, supported geometry, invalid-row behavior and allocation budgets.

Comparison is outside this path. No reference framework, measured score,
tolerance, payload digest or calibration result authorizes a compiler rewrite.
Tests and physical fixtures validate the integration, not universal correctness
or full-checkpoint quality. The earlier compressor integration remains in use;
this iteration does not claim every decoder operator has been migrated.

## Implemented

1. Move final normalization, vocabulary projection and learned stream collapse
   arithmetic into immutable numerical programs consumed by existing generators.
2. Derive the prepared head execution order from explicit read/write effects;
   keep the decoder boundary between embedding and egress, not a false direct
   dependency from embedding to output.
3. Expose borrowed offline replay ports at generic frames and both model owners.
   Include output count, selected local rows and input count. A selected local
   row is not an absolute token position; the offline caller owns that mapping
   and must retire the producing queue before reading.
4. Validate default source preservation, selection/rounding contracts, lifetime
   rejection and allocation-free warm execution, followed by bounded physical
   correctness, sanitizer and leak checks of generated model-head fixtures.

No recurring automation is created. Actual teacher-forced checkpoint captures,
held-out quality and cross-framework comparisons remain a separate evaluation
task, never a compiler dependency.

## Actual compiler/execution path

| Boundary | GLM | DeepSeek |
| --- | --- | --- |
| Model adapter | `GlmTextIo` | `DeepSeekTextIo` |
| Stream semantics | Ordered mean, BF16 publication | Learned normalized controls, ordered collapse, BF16 publication |
| Normalization | BF16 intermediate before learned weight | F32 intermediate before learned weight |
| Logits | BF16-rounded, stored as F32 | F32 |
| Effect-derived suffix | Normalize → head → greedy | Collapse → normalize → head → greedy |
| Offline replay | Five ports | Six ports including collapsed rows |

`TextIoPrecision::expression/fold` and
`LearnedStreamCollapsePrecision::expression/fold` define arithmetic; the CUDA
generator consumes them rather than copying a model-specific formula. The
existing parallel compiler derives ordering from actual read/write dependencies;
prepared frames bind its schedule once. The warm path has the same launches,
buffers, allocation behavior and queue completion boundary. `projection_suffix`
borrows all head work except greedy sampling for teacher-forced evaluation.

The common replay descriptor now distinguishes external inputs (`after_stage=None`)
from produced outputs, and carries a typed live-row extent referring to an I32
count port. This replaces the earlier compressor-specific append-index field.
DeepSeek's compressor owner also forwards its existing eight actual replay ports.
No capture, comparison, hashing or new device/host synchronization is performed
by these accessors. Borrowed ports remain valid only while their owning frame and
caller allocations remain alive; capture must precede reuse. Invalid selection
and count metadata must be rejected by the offline caller, not silently relabeled
as absolute positions or padded logits.

## Validation and limitations

The MoonBit implementation/refactoring workflow kept changes in focused packages,
preserved existing error/API behavior where possible, and tested immutable plans,
frame ownership, closed/unprepared rejection and zero-allocation warmed execution.
Arithmetic tests reject treating GLM's and DeepSeek's different rounding as an
exact replacement. Lowering tests cover typed sqrt versus reciprocal-sqrt and
parenthesized negation of compound operands.

Validation: 103/103 affected native tests, 5,095/5,095 full working-tree native
tests and both runner regressions pass. The transitive compiler dependency check
passes. `moon info` completes with 947 existing migration warnings and zero
errors; existing native warning exclusions remain `-20-25-29-35-79-92`. The full
suite includes unrelated in-progress toolchain/TLS changes, which are neither
staged nor committed here; this is not a clean commit-only suite claim. One old
generated-source snapshot changed because of redundant negation parentheses and
was refreshed after direct source/arithmetic inspection. No payload hashing or
integrity scan was added.

Physical root, retained locally and on `.179`:
`/tmp/lunaflux-text-numeric-20261012-v3`.
Both generated fixtures use hidden width 8, four streams, vocabulary 17, capacity
five input/two output rows and live-row counts 1/3/5. Selected output rows are
reordered. Both report zero observed output error on these fixture operands,
deterministic repeats, exact greedy results, untouched inactive rows and released
allocations. There are 11 mean-head intermediate-rounding witnesses and 16/78
learned-head normalization/logit-rounding witnesses. Memcheck, full leak check,
racecheck and synccheck all report zero errors/leaks/hazards. The serial user
units use 2 GiB limits and report zero swap. No GPU job was launched on `.178`.

Default-preservation root:
`/tmp/lunaflux-text-preservation-20261012-v3`.
Current emitted source matches the retained pre-refactor fixtures after only
outer whitespace, redundant unary parentheses and the previously introduced
private sigmoid helper alpha-renaming are normalized. This is a source regression
check, not an inference from approximate tensor agreement. The same fixture runs
against both binaries. Three alternating old/current pairs each record seven
CUDA-event samples of 500 suffix launches after 50 warmups. This small,
launch-sensitive timing is not full-size GEMM or serving throughput.

Median suffix time over 21 samples (microseconds): mean head old **10.237440**,
new **10.237312**; learned head old **20.473087**, new **20.471552**. These are
effectively unchanged; no speedup is claimed. Reproduce the summary with
`scripts/summarize-text-numeric-preservation.mbtx` on the preservation root.

Earlier incomplete preservation roots `v1`/`v2` are retained: they diagnosed
formatting/alpha-renaming differences, not a changed arithmetic result. The final
physical `v3` uses the repaired compound-negation renderer and shared IR operands.
No reference values, model parameters or tolerances were edited.

This work **does not** claim complete decoder expression-IR migration, new
whole-checkpoint token/quality results, bitwise parity with every backend, or
performance improvement. Actual token/absolute-position capture from both native
and external engines, matched checkpoint/dtype/context evaluation, within-engine
controls and held-out quality remain independent evaluation work.
