# DSpark executable prediction prefixes

The model adapter binds actual `mtp.0.main_proj.weight`, its UE8M0 scale plane,
and `mtp.0.main_norm.weight`. Three official target hidden captures form ordered
12,288-wide rows; the shared precision plan projects to 4,096 BF16 values then
applies weighted RMSNorm with the model epsilon.

`DeepSeekPredictionMain` streams compact checkpoint weights through bounded
scratch and joins two prepared launches to a borrowing parent queue. Device
budget includes both packed weights and both BF16 frames; caller input/count
storage stays separately owned. Partial or reordered target-layer lists fail
before upload. A distributed caller must assemble all ordered segments before
preparation, not silently reinterpret a local capture as complete input.

`DeepSeekPredictionAttention` now binds eight compact query/KV/norm/sink/output
weights for each actual `mtp.0` through `mtp.2` stage. Main and draft projection
share the same KV matrix/vector. Priming projects, normalizes, rotates/simulates
main KV and retains the ring without draft computation (seven launches).
Prediction also computes draft queries/KV and reads all draft slots after main
publication, inverse rotary and grouped Output-A/Output-B (nineteen launches).
Output-A retains its BF16 parameter boundary and Output-B block-FP8 activation
arithmetic. The result is five hidden-width BF16 rows, not unprojected heads.
An enclosing mHC block can lend its output allocation: its bytes are excluded
from private workspace and the attention owner never releases that allocation.
Separate prepared main/draft ports preserve the
reference's distinct count/position laws; no request-time source generation or
weight expansion is introduced.

`DeepSeekPredictionBlock` composes the same model-owned prediction address through
mHC attention and routed/shared MoE adapters. It prepares a main-only prime table
and a complete mHC/attention/FFN prediction table. Compact expert banks, router,
weights and workspace are counted before upload; caller residual output remains
borrowed. No base-layer address or base checkpoint namespace substitutes for an
`mtp.*` stage.

`DeepSeekPredictionHead` binds the last prediction stage's learned mHC controls,
norm and Markov/confidence planes while borrowing the global vocabulary matrix.
It selects all five output rows, borrows the learned frame's pre-normalization
hidden, and appends the dependent sampling chain instead of its independent
greedy suffix. Sampling/precision laws are shared; model checkpoint addressing
stays here and CUDA scheduling stays in the backend.

`DeepSeekPredictor` now composes the complete local prediction dataflow: ordered
base captures → main projection; seed plus four noise embeddings → three actual
prediction blocks → learned head and dependent Markov/confidence output. A
separate priming table publishes main KV without running the draft path. Private
weights and workspace are budgeted together before upload; global embedding and
vocabulary tables, captures and explicit result/RNG ports remain borrowed.
The generic seeded-block precision operation supplies absolute positions and
count planes on device, without token-step host construction.

The composed source passes native tests and compiles on GB10. Full checkpoint
preparation/replay, attachment to the base serving queue, distributed capture
assembly and base verification/commit remain required integration. Head and
block physical numerical validation are pending.
Whole-model correctness and performance are not established by component
ownership tests or CUDA compilation.
