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

These owners are not the complete DSpark predictor. Prediction mHC mixing,
and MoE blocks, sequential Markov/confidence, distributed
capture assembly and base verification/commit remain required integration.
Whole-model correctness and performance are not established by component
ownership tests or CUDA compilation.
