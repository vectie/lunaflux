# DSpark executable main-hidden prefix

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

This owner produces the main stream consumed by prediction attention. It is not
the complete DSpark predictor: committed-main ring publication, noncausal draft
attention, three prediction blocks, sequential Markov/confidence and draft
verification remain distinct required operations. It never substitutes ordinary
causal base layers or the base text head for those operations.
