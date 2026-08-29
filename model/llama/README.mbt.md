# Dense Llama semantic plan builder

This package terminates dense-Llama model-family branching in immutable model
plans. Full-context construction remains a canonical single-row compatibility
path. Paged-KV construction accepts an opaque positive
`LlamaPagedBatchEnvelope`, defaults to one row, and binds its exact row ceiling
into shape constraints, workspace bounds, and the plan digest.

For every full-context or paged plan, the builder passes only the metadata's
verified content digest, explicit limits, and complete semantic inputs into
`ModelPlan`. The generic self-authenticating plan encoder derives and mints the
plan digest after validation; no family digest or caller `ModelIdentity` is
injected. A different batch envelope changes authenticated constraints and
workspace semantics, so it necessarily produces a different plan digest.

The symmetric-I8 weight-only v1 builders are all-or-nothing. They keep the
embedding, every normalization parameter, activations, outputs, and KV state
in BF16. Every layer's Q, K, V, output, gate, up, and down matrix, plus the
language-model head, is dense row-major I8 with one unique plain-F32 scale of
shape `[out_channels]`. Tied embeddings fail closed before plan construction;
the opaque `LlamaI8WeightOnlyPolicy` exposes no layer, tensor, matrix, or
operation selection surface. Both full-context and paged variants delegate
identity minting exclusively to `ModelPlan`.

The opaque model spec admits at most 454 layers before any builder-owned tensor
or operation array exists. At that ceiling the I8 builder emits exactly 4,089
operations, 7,268 numeric tensors, and 9,086 total value inputs. The generic
plan envelope remains independently capped at 16,384 total value inputs.

The finite-E4M3 FP8 W8A8 v1 builders are likewise all-or-nothing. Embeddings,
normalization parameters, activation boundaries, outputs, and KV state remain
BF16. Every layer's Q, K, V, output, gate, up, and down matrix plus the
language-model head uses dense row-major `F8_E4M3`, owns one distinct scalar
F32 weight scale, and declares dynamic per-tensor F32 activation scaling with
F32 accumulation and BF16 output. Tied embeddings fail closed. Full-context
and paged variants bind the complete numeric schema into `ModelPlan`; neither
builder grants kernel, device, executor, service, or readiness authority. The
same 454-layer ceiling produces 4,089 operations, 7,268 numeric tensors, and
9,086 total value inputs, while 455 layers fail before construction.
