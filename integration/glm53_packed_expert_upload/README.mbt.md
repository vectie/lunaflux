# Checkpoint-backed GLM decoder execution

The adapter maps official Flash layer geometry and physical tensor names onto
generic precision plans, AOT sources and prepared execution frames. KDA, DSA,
mHC, dense FFN, routing and packed expert arithmetic retain their existing
owners; model adaptation does not implement another numerical backend.

`GlmRecurrentLayer` executes complete dense or routed recurrent blocks.
`GlmSparseLayer` executes retained indexed attention with both mHC envelopes,
output projection and routed/shared FFN. Both support either a standalone queue
or `prepare_frame` plus a borrowed `launches` view for a caller-owned queue.
The binding implementation is shared between these modes.

`GlmDecoderSlice` selects a contiguous interval of the official 45-layer hybrid
schedule. Preparation connects each layer's output to its successor using at
most two reusable four-stream BF16 residual buffers. All layers share one queue
and one completion per step. The warmed path only enqueues the prepared count;
it does not branch on model family, bind tensors or allocate intermediate frames.
Caller input/output are distinct borrowed residual allocations, not in-place
mHC destinations. They must remain alive until the slice queue is closed.

Before acquiring device buffers, preparation sums checkpoint weights, aligned
expert banks, request state, per-layer workspaces and residual scratch. The
caller supplies a total device ceiling. This enables bounded stage placement;
it does not imply that all 194 GB of GLM weights fit on one Spark or implement
cross-device handoff. AOT source export and compilation remain startup/offline.

The current DSA history represents one contiguous request. Slices containing
DSA therefore require one sequence and one cache slot. Their append descriptors
are borrowed device error ports for future model-level output delivery. Batched
DSA request slots, latent-cache compression and whole-model worker delivery are
not implemented by this slice. Recurrent-only slices retain independent CSR
request slots. Close the model queue before closing its borrowed layer frames;
active cancellation terminally discards all state in the slice.

Native regressions execute official-shaped zero-filled checkpoint fixtures and
device test doubles. Three early dense layers compose 78 launches per step;
the sparse block composes 37. These tests cover ownership and warm-path behavior,
not official-weight GPU numerics or throughput. Embedding, final normalization,
vocabulary head/sampling, worker integration and two-host execution remain.
