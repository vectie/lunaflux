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

`GlmDecoderStage` now assembles the slice, text boundary and distinct residual
owners. Its complete stage budget is checked before the first allocation/upload.
Startup also matches worker row/vocabulary/history geometry. Only a complete
model range prepared against that worker's actual ports can bind token delivery;
an early/interior/last stage alone cannot publish a whole-model completion.
Partial stages expose residual ports for a composed pipeline owner.

The compiler's immutable `DecoderStageRange` determines ingress, interior,
egress or complete ownership. Ingress uploads only embedding and owns no output
workspace; egress uploads only final norm/head plus selected-row workspace;
interior stages load neither. All roles reuse the same numerical lowering.
`GlmDecoderSlice::partition` passes exact checkpoint layer footprints to the
generic suffix-feasibility planner, with boundary costs from
`GlmTextIo::boundary_device_bytes`. Four conservative residual frames and the
caller's explicit additional stage reserve are counted before host placement.
This is a memory-feasible contiguous partition, not a performance optimizer.
Runtime cross-host handoff and whole-model checkpoint generation are still open.

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
not official-weight GPU numerics or throughput.

`GlmTextIo` now streams the installed BF16 embedding, final norm and untied head
weights into a generic text frame. Its prefix replicates embedding rows into
residual streams. Its suffix averages streams, normalizes selected output rows,
projects to logits and performs deterministic greedy selection. Mean and
normalized activations round to BF16 before affine multiplication; this is not
DeepSeek's learned head collapse. The semantics follow Transformers'
`Glm5NextTextHyperHead` and `Glm5NextTextRMSNorm`:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py

Pass `prefix()` and `suffix()` to `GlmDecoderSlice::prepare` to place ingress,
decoder and egress in one queue. A complete GLM inference requires all decoder
layers, not an early-layer slice. Output count/row IDs are caller-owned device
ports; caller validates token IDs and counts before submission. Long prefill can
select only its final row, using 627,712 bytes for normalized state and logits
rather than a vocabulary buffer for every token. Global weight bytes, residual
allocations and slice buffers must also be included in the model device budget.
Invalid selected rows or non-finite logits yield a sampled-token sentinel `-1`;
the worker must reject it and check sparse append error ports before delivery.

The text kernels are ordered reference schedules, not throughput-tuned kernels.
Their small GB10 numerical/deterministic/sanitizer fixture passes; this is not
full-dimension or real-checkpoint validation. Native composition tests use
device doubles. The official-shaped first-stage fixture streams embedding and
one dense decoder (without norm/head tensors), runs 32 steps / 864 launches
with zero measured hot allocations/blocking waits, rejects an undersized budget
before allocation and releases active cancellation. Full worker delivery, non-greedy sampling,
model-wide two-host execution and actual checkpoint numerical comparisons remain
unfinished. Close the borrowing slice queue before closing the text frame.
