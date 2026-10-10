# Complete DeepSeek base decoder block

The model adapter composes both checkpoint-backed F32 mHC envelopes, retained
window/compressed attention with learned output, and automatic routed/shared
compact MoE. All launches belong to the containing stage's single queue. One
intermediate residual connects attention to FFN; learned output and expert
combine write directly into their mHC branch results, with no extra copy.

Expert banks, router operands and startup-narrowed token tables belong to the
stage loader. `bank_bytes()` describes the two compact expert capacities;
`required_device_bytes()` describes private frames and streamed non-expert
weights. A whole-stage budget must sum these disjoint allocations, other
borrowed operands and external residual ports before any upload. Close the
borrowing queue, this block, then stage-owned banks and ports.

This is complete **single-rank base-block** composition, not a tensor-parallel
head/expert implementation, a full-model worker or DSpark draft/verify execution.
DSpark's separate prediction blocks must not be replaced by ordinary causal
base blocks. Numerical components alone do not prove whole-model accuracy or speed.

## Contiguous checkpoint stages

`DeepSeekDecoderSlice` uploads complete routed/shared compact banks and router
weights only for its base-layer interval. It owns at most two reusable
inter-layer residual frames and one ordered queue. Request positions, token
IDs and startup-narrowed token-hash tables are borrowed;
the caller budgets those once. No checkpoint file access occurs in submit/poll.

`DeepSeekDecoderStage` composes embedding only at ingress, learned output only
at egress, and two distinct boundary residuals. Its aggregate pre-upload budget
includes text weights, compact banks, private state/workspace, residuals and its
own checked I32 token-hash tables. Startup streams exact I64 checkpoint tables
through bounded source/output chunks and row-local uniqueness state. It does not
retain a whole-model I64/I32 host arena. `maximum_host_bytes` bounds those payload
buffers; reader authentication/header metadata and request/transport ports must
still be budgeted by the process-level loader. Missing tables or insufficient
table payload budgets fail before any stage allocation.
It requires the worker's explicit absolute-position port. Text prefix/suffix
join the decoder queue; only a prepared complete range can bind whole-model
worker delivery. A partial stage is not an independently runnable model.

Compressed selection uses an explicit cache-relative precision-IR contract:
the reader accesses a separate compressed buffer, not a joined tensor with a
window prefix. No caller offset tensor, host offset update or per-step validation
is needed. Generic offset-relative plans remain available for joined views.

`DeepSeekActivationEdge` binds stage residuals to the common plaintext TCP/pinned
DMA owner. Its lease blocks closure and resubmission until DMA/network retirement;
rank and sender observers retire the same queue only once. `bind_rank` enforces
ingress/interior versus egress token ownership and exact prepared request ports.
`partition` uses shared pure feasibility planning with actual compact layer costs,
I32 hash tables, boundary text weights, four residual frames and caller reserve.

`cmd/deepseek_checkpoint` now joins these stages to shared two-rank control,
prompt frames and greedy generation. Base layers remain distinct from DSpark
prediction layers. Native source/build and lifecycle tests are not full-checkpoint
numerical execution. Actual two-host checkpoint generation, DSpark prediction
execution and independent reference comparison remain required before claiming
the complete DeepSeek/DSpark model is runnable.

## DSpark target capture

With explicit `capture_prediction_inputs=true`, each slice captures only the
official DSpark target layers in its own interval. Target means run directly
after the producing layer in the existing ordered queue, before residual reuse;
they are not the learned final text head. The shared precision IR owns stream
mean/concatenation semantics and CUDA lowering owns physical launch geometry.
The output and its functions are slice-owned and included in exact device bytes.

`prediction_capture_layers` exposes the local segment order. If a placement
splits target layers across ranks, the later prediction-input consumer must
assemble all ordered segments; a partial local capture is not a complete DSpark
input. The default base-only runner does not enable capture or incur its cost.
Target capture is executable infrastructure, not the still-missing prediction
blocks, sequential Markov adjustment, confidence output or draft verification.
