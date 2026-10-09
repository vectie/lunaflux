# Executable compact expert program

bind prepares reusable ordered launches from the AOT source and caller-owned
device functions/allocations. Submit them on one stream through the existing
OrderedKernelExecutor; completion and deterministic release stay with its owner.
No launch-path file inspection, format switch, compilation or allocation occurs.

Routing arrays contain unique global expert IDs per row. The immutable map from
ExpertMlpPrecision maps them to local bank positions, or -1 for another rank.
Unowned contributions become zero; the result is rank-local F32. A distributed
caller must reduce contributions and apply its model's final output rounding.

MoeProgram is the startup owner for the complete routed/shared path. Its immutable
precision plans determine the exact compact workspace budget. prepare loads the
offline AOT entry points, allocates intermediate and optional FP8 quantization
buffers, initializes the immutable expert maps and shared unit route, and binds
the reusable queue. Weight banks, input/output, counts, routing, stream and module
remain caller-owned. Update counts/routing on the same stream before submit.
The queue performs routed work, optional F32 reduction, shared work, and explicit
BF16 finalization without heap allocation in submit/poll. close releases the
queue before its functions and intermediate buffers; abort cancels active work.
Retain the owner on a preparation error and close it to release partial storage.

With an optional MoeRoutingCudaSource, prepare_routed instead owns the logits,
scores, corrected choice scores, expert IDs and selected weights. The queue is
projection -> score/correction -> group/top-k -> routed experts -> optional F32
reduction -> replicated shared expert -> BF16 finalization. Selection stays on
device and introduces no separate completion or host readback. Router weight and
the F32 correction vector remain caller-owned. GLM's adapter binds actual BF16
router matrices and rank-one F32 correction vectors from its Flash checkpoint;
the materializer does not reinterpret their physical ranks or expand weights.

source_bytes is offline compilation input, not permission to compile or JIT in
the request path. The owning Program still needs integration into whole-model
decoder blocks; it is not a complete serving runtime by itself.
