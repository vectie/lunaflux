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

An explicitly selected tiled-bank source also permutes the compact weight bank
in place during prepare. The caller lends that newly uploaded bank exclusively:
it must not concurrently bind another program or reuse the permuted bytes with
a row-major source. The startup-only AOT preparation and copy-back functions
reuse one expert's scratch, preserve every payload/scale byte, synchronize
before returning, and release scratch and the startup stream before submission.
Preparation is once per fresh program/bank; retry after failure requires a fresh
upload. Workspace planning includes this temporary storage. Ordinary sources
do not transform their banks. There is no request-path preparation or hashing.

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

prepare_routed_frame uses those same router/expert binding functions but returns
an owned launch view without creating an executor. A complete decoder borrows
that view into its single ordered queue, alongside its attention and residual
envelopes. Close the borrowing queue before closing the Program. This local
composition requires all logical experts in the bank: it does not drop remote
contributions or silently replace a collective. Partitioned banks retain the
existing prepare_routed/reduction path. The generic frame regression executes
32 steps / 320 launches with zero measured step allocations or blocking waits.
