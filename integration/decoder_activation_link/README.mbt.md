# Decoder activation link

A model-neutral execution edge between prepared decoder stages. A sender waits
for the producer queue to retire, then downloads only live activation rows into
one startup-owned pinned lane. Plain framed TCP carries the step epoch, row count
and unchanged tensor bytes. A receiver uploads only a complete matching frame;
its completion is the dependency required before consumer submission.

Storage is fixed and budgeted before acquisition. No JIT, model-family branch,
TLS, per-step artifact scan or allocation of activation-sized buffers occurs.
The channel's async polling costs are not claimed heap-free. This host-staged
transport is a correctness path, not GPUDirect/RDMA or a performance claim.

Close links before their borrowed activation allocations and contexts. Failed
preparation/progress remains closeable; cancellation drains the transfer lane.
The producer allocation must not be overwritten until sender completion; the
consumer must not be submitted until receiver completion. The composed owner
must enforce these dependencies, just as for borrowed launch arguments.
Request metadata propagation and whole-model orchestration are separate work.
