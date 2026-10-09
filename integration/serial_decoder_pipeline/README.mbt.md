# Two-rank serial decoder pipeline

One model-neutral coordinator joins a local ingress rank, plain remote terminal
rank control and explicit activation effects. Both ranks receive the same
canonical request plan. Metadata retirement precedes execution; residual DMA
retirement precedes remote execution; all stage results and the remote commit
receipt precede the coordinator's single model completion.

Only one contiguous request is active. Prefill continuation/decode use retained
rank-local history; request release must retire on both ranks before reuse.
Local or peer failure poisons the entire step. Close control, drain activation,
then close rank queues/ports. This implementation has two pipeline stages, not
tensor parallelism or an arbitrary topology scheduler.

The native regression uses real loopback control/activation sockets and a
device test double. It proves ordering, transferred bytes, token publication,
reuse and failure ownership; it does not prove real-checkpoint GPU arithmetic,
GPUDirect, RDMA, whole-model throughput or three-model support.
