# TP submission and graph replay

## Question and bounded experiment

The existing decoder submits a collective, queues its consumers, then waits
for `cuStreamQuery` before submitting the next collective. This is a GPU
completion dependency in the host submission loop, not a necessary dependency
between same-stream operations. The previous scoped trace measured 132 NCCL
calls per GLM step and substantial pre-collective gaps; profiling overhead and
peer wait prevent interpreting those gaps as recoverable wall time.

Compare three configurations on the retained six literal prompts, one excluded
warmup, unchanged AOT/numerics/TP2 placement: per-site completion, queued
submission, then queued submission with captured compute segments. Budget two
implementation stages with focused failure/correctness tests and repeated
end-to-end cohorts. Keep vLLM results and comparison outside compiler policy.
No payload hashing. Both hosts retain a 16 GiB available-memory reserve and
each task/device ceiling remains at most 100 GiB, with no task swap.

## Functional separation

`parallel_ir` describes immutable ordered compute intervals and collective
cuts, submission policy, and replay policy. Same-stream ordering preserves
producer/collective/consumer dependencies. Submission advances only after the
communication backend confirms its work has been enqueued; this is **not**
completion and does not release any operand.

The private native owner preallocates bounded pending-lease storage. Enqueued
collectives retain their buffers, stream and communicator until the terminal
stream fence retires the entire frame. Pending work rejects close/rebinding;
cancellation/failure stops submission and follows explicit abort/drain cleanup.
No steady-state allocation or numerical-policy change is needed.

Graph replay captures compute intervals only, with eager collectives between
them. This avoids treating NCCL capture support or dynamic row counts as an
implicit capability. All graphs and argument leases are startup-owned; fixed
launch geometry consumes mutable request metadata through existing buffers.
CUDA handles, capture ABI and events stay in the device backend. A bounded
graph reservation and actual selected-mode reporting are required. No fallback
may silently claim replay performance.

## Results

Pending implementation and separate physical measurements. Retained baseline:
9.756/10.189 s summed completion across six sequential requests; vLLM 3.163588 s.
Native uses BF16 KV; the reference uses FP8 MLA KV and breakable CUDA graphs.
These are not identical numerical implementations or a correctness oracle.
