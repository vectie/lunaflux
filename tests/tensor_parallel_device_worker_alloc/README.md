# Tensor-parallel device-worker warmed allocation evidence

This native executable admits and prepares two real tensor-parallel device
workers against deterministic fake device, ordered-executor, and collective
ABIs. It warms both ranks, then drives 65 additional plan cycles per rank
through stage, nonblocking execution and collective polling, leader/follower
completion, and reset.

The measured window redirects MoonBit and native allocation entry points to an
allocation probe. Positive controls first prove that record, array, and string
allocations are visible. The warmed window must have zero detected allocation,
zero blocking synchronization, exact false-then-true poll counts, exact
enqueue/collective/reset counts, no resource lifecycle changes, and no live
native children after deterministic close.

The fake native boundary is an evidence environment only. It does not make a
physical-GPU execution claim.

The executable repeats the full campaign with the overlap policy. The delayed
NCCL submission fixture rejects communication events before submission readiness;
the event fixture rejects unrecorded and same-stream waits. It verifies that
waits are issued while collective GPU completion is still pending. Both policy
runs retain the same zero-allocation, no-blocking and resource-balance gates.
Outside the measured window, overlap workers also exercise a failed event
close followed by retry, and abort while a collective is in flight.
