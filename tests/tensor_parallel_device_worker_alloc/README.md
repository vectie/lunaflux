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
