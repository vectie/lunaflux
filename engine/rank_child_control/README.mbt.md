# Rank-child control

`engine/rank_child_control` is the cooperative worker-side owner for one
same-host rank-control channel. It owns one inherited process channel, one RX
destination, one disjoint RX staging buffer, and one TX buffer. One read and
one write may progress independently; each progress call performs at most one
native I/O transition through `internal/process`.

The first `Configure` locks the exact rank-group wire binding but does not make
the child ready. The device-specific child must independently admit its local
manifest, artifacts, roots, rank bootstrap, and predecessor sequence, then
pass only the resulting root-free binding and predecessor to `begin_ready`.
This package deliberately imports no scheduler, device, NCCL, filesystem, or
model-family package.

Inbound values are owner-and-epoch capabilities. They retain no caller bytes,
become stale as soon as the owner consumes the command, and only copy payload
bytes from the owner's RX storage into a caller destination. Completed reads
are held unpublished while a prior TX response is active. Any transport,
wire, or local-admission failure permanently faults the owner and synthesizes
no protocol success.

An exact live `PollComplete` capability may be consumed by
`begin_collective_failed` or `begin_execution_failed`. Each emits the matching
payload-free v1 plan-failure response from retained TX storage, retires the
active sequence, and prevents later successful completion. Drain remains
publish-blocked behind an unconsumed inbound command or active TX response;
once published it permits the existing `Drained -> Close -> Closed` recovery
without inventing an `Aborted` response.
