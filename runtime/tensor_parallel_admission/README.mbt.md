# Local tensor-parallel runtime admission

This package authenticates one explicit multi-device startup declaration
against an already admitted local device topology. It is a new Phase-7
foundation and does not reinterpret or widen the existing v1 single-device
runtime descriptor or `RuntimeInstanceAdmission`.

Every declaration repeats its stable rank, world size, process-visible ordinal,
bounded device identity, exact homogeneous target, and number of peer ranks.
Admission compares every scalar to the opaque topology evidence. Initial
support requires one host, one pipeline stage, and the complete local full mesh
already established by `engine/device_topology`; cross-node, pipeline,
heterogeneous, reordered, substituted, and incomplete declarations fail closed.

Each rank has an explicit physical inventory size, deployment memory ceiling,
and separate weight, activation/workspace, KV, and collective reservations.
Checked `Int64` arithmetic publishes exact per-rank totals, free capacity,
aggregate totals, and the minimum free capacity of any rank. This package opens
no device, spawns no worker, creates no NCCL handle, schedules no request, and
does not claim a tensor-parallel model plan or readiness.
