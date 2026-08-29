# Local device topology admission

`engine/device_topology` implements the first ADR-0010 startup capability: one
explicit same-host tensor-parallel rank order with homogeneous exact device
targets and bidirectional full-mesh peer access. Full mesh is LunaFlux's initial
supported topology, not a general NCCL requirement. Sparse, heterogeneous,
cross-node, duplicate, reordered, or ambient-extra device sets fail closed.

Rank is the declaration array index. Process-visible ordinals may be
non-contiguous but must be strictly increasing, so discovery cannot silently
reorder or substitute a device. Each rank binds its ordinal, bounded device
name, exact total memory, and exact catalog target.

The probe projection is intentionally authority-free and host-testable. It
contains only immutable scalar inventory and directed peer-access observations;
it cannot open contexts, enable peer access, create communicators, start
workers, or schedule a request. Production projection queries every unequal
ordered rank pair once in canonical rank order and feeds the same pure exact
admission used by tests. One-worker-per-rank ownership, model sharding,
collective contracts, and group generations remain later Phase-7 integration
boundaries.
