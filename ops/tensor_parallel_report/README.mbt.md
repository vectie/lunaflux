# Tensor-parallel capacity report

This package renders fixed-order operator evidence from the separate Phase-7
`TensorParallelRuntimeAdmission`. Reports show the authenticated local rank
ordering, process ordinals, bounded device identities, homogeneous target,
full-mesh link count, exact per-rank reservations and free capacity, aggregate
capacity, and the minimum free capacity of any rank.

Rejection rendering uses only stable area, reason, and rank scalars. It never
includes a device name, vendor diagnostic, path, model payload, or exception
text. The report does not probe hardware, allocate memory, spawn workers,
create collectives, or claim readiness. It is intentionally separate from the
v1 single-device descriptor report.
