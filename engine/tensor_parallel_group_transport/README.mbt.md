# Tensor-parallel group transport

This package is the private physical group owner between scheduler plans and a
same-host rank group. It retains one duplicated approved model/kernel root pair
and immutable, authority-free v5 tensor-parallel source and planning evidence.
It retains only the already admitted rank-child activation path bytes returned
by `WorkerExecutableAdmission.activation_path()`; deployment owns executable
snapshot verification. It never opens a device context and never exposes a
backend handle.

Every start attempt consumes a fresh nonzero rank-group generation, re-admits
the source-bound runtime policy and collective runtime, creates a fresh opaque
rendezvous identity, and reauthenticates each rank's sharded source. A
capability-limited spawn view lets RankGroupProcess borrow that pair without
gaining role duplication or close authority. Rank-local
device, KV, manifest, execution, bootstrap, wire, and Configure contracts are
rebuilt in canonical rank order. Artifact-bearing manifest admissions remain
loop-local; only inert execution evidence survives long enough to build the
group bootstrap, and child processes load their own module bytes.

The v5 source digest binds an explicit bounded collective/exchange timeout.
Rank-group begin samples its owned monotonic clock and derives the absolute
deadline internally with checked arithmetic. No scheduler or service caller
supplies an ambient timestamp or sleeps on behalf of the group.

A process is published only after every rank has completed Configure/Ready.
Failure or partial startup retains only whole-group cleanup authority. A
replacement cannot be built until all old children have been reaped, and it
never reuses the old generation or rendezvous identity. The mutable scheduler
predecessor advances only after the exact accepted completion is committed and
retired. Failed submitted and scheduler-only buffered plans advance through
separate authenticated retirement paths, so a replacement cannot replay the
construction-time predecessor. Recovery retains an explicit replacement
obligation even for an idle rank loss whose plan sequence is zero.

Healthy close is a rank-ordered `Drain/Drained` then `Close/Closed` exchange
before child reap. If one transport is lost, the exact lost rank is excluded
while every surviving rank still receives best-effort drain and close; only
the irreducibly lost process falls back to OS-owned termination/reap.

Public vocabulary stays backend- and model-family-neutral. Model-family
inspection details remain private to the opaque materializer inspection; the
concrete collective implementation remains behind the device/private ABI.
