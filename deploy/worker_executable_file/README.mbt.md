# Worker executable admission

Live admission opens every component of one strict absolute path without
following symlinks, requires a regular executable file, and snapshots it while
holding the same descriptor. Metadata is compared before and after the read.
On Linux the authenticated bytes are copied into a private executable memfd
and sealed against write, growth, shrink, and further seal changes before the
source descriptor is consumed. The digest is computed from that snapshot and
the published `WorkerExecutableAdmission` retains only the opaque pinned
descriptor capability plus diagnostic path and digest evidence.

Process activation duplicates that capability and executes descriptor 5 with
`fexecve`; it never reopens or rehashes the diagnostic path. Replacement,
symlink, relink, and hardlink mutation of the original namespace therefore
cannot change the executed bytes. Linux is the only live-spawn platform in
this version. Other native targets may verify materialized evidence but fail
closed with `UnsupportedPlatform` before child preparation.

`MaterializedWorkerExecutableEvidence` is deliberately separate from live
activation authority. Offline release verification can carry the exact target
label and digest without creating a value accepted by any spawn API. Live
admissions require explicit deterministic close; duplicate leases make close
return busy rather than invalidating an in-flight spawn.
