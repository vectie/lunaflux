# Worker executable admission

Live admission opens every component of one strict absolute path without
following symlinks and requires a regular executable file within the size ceiling.
It retains that descriptor directly, with no executable payload read, checksum,
or sealed-memfd copy. The supplied digest is a deployment label, not verified
integrity evidence. Deployment owns keeping the opened inode immutable.

Process activation duplicates that capability and executes descriptor 5 with
`fexecve`; it never reopens or rehashes the diagnostic path. Replacement,
and relink of the namespace do not retarget that descriptor. In-place mutation
of the opened inode is not prevented or authenticated. Linux is the live-spawn platform in
this version. Other native targets may verify materialized evidence but fail
closed with `UnsupportedPlatform` before child preparation.

`MaterializedWorkerExecutableEvidence` is deliberately separate from live
activation authority. Offline metadata inspection carries the exact target
and supplied label without reading its payload or creating a spawn capability. Live
admissions require explicit deterministic close; duplicate leases make close
return busy rather than invalidating an in-flight spawn.
