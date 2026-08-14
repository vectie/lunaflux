# Approved read-only filesystem capabilities

This package turns an independently approved canonical absolute directory path
into an opaque pinned directory capability. `ApprovedRelativeLocator::new`
copies and validates strict relative text into an opaque no-`Debug` admission;
`ApprovedRoot::open_file` accepts only that type. Files are then opened using
descriptor-relative component traversal with no-follow semantics and final
`fstat` type checks. Path strings, descriptors, native handles, and platform
errors never escape.

`ApprovedRoot::require_absolute_identity` re-traverses one strict canonical
absolute label without following symlinks and requires its directory device
and inode to match the pinned capability. It exposes only success or a
payload-safe error: no path, descriptor, device, or inode value is returned.
This is an admission-time binding primitive, not ambient lookup authority.

Opening proves directory/file identity at that instant; it does not infer that
a deployment root is approved or read-only. The caller supplies that authority.
Once a file is opened, rename or replacement of its former pathname cannot
redirect positional reads. Content digests and same-handle stamps remain the
authority for file contents. Stamps include byte count, modification time, and
metadata-change time from that same pinned descriptor.

Startup readers may request one bounded immutable snapshot. That operation
holds one file lease across its before-stamp, exact positional read,
trailing-growth probe, and after-stamp. It allocates exactly one immutable
payload after the initial size is accepted, and rejects truncation, growth, or
size/mtime/ctime changes without publishing bytes. This is a startup primitive,
not a token-step API and not a substitute for content-digest verification.

Both roots and files require explicit deterministic close. Close is idempotent.
Each descriptor operation holds an atomic lifecycle lease; close reports
payload-safe `Busy(Close)` without consuming authority while a lease is active.
POSIX does not portably preserve descriptor retry authority after a close
error, so the wrapper becomes closed even when reporting that error. Finalizers
are a last-resort safety net rather than the production lifecycle.

`acquire_worker_approved_roots` creates an opaque reusable pair of independent
model and kernel root leases. Original roots remain valid and may close
independently. Rooted spawn borrows the same pinned pair across replacements
and maps only fixed roles; child import immediately creates an owned
close-on-exec `ApprovedRoot` and consumes the fixed descriptor. No descriptor,
path, handle, or role-reordering surface is exposed.
