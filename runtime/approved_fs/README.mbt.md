# Approved read-only filesystem capabilities

This package turns an independently approved canonical absolute directory path
into an opaque pinned directory capability. Files are opened only through
strict relative locators, using descriptor-relative component traversal with
no-follow semantics and final `fstat` type checks. Path strings, descriptors,
native handles, and platform errors never escape.

Opening proves directory/file identity at that instant; it does not infer that
a deployment root is approved or read-only. The caller supplies that authority.
Once a file is opened, rename or replacement of its former pathname cannot
redirect positional reads. Content digests and same-handle stamps remain the
authority for file contents. Stamps include byte count, modification time, and
metadata-change time from that same pinned descriptor.

Both roots and files require explicit deterministic close. Close is idempotent.
Each descriptor operation holds an atomic lifecycle lease; close reports
payload-safe `Busy(Close)` without consuming authority while a lease is active.
POSIX does not portably preserve descriptor retry authority after a close
error, so the wrapper becomes closed even when reporting that error. Finalizers
are a last-resort safety net rather than the production lifecycle.
