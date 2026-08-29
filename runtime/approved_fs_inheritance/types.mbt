///|
/// Fixed child descriptor roles for approved worker roots. The numeric
/// descriptor mapping is private to the native spawn boundary.
/// Opaque reusable owner of one ordered pair of pinned model and kernel root
/// leases. It exposes no descriptor, path, native handle, or role mutation.
pub type WorkerApprovedRoots

///|
/// Opaque spawn-only view of one retained root pair. It carries a reference to
/// the underlying pair but exposes no duplication, inspection, or close API.
pub type WorkerApprovedRootSpawnAuthority
