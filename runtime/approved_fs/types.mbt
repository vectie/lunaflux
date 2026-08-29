///|
/// Opaque explicit owner of one pinned read-only directory descriptor.
pub struct ApprovedRoot {
  priv handle : @native.ApprovedHandle
}

///|
/// Borrowed offline-materialization view of one pinned source root and the
/// canonical absolute label that the exact copied bytes will have in the
/// deployment namespace.
///
/// Construction authenticates the source root's current no-follow identity.
/// It deliberately does not assert that the target label exists on this host;
/// only the no-overwrite materializer may consume this view. The caller keeps
/// sole close authority over the underlying `ApprovedRoot`.
pub struct ApprovedRootMaterializationView {
  priv source_root : ApprovedRoot
  priv target_label : String
}

///|
/// Opaque reusable owner of ordered model and kernel worker-root leases.
pub type WorkerApprovedRoots = @inheritance.WorkerApprovedRoots

///|
/// Capability-limited root-pair view accepted only by the private spawn ABI.
pub type WorkerApprovedRootSpawnAuthority = @inheritance.WorkerApprovedRootSpawnAuthority

///|
/// Closed, authority-free allocation shell consumed by exactly one root-pair
/// acquisition attempt.
pub struct PreparedWorkerApprovedRoots {
  priv roots : WorkerApprovedRoots
  priv mut consumed : Bool
}

///|
/// Fixed child descriptor roles for approved worker roots.
pub(all) enum ApprovedRootRole {
  Model
  Kernel
} derive(Debug, Eq)

///|
/// Opaque validated strict relative descendant locator. Construction performs
/// the complete lexical admission once; the retained text is intentionally
/// unavailable to callers and the type has no diagnostic formatting surface.
pub struct ApprovedRelativeLocator {
  priv value : String
}

///|
pub fn ApprovedRelativeLocator::new(
  value : StringView,
) -> ApprovedRelativeLocator raise ApprovedFsError {
  validate_relative_locator(value)
  { value: value.to_owned() }
}

///|
/// Opaque explicit owner of one pinned read-only regular-file descriptor.
pub struct ApprovedFile {
  priv handle : @native.ApprovedHandle
}

///|
/// Bounded same-handle file metadata used to detect changes across reads.
pub struct ApprovedFileStamp {
  priv byte_count : Int64
  priv modified_seconds : Int64
  priv modified_nanoseconds : Int
  priv changed_seconds : Int64
  priv changed_nanoseconds : Int
} derive(Eq)

///|
pub fn ApprovedFileStamp::byte_count(self : ApprovedFileStamp) -> Int64 {
  self.byte_count
}

///|
