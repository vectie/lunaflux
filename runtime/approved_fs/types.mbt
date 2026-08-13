///|
/// Opaque explicit owner of one pinned read-only directory descriptor.
pub struct ApprovedRoot {
  priv handle : @native.ApprovedHandle
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
