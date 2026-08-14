///|
const STATUS_OK : Int = 0

///|
const STATUS_INVALID : Int = 1

///|
const STATUS_UNAVAILABLE : Int = 2

///|
const STATUS_UNSUPPORTED : Int = 3

///|
const STATUS_CLOSED : Int = 4

///|
const STATUS_TRUNCATED : Int = 5

///|
const STATUS_BUSY : Int = 7

///|
const STATUS_CHANGED : Int = 8

///|
const STATUS_TOO_LARGE : Int = 9

///|
const STATUS_IDENTITY_MISMATCH : Int = 10

///|
fn raise_open_status(
  status : Int,
  object : ApprovedFsObject,
) -> Unit raise ApprovedFsError {
  match status {
    STATUS_OK => ()
    STATUS_INVALID =>
      raise Invalid(if object == Root { RootPath } else { RelativeLocator })
    STATUS_UNAVAILABLE => raise Unavailable(object)
    STATUS_UNSUPPORTED => raise UnsupportedType(object)
    STATUS_CLOSED => raise Invalid(Lifecycle)
    STATUS_BUSY => raise Busy(Open)
    _ => raise OperationFailed(Open)
  }
}

///|
fn raise_operation_status(
  status : Int,
  operation : ApprovedFsOperation,
) -> Unit raise ApprovedFsError {
  match status {
    STATUS_OK => ()
    STATUS_CLOSED => raise Invalid(Lifecycle)
    STATUS_TRUNCATED => raise Truncated
    STATUS_BUSY => raise Busy(operation)
    STATUS_CHANGED => raise ChangedDuringRead
    STATUS_TOO_LARGE => raise FileTooLarge
    _ => raise OperationFailed(operation)
  }
}

///|
/// Opens one independently approved canonical absolute directory path without
/// following any component symlink. The returned value retains no path text.
pub fn ApprovedRoot::open_absolute(
  path : StringView,
) -> ApprovedRoot raise ApprovedFsError {
  validate_root_path(path)
  let status = Ref(STATUS_INVALID)
  let handle = @native.open_root(@utf8.encode(path), status)
  raise_open_status(status.val, Root)
  { handle, }
}

///|
/// Requires this pinned root to identify the exact directory currently named
/// by one strict canonical absolute label. Traversal follows no symlink and no
/// path, descriptor, device, or inode value is exposed.
pub fn ApprovedRoot::require_absolute_identity(
  self : ApprovedRoot,
  path : StringView,
) -> Unit raise ApprovedFsError {
  validate_root_path(path)
  let status = @native.require_absolute_identity(
    self.handle,
    @utf8.encode(path),
  )
  if status == STATUS_IDENTITY_MISMATCH {
    raise Invalid(RootIdentity)
  }
  raise_open_status(status, Root)
}

///|
/// Retains an ordered, reusable pair of independently duplicated model and
/// kernel root capabilities for worker replacement. Neither original root is
/// consumed, and no descriptor or locator is exposed.
pub fn acquire_worker_approved_roots(
  model_root~ : ApprovedRoot,
  kernel_root~ : ApprovedRoot,
) -> WorkerApprovedRoots raise ApprovedFsError {
  let status = Ref(STATUS_INVALID)
  let roots = @native.acquire_worker_roots(
    model_root.handle,
    kernel_root.handle,
    status,
  )
  raise_open_status(status.val, Root)
  roots
}

///|
/// Preallocates a closed native root-pair owner. It carries no filesystem
/// authority until `acquire_prepared_worker_approved_roots` succeeds.
pub fn prepare_worker_approved_roots() -> PreparedWorkerApprovedRoots {
  { roots: @native.prepare_worker_roots(), consumed: false }
}

///|
/// Internal owner-sharing hook used only to preinstall allocation-free rooted
/// publication shells. The prepared wrapper remains the sole acquisition key.
pub fn PreparedWorkerApprovedRoots::root_owner(
  self : PreparedWorkerApprovedRoots,
) -> WorkerApprovedRoots {
  self.roots
}

///|
/// Transactionally installs exact pinned duplicates into a preallocated owner.
pub fn acquire_prepared_worker_approved_roots(
  prepared : PreparedWorkerApprovedRoots,
  model_root : ApprovedRoot,
  kernel_root : ApprovedRoot,
) -> Unit raise ApprovedFsError {
  if prepared.consumed {
    raise Invalid(Lifecycle)
  }
  prepared.consumed = true
  raise_open_status(
    @native.acquire_prepared_worker_roots(
      prepared.roots,
      model_root.handle,
      kernel_root.handle,
    ),
    Root,
  )
}

///|
/// Allocation-sensitive activation primitive. The prepared owner is consumed
/// before the native call and the exact native status is returned unchanged.
pub fn acquire_prepared_worker_approved_roots_status(
  prepared : PreparedWorkerApprovedRoots,
  model_root : ApprovedRoot,
  kernel_root : ApprovedRoot,
) -> Int {
  if prepared.consumed {
    return STATUS_CLOSED
  }
  prepared.consumed = true
  @native.acquire_prepared_worker_roots(
    prepared.roots,
    model_root.handle,
    kernel_root.handle,
  )
}

///|
/// Opens and consumes the fixed descriptor for one child root role. The
/// returned root owns a fresh close-on-exec duplicate and behaves like any
/// other `ApprovedRoot`.
pub fn ApprovedRoot::open_inherited(
  role : ApprovedRootRole,
) -> ApprovedRoot raise ApprovedFsError {
  let status = Ref(STATUS_INVALID)
  let handle = @native.open_inherited_root(
    match role {
      Model => 0
      Kernel => 1
    },
    status,
  )
  raise_open_status(status.val, Root)
  { handle, }
}

///|
pub fn worker_approved_roots_is_closed(roots : WorkerApprovedRoots) -> Bool {
  @native.worker_roots_is_closed(roots)
}

///|
/// Deterministically consumes the retained pair. An active spawn returns
/// `Busy` without consuming either lease; an already closed owner is a no-op.
pub fn close_worker_approved_roots(
  roots : WorkerApprovedRoots,
) -> Unit raise ApprovedFsError {
  raise_operation_status(@native.close_worker_roots(roots), Close)
}

///|
/// Allocation-sensitive lifecycle primitive for internal aggregate owners.
/// The scalar is the native status; callers must still query closed state to
/// distinguish retained Busy authority from consumed failure evidence.
pub fn worker_approved_roots_close_status(roots : WorkerApprovedRoots) -> Int {
  @native.close_worker_roots(roots)
}

///|
/// Opens one strict relative locator beneath this pinned root. Every
/// intermediate component and the final regular file are opened with
/// no-follow semantics.
pub fn ApprovedRoot::open_file(
  self : ApprovedRoot,
  locator : ApprovedRelativeLocator,
) -> ApprovedFile raise ApprovedFsError {
  let status = Ref(STATUS_INVALID)
  let handle = @native.open_file(
    self.handle,
    @utf8.encode(locator.value),
    status,
  )
  raise_open_status(status.val, File)
  { handle, }
}

///|
pub fn ApprovedRoot::is_closed(self : ApprovedRoot) -> Bool {
  @native.is_closed(self.handle)
}

///|
pub fn ApprovedFile::is_closed(self : ApprovedFile) -> Bool {
  @native.is_closed(self.handle)
}

///|
/// Deterministically consumes this root descriptor. Closing an already closed
/// owner is a successful no-op. A native close error is reported, but the
/// wrapper remains closed because POSIX does not portably preserve retry
/// authority after a failed close.
pub fn ApprovedRoot::close(self : ApprovedRoot) -> Unit raise ApprovedFsError {
  raise_operation_status(@native.close(self.handle), Close)
}

///|
/// Deterministically consumes this file descriptor. Closing an already closed
/// owner is a successful no-op. A native close error is reported, but the
/// wrapper remains closed because POSIX does not portably preserve retry
/// authority after a failed close.
pub fn ApprovedFile::close(self : ApprovedFile) -> Unit raise ApprovedFsError {
  raise_operation_status(@native.close(self.handle), Close)
}

///|
/// Reads exactly one validated caller-owned range at an absolute file
/// position. EOF before completion fails without publishing a short count.
pub fn ApprovedFile::read_exactly_at(
  self : ApprovedFile,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
  byte_count~ : Int,
  position~ : Int64,
) -> Unit raise ApprovedFsError {
  if destination_offset < 0 ||
    byte_count < 0 ||
    position < 0L ||
    destination_offset > destination.length() - byte_count ||
    position > 0x7fffffffffffffffL - byte_count.to_int64() {
    raise Invalid(ReadRange)
  }
  raise_operation_status(
    @native.read_exact_at(
      self.handle,
      destination,
      destination_offset,
      byte_count,
      position,
    ),
    Read,
  )
}

///|
/// Returns bounded metadata from the same pinned handle used for reads.
pub fn ApprovedFile::stamp(
  self : ApprovedFile,
) -> ApprovedFileStamp raise ApprovedFsError {
  let output = FixedArray::make(5, 0L)
  raise_operation_status(@native.stamp(self.handle, output), Stat)
  {
    byte_count: output[0],
    modified_seconds: output[1],
    modified_nanoseconds: output[2].to_int(),
    changed_seconds: output[3],
    changed_nanoseconds: output[4].to_int(),
  }
}

///|
/// Reads the complete pinned regular file into one immutable startup payload.
/// The same operation lease covers both metadata stamps, the exact positional
/// read, and a trailing-byte probe. Size, modification time, or metadata-change
/// time differences fail without publishing the snapshot.
pub fn ApprovedFile::read_immutable_snapshot(
  self : ApprovedFile,
  maximum_bytes~ : Int64,
) -> Bytes raise ApprovedFsError {
  if maximum_bytes < 0L {
    raise Invalid(ReadRange)
  }
  let status = Ref(STATUS_INVALID)
  let snapshot = @native.read_immutable_snapshot(
    self.handle,
    maximum_bytes,
    status,
  )
  raise_operation_status(status.val, Read)
  snapshot
}
