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
/// Opens one strict relative locator beneath this pinned root. Every
/// intermediate component and the final regular file are opened with
/// no-follow semantics.
pub fn ApprovedRoot::open_file(
  self : ApprovedRoot,
  locator : StringView,
) -> ApprovedFile raise ApprovedFsError {
  validate_relative_locator(locator)
  let status = Ref(STATUS_INVALID)
  let handle = @native.open_file(self.handle, @utf8.encode(locator), status)
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
