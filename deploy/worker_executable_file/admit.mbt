///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub fn WorkerExecutableDigest::from_sha256(
  value : StringView,
) -> WorkerExecutableDigest raise WorkerExecutableFileError {
  if !is_lower_sha256(value) {
    raise InvalidDigest
  }
  WorkerExecutableDigest(value.to_owned())
}

///|
pub fn WorkerExecutableDigest::as_hex(self : WorkerExecutableDigest) -> String {
  self.0
}

///|
pub fn WorkerExecutableFileLimits::new(
  maximum_bytes~ : Int64,
) -> WorkerExecutableFileLimits raise WorkerExecutableFileError {
  if maximum_bytes <= 0L || maximum_bytes > 1073741824L {
    raise InvalidLimit
  }
  { maximum_bytes, }
}

///|
pub fn WorkerExecutableAdmission::activation_path(
  self : WorkerExecutableAdmission,
) -> Bytes {
  self.activation_path
}

///|
pub fn WorkerExecutableAdmission::digest(
  self : WorkerExecutableAdmission,
) -> WorkerExecutableDigest {
  self.digest
}

///|
pub fn MaterializedWorkerExecutableEvidence::activation_path(
  self : MaterializedWorkerExecutableEvidence,
) -> Bytes {
  self.activation_path
}

///|
pub fn MaterializedWorkerExecutableEvidence::digest(
  self : MaterializedWorkerExecutableEvidence,
) -> WorkerExecutableDigest {
  self.digest
}

///|
/// Returns whether this admission still owns live pinned activation authority.
/// Offline materialization evidence deliberately returns false.
pub fn WorkerExecutableAdmission::is_activation_live(
  self : WorkerExecutableAdmission,
) -> Bool {
  !raw_executable_is_closed(self.handle)
}

///|
/// Explicitly retires the pinned executable. A busy lease remains retryable;
/// every other close attempt consumes the descriptor authority.
pub fn WorkerExecutableAdmission::close(
  self : WorkerExecutableAdmission,
) -> Unit raise WorkerExecutableFileError {
  match raw_close_executable(self.handle) {
    0 => ()
    7 => raise FileBusy
    _ => raise FileOperationFailed
  }
}

///|
/// Preallocates a child whose only executable authority is this pinned object.
/// The path label and digest remain diagnostic/deployment evidence only.
pub fn WorkerExecutableAdmission::prepare_child(
  self : WorkerExecutableAdmission,
  limits : @process.ProcessLimits,
) -> @process.PreparedChildProcess raise WorkerExecutableFileError {
  if raw_executable_is_closed(self.handle) {
    raise FileUnavailable
  }
  if !raw_executable_spawn_supported(self.handle) {
    raise UnsupportedPlatform
  }
  @process.prepare_child_from_approved_executable(self.handle, limits)
}

///|
fn split_absolute_path(
  path : StringView,
) -> (String, String) raise WorkerExecutableFileError {
  if path.length() < 4 || path.length() > 4096 || path[0] != '/' {
    raise InvalidPath
  }
  let mut slash = -1
  let mut segment_length = 0
  let mut dots = 0
  for index in 1..<path.length() {
    let unit = path[index]
    if unit == '/' {
      if segment_length == 0 || (segment_length <= 2 && dots == segment_length) {
        raise InvalidPath
      }
      slash = index
      segment_length = 0
      dots = 0
    } else {
      if !(unit is ('A'..='Z' | 'a'..='z' | '0'..='9' | '_' | '-' | '.')) {
        raise InvalidPath
      }
      segment_length += 1
      if unit == '.' {
        dots += 1
      }
    }
  }
  if slash <= 0 ||
    segment_length == 0 ||
    (segment_length <= 2 && dots == segment_length) {
    raise InvalidPath
  }
  (path[0:slash].to_owned(), path[slash + 1:path.length()].to_owned())
}

///|
fn map_fs_error(
  error : @approved_fs.ApprovedFsError,
) -> WorkerExecutableFileError {
  match error {
    Unavailable(_) => FileUnavailable
    UnsupportedType(_) => UnsupportedFileType
    FileTooLarge => FileTooLarge
    Truncated => FileTruncated
    ChangedDuringRead => FileChangedDuringRead
    Busy(_) => FileBusy
    Invalid(_) | OperationFailed(_) => FileOperationFailed
  }
}

///|
/// Snapshot-verifies an executable before retaining only path/digest evidence.
pub fn verify(
  absolute_path : StringView,
  expected : WorkerExecutableDigest,
  limits : WorkerExecutableFileLimits,
) -> WorkerExecutableAdmission raise WorkerExecutableFileError {
  ignore(split_absolute_path(absolute_path))
  let status = Ref(1)
  let handle = raw_open_executable(
    @utf8.encode(absolute_path),
    limits.maximum_bytes,
    status,
  )
  match status.val {
    0 => ()
    1 => raise InvalidPath
    2 => raise FileUnavailable
    3 => raise UnsupportedFileType
    7 => raise FileBusy
    9 => raise FileTooLarge
    _ => raise FileOperationFailed
  }
  let snapshot_status = Ref(1)
  let bytes = raw_snapshot_and_pin(
    handle,
    limits.maximum_bytes,
    b"",
    snapshot_status,
  )
  if snapshot_status.val != 0 {
    ignore(raw_close_executable(handle))
    raise match snapshot_status.val {
      3 => UnsupportedFileType
      4 => FileUnavailable
      5 => FileTruncated
      7 => FileBusy
      8 => FileChangedDuringRead
      9 => FileTooLarge
      _ => FileOperationFailed
    }
  }
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected.as_hex() {
    ignore(raw_close_executable(handle))
    raise DigestMismatch
  }
  { activation_path: @utf8.encode(absolute_path), digest: expected, handle }
}

///|
fn verify_bound(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  activation_path : StringView,
  expected : WorkerExecutableDigest,
  limits : WorkerExecutableFileLimits,
) -> MaterializedWorkerExecutableEvidence raise WorkerExecutableFileError {
  let file = root.open_file(locator) catch {
    error => raise map_fs_error(error)
  }
  let read : Result[Bytes, WorkerExecutableFileError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=limits.maximum_bytes),
  ) catch {
    error => Err(map_fs_error(error))
  }
  let file_close = Ref(true)
  file.close() catch {
    _ => file_close.val = false
  }
  match read {
    Err(primary) => raise primary
    Ok(_) if !file_close.val => raise FileOperationFailed
    Ok(bytes) => {
      let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
      if actual != expected.as_hex() {
        raise DigestMismatch
      }
      { activation_path: @utf8.encode(activation_path), digest: expected }
    }
  }
}

///|
/// Snapshot-verifies the staged worker bytes through a typed source root while
/// retaining only the exact final deployment activation path.
///
/// The source basename is explicit because an offline assembler may rename an
/// independently built artifact. The target parent must exactly match the
/// view's canonical target label before any staged descendant is opened.
pub fn verify_materialized(
  source : @approved_fs.ApprovedRootMaterializationView,
  source_basename : StringView,
  target_path : StringView,
  expected : WorkerExecutableDigest,
  limits : WorkerExecutableFileLimits,
) -> MaterializedWorkerExecutableEvidence raise WorkerExecutableFileError {
  let (target_parent, target_basename) = split_absolute_path(target_path)
  if target_parent != source.target_label() ||
    target_basename != source_basename.to_owned() {
    raise InvalidPath
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(source_basename) catch {
    _ => raise InvalidPath
  }
  verify_bound(source.source_root(), locator, target_path, expected, limits)
}
