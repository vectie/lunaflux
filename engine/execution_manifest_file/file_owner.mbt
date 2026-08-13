///|
fn map_fs_error(
  error : @approved_fs.ApprovedFsError,
  operation : ExecutionManifestOperation,
  maximum : Int64,
) -> ExecutionManifestFileError {
  match error {
    Unavailable(_) => FileUnavailable
    UnsupportedType(_) => UnsupportedFileType
    FileTooLarge => FileTooLarge(maximum~)
    Truncated => FileTruncated
    ChangedDuringRead => FileChangedDuringRead
    Busy(_) => FileBusy(operation)
    Invalid(_) | OperationFailed(_) => FileOperationFailed(operation)
  }
}

///|
fn close_manifest_file(
  file : @approved_fs.ApprovedFile,
  maximum : Int64,
) -> Unit raise ExecutionManifestFileError {
  file.close() catch {
    error => raise map_fs_error(error, Close, maximum)
  }
}

///|
fn[T] raise_primary_after_close(
  primary : ExecutionManifestFileError,
  file : @approved_fs.ApprovedFile,
  maximum : Int64,
) -> T raise ExecutionManifestFileError {
  try close_manifest_file(file, maximum) catch {
    _ => raise primary
  } noraise {
    _ => raise primary
  }
}

///|
fn read_manifest_snapshot(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected : ExecutionManifestDigest,
  limits : ExecutionManifestFileLimits,
) -> Bytes raise ExecutionManifestFileError {
  let maximum = limits.max_manifest_bytes()
  let file = root.open_file(locator) catch {
    error => raise map_fs_error(error, Open, maximum)
  }
  let bytes = file.read_immutable_snapshot(maximum_bytes=maximum) catch {
    error => {
      let primary = map_fs_error(error, Snapshot, maximum)
      raise_primary_after_close(primary, file, maximum)
    }
  }
  // Successful bytes are not published until deterministic close succeeds.
  close_manifest_file(file, maximum)
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected.as_hex() {
    raise DigestMismatch
  }
  bytes
}
