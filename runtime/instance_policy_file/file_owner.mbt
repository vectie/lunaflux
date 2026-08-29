///|
fn map_fs_error(
  error : @approved_fs.ApprovedFsError,
  operation : InstancePolicyFileOperation,
  maximum : Int64,
) -> InstancePolicyFileError {
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
fn finish_policy_file(
  outcome : Result[Bytes, InstancePolicyFileError],
  close : () -> Unit raise InstancePolicyFileError,
) -> Bytes raise InstancePolicyFileError {
  match outcome {
    Err(primary) => {
      close() catch {
        _ => ()
      }
      raise primary
    }
    Ok(bytes) => {
      close()
      bytes
    }
  }
}

///|
fn read_policy_snapshot(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected : InstancePolicyDigest,
  limits : InstancePolicyFileLimits,
) -> Bytes raise InstancePolicyFileError {
  let maximum = limits.max_policy_bytes()
  let file = root.open_file(locator) catch {
    error => raise map_fs_error(error, Open, maximum)
  }
  let outcome : Result[Bytes, InstancePolicyFileError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=maximum),
  ) catch {
    error => Err(map_fs_error(error, Snapshot, maximum))
  }
  let bytes = finish_policy_file(outcome, () => {
    file.close() catch {
      error => raise map_fs_error(error, Close, maximum)
    }
  })
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected.as_hex() {
    raise DigestMismatch
  }
  bytes
}
