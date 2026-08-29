///|
fn map_fs_error(
  error : @approved_fs.ApprovedFsError,
  operation : RuntimeDescriptorFileOperation,
  maximum : Int64,
) -> RuntimeDescriptorFileError {
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
fn bind_root_label(
  root : @approved_fs.ApprovedRoot,
  label : StringView,
  operation : RuntimeDescriptorFileOperation,
  maximum : Int64,
) -> Unit raise RuntimeDescriptorFileError {
  root.require_absolute_identity(label) catch {
    error => raise map_fs_error(error, operation, maximum)
  }
}

///|
fn close_descriptor_file(
  file : @approved_fs.ApprovedFile,
  maximum : Int64,
) -> Unit raise RuntimeDescriptorFileError {
  file.close() catch {
    error => raise map_fs_error(error, Close, maximum)
  }
}

///|
fn[T] finish_descriptor_body(
  outcome : Result[T, RuntimeDescriptorFileError],
  close : () -> Unit raise RuntimeDescriptorFileError,
) -> T raise RuntimeDescriptorFileError {
  match outcome {
    Err(primary) => {
      close() catch {
        _ => ()
      }
      raise primary
    }
    Ok(value) => {
      close()
      value
    }
  }
}

///|
fn read_descriptor_snapshot(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
) -> Bytes raise RuntimeDescriptorFileError {
  let maximum = limits.max_descriptor_bytes()
  let file = root.open_file(locator) catch {
    error => raise map_fs_error(error, Open, maximum)
  }
  let outcome : Result[Bytes, RuntimeDescriptorFileError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=maximum),
  ) catch {
    error => Err(map_fs_error(error, Snapshot, maximum))
  }
  // The immutable snapshot cannot escape until terminal close succeeds.
  let bytes = finish_descriptor_body(outcome, () => {
    close_descriptor_file(file, maximum)
  })
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected.as_hex() {
    raise DigestMismatch
  }
  bytes
}

///|
/// Reads one exact execution-manifest snapshot for startup-only external
/// approval authentication. The snapshot never escapes before close succeeds.
fn read_execution_manifest_snapshot(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_sha256 : String,
  maximum : Int64,
) -> Bytes raise RuntimeDescriptorFileError {
  let file = root.open_file(locator) catch {
    error => raise map_fs_error(error, Open, maximum)
  }
  let outcome : Result[Bytes, RuntimeDescriptorFileError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=maximum),
  ) catch {
    error => Err(map_fs_error(error, Snapshot, maximum))
  }
  let bytes = finish_descriptor_body(outcome, () => {
    close_descriptor_file(file, maximum)
  })
  if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) != expected_sha256 {
    raise DigestMismatch
  }
  bytes
}
