///|
fn filesystem_error(
  error : @approved_fs.ApprovedFsError,
  operation : ModelConfigFileOperation,
) -> ModelConfigFileError {
  match error {
    Unavailable(_) => FileUnavailable
    UnsupportedType(_) => UnsupportedPathType
    Invalid(_) => FileIo(operation~, issue=AuthorityClosed)
    Busy(_) => FileIo(operation~, issue=Busy)
    Truncated => FileTruncated
    ChangedDuringRead => FileChangedDuringRead
    FileTooLarge => FileIo(operation~, issue=Failed)
    OperationFailed(_) => FileIo(operation~, issue=Failed)
  }
}

///|
fn close_file(
  file : @approved_fs.ApprovedFile,
) -> Unit raise ModelConfigFileError {
  file.close() catch {
    error => raise filesystem_error(error, CloseFile)
  }
}

///|
fn[T] finish_file_body(
  outcome : Result[T, ModelConfigFileError],
  close : () -> Unit raise ModelConfigFileError,
) -> T raise ModelConfigFileError {
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
fn[T] with_config_file(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  body : (@approved_fs.ApprovedFile) -> T raise ModelConfigFileError,
) -> T raise ModelConfigFileError {
  let file = root.open_file(locator) catch {
    error => raise filesystem_error(error, OpenFile)
  }
  let outcome : Result[T, ModelConfigFileError] = Ok(body(file)) catch {
    error => Err(error)
  }
  finish_file_body(outcome, () => close_file(file))
}

///|
fn snapshot_error(
  error : @approved_fs.ApprovedFsError,
  maximum : Int64,
) -> ModelConfigFileError {
  match error {
    FileTooLarge => FileTooLarge(maximum~)
    _ => filesystem_error(error, SnapshotFile)
  }
}

///|
/// Loads one exact digest-pinned dense-Llama configuration beneath a caller-
/// owned approved root. The file is terminally closed before digest-verified
/// bytes can be parsed or metadata published; the root remains caller-owned.
/// The model content digest is independent of the configuration-file digest.
pub fn load(
  approved_root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : ModelConfigDigest,
  model_content_digest : @spec.ContentDigest,
  limits : ModelConfigFileLimits,
) -> @spec.LlamaModelMetadata raise ModelConfigFileError {
  let bytes = with_config_file(approved_root, locator, file => {
    file.read_immutable_snapshot(maximum_bytes=limits.maximum_bytes) catch {
      error => raise snapshot_error(error, limits.maximum_bytes)
    }
  })
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected_digest.hex {
    raise DigestMismatch
  }
  @config_reader.parse_bytes(
    bytes,
    model_content_digest,
    max_bytes=limits.maximum_bytes.to_int(),
    max_depth=limits.maximum_depth,
  ) catch {
    error => raise InvalidConfiguration(error)
  }
}
