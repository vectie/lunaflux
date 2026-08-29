///|
fn map_fs_error(
  error : @approved_fs.ApprovedFsError,
  operation : TokenizerFileOperation,
  maximum : Int64,
) -> TokenizerFileError {
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
fn finish_tokenizer_file(
  outcome : Result[Bytes, TokenizerFileError],
  close : () -> Unit raise TokenizerFileError,
) -> Bytes raise TokenizerFileError {
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

///| Load one exact tokenizer below a caller-owned identity-bound model root.

///|
/// The file is terminally closed before parsing and neither root nor file
/// authority survives in the returned evidence.
pub fn load(
  root : @approved_fs.ApprovedRoot,
  root_label : StringView,
  locator_text : StringView,
  expected_digest : @tokenizer.TokenizerDigest,
  file_limits : TokenizerFileLimits,
  tokenizer_limits : @tokenizer.TokenizerLimits,
) -> TokenizerFileAdmission raise TokenizerFileError {
  let maximum = file_limits.max_file_bytes().to_int64()
  root.require_absolute_identity(root_label) catch {
    error => raise map_fs_error(error, BindRoot, maximum)
  }
  load_bound_tokenizer(
    root, root_label, locator_text, expected_digest, file_limits, tokenizer_limits,
  )
}

///|
fn load_bound_tokenizer(
  root : @approved_fs.ApprovedRoot,
  root_label : StringView,
  locator_text : StringView,
  expected_digest : @tokenizer.TokenizerDigest,
  file_limits : TokenizerFileLimits,
  tokenizer_limits : @tokenizer.TokenizerLimits,
) -> TokenizerFileAdmission raise TokenizerFileError {
  let maximum = file_limits.max_file_bytes().to_int64()
  let locator = @approved_fs.ApprovedRelativeLocator::new(locator_text) catch {
    error => raise map_fs_error(error, ValidateLocator, maximum)
  }
  let file = root.open_file(locator) catch {
    error => raise map_fs_error(error, Open, maximum)
  }
  let outcome : Result[Bytes, TokenizerFileError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=maximum),
  ) catch {
    error => Err(map_fs_error(error, Snapshot, maximum))
  }
  let bytes = finish_tokenizer_file(outcome, () => {
    file.close() catch {
      error => raise map_fs_error(error, Close, maximum)
    }
  })
  let actual = @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
  if actual != expected_digest.as_string() {
    raise DigestMismatch
  }
  let spec = @json_reader.parse_bytes(
    bytes,
    digest=expected_digest,
    limits=tokenizer_limits,
    max_bytes=file_limits.max_file_bytes(),
    max_depth=file_limits.max_json_depth(),
  ) catch {
    _ => raise InvalidTokenizer
  }
  {
    root_label: root_label.to_owned(),
    locator: locator_text.to_owned(),
    digest: expected_digest,
    file_limits,
    tokenizer_limits,
    spec,
  }
}

///|
/// Loads one exact tokenizer from a pinned materialization source while the
/// returned source evidence retains the deployment target label.
pub fn load_materialized(
  model : @approved_fs.ApprovedRootMaterializationView,
  locator_text : StringView,
  expected_digest : @tokenizer.TokenizerDigest,
  file_limits : TokenizerFileLimits,
  tokenizer_limits : @tokenizer.TokenizerLimits,
) -> TokenizerFileAdmission raise TokenizerFileError {
  load_bound_tokenizer(
    model.source_root(),
    model.target_label(),
    locator_text,
    expected_digest,
    file_limits,
    tokenizer_limits,
  )
}
