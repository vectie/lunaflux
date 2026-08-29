///|
fn digest(bytes : BytesView) -> ArtifactDigest {
  { hex: @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) }
}

///|
fn filesystem_error(
  error : @approved_fs.ApprovedFsError,
  file_kind : ArtifactFile,
  operation : ArtifactFileOperation,
) -> ArtifactError {
  match error {
    Unavailable(_) => FileUnavailable(file_kind)
    UnsupportedType(_) => UnsupportedPathType(file_kind)
    Invalid(_) => FileIo(file=file_kind, operation~, issue=AuthorityClosed)
    Busy(_) => FileIo(file=file_kind, operation~, issue=Busy)
    Truncated => FileTruncated(file_kind)
    ChangedDuringRead => FileChangedDuringRead(file_kind)
    FileTooLarge => FileIo(file=file_kind, operation~, issue=Failed)
    OperationFailed(_) => FileIo(file=file_kind, operation~, issue=Failed)
  }
}

///|
fn close_file(
  file : @approved_fs.ApprovedFile,
  file_kind : ArtifactFile,
) -> Unit raise ArtifactError {
  file.close() catch {
    error => raise filesystem_error(error, file_kind, CloseFile)
  }
}

///|
fn[T] finish_file_body(
  outcome : Result[T, ArtifactError],
  close : () -> Unit raise ArtifactError,
) -> T raise ArtifactError {
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
fn snapshot_error(
  error : @approved_fs.ApprovedFsError,
  file_kind : ArtifactFile,
) -> ArtifactError {
  match error {
    // This handle was already stamped within both limits. A later too-large
    // result therefore proves growth between stamp and snapshot.
    FileTooLarge => FileChangedDuringRead(file_kind)
    _ => filesystem_error(error, file_kind, SnapshotFile)
  }
}

///|
fn snapshot_bounded(
  file : @approved_fs.ApprovedFile,
  file_kind : ArtifactFile,
  file_maximum : Int64,
  total_remaining : Int64,
) -> Bytes raise ArtifactError {
  let stamp = file.stamp() catch {
    error => raise filesystem_error(error, file_kind, StatFile)
  }
  let size = stamp.byte_count()
  if size < 0L || size > file_maximum {
    raise FileTooLarge(file=file_kind, maximum=file_maximum)
  }
  if size > total_remaining {
    raise TotalTooLarge(maximum=total_remaining)
  }
  file.read_immutable_snapshot(maximum_bytes=size) catch {
    error => raise snapshot_error(error, file_kind)
  }
}

///|
fn read_bounded(
  root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  file_kind : ArtifactFile,
  file_maximum : Int64,
  total_remaining : Int64,
) -> Bytes raise ArtifactError {
  let file = root.open_file(locator) catch {
    error => raise filesystem_error(error, file_kind, OpenFile)
  }
  let outcome : Result[Bytes, ArtifactError] = Ok(
    snapshot_bounded(file, file_kind, file_maximum, total_remaining),
  ) catch {
    error => Err(error)
  }
  finish_file_body(outcome, () => close_file(file, file_kind))
}

///|
fn verify(
  bytes : BytesView,
  expected : ArtifactDigest,
  file : ArtifactFile,
) -> ArtifactDigest raise ArtifactError {
  let actual = digest(bytes)
  if actual != expected {
    raise DigestMismatch(file)
  }
  actual
}

///|
/// Reads and admits a Phase 1 host/reference snapshot of exactly the three
/// explicitly named dense-Llama v1 files.
///
/// Each regular file is sized against both its own and the aggregate limit
/// before allocating owned bytes. The exact returned snapshot is hashed and
/// compared with an independently supplied SHA-256 identity before parsing.
/// Model and tokenizer readers then reject executable metadata and unsupported
/// semantics; safetensors metadata must describe the complete payload.
///
/// This function intentionally owns whole-file host bytes. It must not be used
/// as the production device loader, which streams or maps weights directly to
/// final allocations through the materialization boundary.
pub fn load_reference_bundle(
  approved_root : @approved_fs.ApprovedRoot,
  source : ArtifactSource,
  expected : ExpectedDigests,
  limits : ArtifactLimits,
) -> ReferenceArtifactBundle raise ArtifactError {
  let config_bytes = read_bounded(
    approved_root,
    source.model_config,
    ModelConfig,
    limits.max_model_config_bytes,
    limits.max_total_bytes,
  )
  let config_digest = verify(config_bytes, expected.model_config, ModelConfig)
  let approved_model_content = @spec.ContentDigest::from_sha256(
    expected.weights.hex,
  ) catch {
    _ => raise InvalidDigest
  }
  let model = @config_reader.parse_bytes(
    config_bytes,
    approved_model_content,
    max_bytes=limits.max_model_config_bytes.to_int(),
    max_depth=limits.max_model_config_depth,
  ) catch {
    error => raise InvalidModelConfig(error)
  }
  let remaining_after_config = limits.max_total_bytes -
    config_bytes.length().to_int64()

  let tokenizer_bytes = read_bounded(
    approved_root,
    source.tokenizer,
    Tokenizer,
    limits.max_tokenizer_bytes,
    remaining_after_config,
  )
  let tokenizer_digest = verify(tokenizer_bytes, expected.tokenizer, Tokenizer)
  let tokenizer_identity = @tokenizer.TokenizerDigest::parse(
    tokenizer_digest.hex,
  ) catch {
    _ => raise InvalidDigest
  }
  let tokenizer = @json_reader.parse_bytes(
    tokenizer_bytes,
    digest=tokenizer_identity,
    limits=limits.tokenizer_limits,
    max_bytes=limits.max_tokenizer_bytes.to_int(),
    max_depth=limits.max_tokenizer_depth,
  ) catch {
    error => raise InvalidTokenizer(error)
  }
  if tokenizer.vocab_size() != model.spec().vocabulary_size() {
    raise TokenizerVocabularyMismatch
  }
  let remaining_after_tokenizer = remaining_after_config -
    tokenizer_bytes.length().to_int64()

  let weights_bytes = read_bounded(
    approved_root,
    source.weights,
    Weights,
    limits.max_weights_bytes,
    remaining_after_tokenizer,
  )
  ignore(verify(weights_bytes, expected.weights, Weights))
  let safetensors = @safetensors_reader.read_metadata(
    weights_bytes,
    limits=limits.safetensors_limits,
  ) catch {
    error => raise InvalidSafetensors(error)
  }
  {
    model_config_bytes: config_bytes,
    tokenizer_bytes,
    weights_bytes,
    model,
    tokenizer,
    safetensors,
    identity: {
      model_config: config_digest,
      tokenizer: tokenizer_identity,
      model_content: model.content_digest(),
    },
  }
}
