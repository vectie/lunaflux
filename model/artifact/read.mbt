///|
fn digest(bytes : BytesView) -> ArtifactDigest {
  { hex: @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) }
}

///|
async fn read_bounded(
  path : StringView,
  file_kind : ArtifactFile,
  file_maximum : Int64,
  total_remaining : Int64,
) -> Bytes {
  let path_kind = @fs.kind(path, follow_symlink=false) catch {
    _ => raise FileUnavailable(file_kind)
  }
  if path_kind != Regular {
    raise UnsupportedPathType(file_kind)
  }
  let file = @fs.open(path, mode=ReadOnly) catch {
    _ => raise FileUnavailable(file_kind)
  }
  defer file.close()
  if file.kind() != Regular {
    raise UnsupportedPathType(file_kind)
  }
  let size = file.size() catch { _ => raise FileUnavailable(file_kind) }
  if size < 0L || size > file_maximum {
    raise FileTooLarge(file=file_kind, maximum=file_maximum)
  }
  if size > total_remaining {
    raise TotalTooLarge(maximum=total_remaining)
  }
  let length = size.to_int()
  let bytes = file.read_exactly_at(length, position=0L) catch {
    _ => raise FileTruncated(file_kind)
  }
  let probe = FixedArray::make(1, b'\x00')
  let trailing = file.read_at(probe, position=size, offset=0, len=1) catch {
    _ => raise FileChangedDuringRead(file_kind)
  }
  if trailing != 0 {
    raise FileChangedDuringRead(file_kind)
  }
  bytes
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
pub async fn load_reference_bundle(
  paths : ArtifactPaths,
  expected : ExpectedDigests,
  limits : ArtifactLimits,
) -> ReferenceArtifactBundle {
  let config_bytes = read_bounded(
    paths.model_config,
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
    paths.tokenizer,
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
  let remaining_after_tokenizer = remaining_after_config -
    tokenizer_bytes.length().to_int64()

  let weights_bytes = read_bounded(
    paths.weights,
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
      model: model.identity(),
    },
  }
}
