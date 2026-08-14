///|
const MAX_OWNED_FILE_BYTES : Int64 = 2147483647L

///|
fn valid_sha256(hex : StringView) -> Bool {
  hex.length() == 64 &&
  hex.all(c => c.is_ascii_digit() || ('a' <= c && c <= 'f'))
}

///|
/// Validates a canonical lowercase hexadecimal SHA-256 identity.
pub fn ArtifactDigest::parse(
  hex : String,
) -> ArtifactDigest raise ArtifactError {
  if !valid_sha256(hex) {
    raise InvalidDigest
  }
  { hex, }
}

///|
pub fn ArtifactDigest::as_hex(self : ArtifactDigest) -> String {
  self.hex
}

///|
/// Constructs a distinct, explicit three-file source beneath one separately
/// owned approved root.
pub fn ArtifactSource::new(
  model_config~ : StringView,
  tokenizer~ : StringView,
  weights~ : StringView,
) -> ArtifactSource raise ArtifactError {
  let model_config_locator = @approved_fs.ApprovedRelativeLocator::new(
    model_config,
  ) catch {
    _ => raise InvalidLocator(ModelConfig)
  }
  let tokenizer_locator = @approved_fs.ApprovedRelativeLocator::new(tokenizer) catch {
    _ => raise InvalidLocator(Tokenizer)
  }
  let weights_locator = @approved_fs.ApprovedRelativeLocator::new(weights) catch {
    _ => raise InvalidLocator(Weights)
  }
  if model_config == tokenizer {
    raise DuplicateLocator(Tokenizer)
  }
  if model_config == weights || tokenizer == weights {
    raise DuplicateLocator(Weights)
  }
  {
    model_config: model_config_locator,
    tokenizer: tokenizer_locator,
    weights: weights_locator,
  }
}

///|
/// Constructs an approved digest set for all three files.
pub fn ExpectedDigests::new(
  model_config~ : ArtifactDigest,
  tokenizer~ : ArtifactDigest,
  weights~ : ArtifactDigest,
) -> ExpectedDigests {
  { model_config, tokenizer, weights }
}

///|
fn validate_byte_limit(
  value : Int64,
  field : ArtifactLimit,
) -> Unit raise ArtifactError {
  if value <= 0L || value > MAX_OWNED_FILE_BYTES {
    raise InvalidLimit(field)
  }
}

///|
/// Constructs the complete allocation and parser envelope for admission.
pub fn ArtifactLimits::new(
  max_model_config_bytes~ : Int64,
  max_tokenizer_bytes~ : Int64,
  max_weights_bytes~ : Int64,
  max_total_bytes~ : Int64,
  max_model_config_depth~ : Int,
  max_tokenizer_depth~ : Int,
  tokenizer_limits~ : @tokenizer.TokenizerLimits,
  safetensors_limits~ : @safetensors.HeaderLimits,
) -> ArtifactLimits raise ArtifactError {
  validate_byte_limit(max_model_config_bytes, ModelConfigBytes)
  validate_byte_limit(max_tokenizer_bytes, TokenizerBytes)
  validate_byte_limit(max_weights_bytes, WeightsBytes)
  validate_byte_limit(max_total_bytes, TotalBytes)
  if max_model_config_depth <= 0 {
    raise InvalidLimit(ModelConfigDepth)
  }
  if max_tokenizer_depth <= 0 {
    raise InvalidLimit(TokenizerDepth)
  }
  {
    max_model_config_bytes,
    max_tokenizer_bytes,
    max_weights_bytes,
    max_total_bytes,
    max_model_config_depth,
    max_tokenizer_depth,
    tokenizer_limits,
    safetensors_limits,
  }
}

///|
pub fn ExpectedDigests::model_config(self : ExpectedDigests) -> ArtifactDigest {
  self.model_config
}

///|
pub fn ExpectedDigests::tokenizer(self : ExpectedDigests) -> ArtifactDigest {
  self.tokenizer
}

///|
pub fn ExpectedDigests::weights(self : ExpectedDigests) -> ArtifactDigest {
  self.weights
}
