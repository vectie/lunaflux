///|
const MAX_PATH_CODE_UNITS : Int = 4096

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
fn validate_path(
  path : StringView,
  file : ArtifactFile,
) -> Unit raise ArtifactError {
  if path.is_empty() {
    raise InvalidPath(file~, issue=Empty)
  }
  if path.length() > MAX_PATH_CODE_UNITS {
    raise InvalidPath(file~, issue=TooLong)
  }
}

///|
/// Constructs a distinct, explicit three-file path set.
pub fn ArtifactPaths::new(
  model_config~ : String,
  tokenizer~ : String,
  weights~ : String,
) -> ArtifactPaths raise ArtifactError {
  validate_path(model_config, ModelConfig)
  validate_path(tokenizer, Tokenizer)
  validate_path(weights, Weights)
  if model_config == tokenizer {
    raise InvalidPath(file=Tokenizer, issue=Duplicate)
  }
  if model_config == weights || tokenizer == weights {
    raise InvalidPath(file=Weights, issue=Duplicate)
  }
  { model_config, tokenizer, weights }
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
pub fn ArtifactPaths::model_config(self : ArtifactPaths) -> String {
  self.model_config
}

///|
pub fn ArtifactPaths::tokenizer(self : ArtifactPaths) -> String {
  self.tokenizer
}

///|
pub fn ArtifactPaths::weights(self : ArtifactPaths) -> String {
  self.weights
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
