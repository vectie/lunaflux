///|
const ABSOLUTE_MAX_CONFIG_BYTES : Int64 = 67108864L

///|
const ABSOLUTE_MAX_JSON_DEPTH : Int = 64

///|
/// Independently supplied canonical lowercase SHA-256 identity for one model
/// configuration file.
pub struct ModelConfigDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Exact startup allocation and parser ceilings for one model configuration.
pub struct ModelConfigFileLimits {
  priv maximum_bytes : Int64
  priv maximum_depth : Int
}

///|
pub fn ModelConfigDigest::from_sha256(
  value : String,
) -> ModelConfigDigest raise ModelConfigFileError {
  if value.length() != 64 ||
    !value.all(unit => unit.is_ascii_digit() || ('a' <= unit && unit <= 'f')) {
    raise InvalidExpectedDigest
  }
  { hex: value }
}

///|
pub fn ModelConfigDigest::as_hex(self : ModelConfigDigest) -> String {
  self.hex
}

///|
pub fn ModelConfigFileLimits::new(
  maximum_bytes~ : Int64,
  maximum_depth~ : Int,
) -> ModelConfigFileLimits raise ModelConfigFileError {
  if maximum_bytes <= 0L || maximum_bytes > ABSOLUTE_MAX_CONFIG_BYTES {
    raise InvalidLimit(ConfigBytes)
  }
  if maximum_depth <= 0 || maximum_depth > ABSOLUTE_MAX_JSON_DEPTH {
    raise InvalidLimit(JsonDepth)
  }
  { maximum_bytes, maximum_depth }
}

///|
pub fn ModelConfigFileLimits::maximum_bytes(
  self : ModelConfigFileLimits,
) -> Int64 {
  self.maximum_bytes
}

///|
pub fn ModelConfigFileLimits::maximum_depth(
  self : ModelConfigFileLimits,
) -> Int {
  self.maximum_depth
}
