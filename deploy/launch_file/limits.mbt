///|
pub fn LaunchFileLimits::new(
  maximum_bytes~ : Int64,
  maximum_json_depth~ : Int,
) -> LaunchFileLimits raise LaunchFileError {
  if maximum_bytes <= 0L ||
    maximum_bytes > 1048576L ||
    maximum_json_depth <= 0 ||
    maximum_json_depth > 16 {
    raise InvalidLimit
  }
  { maximum_bytes, maximum_json_depth }
}

///|
pub fn LaunchFileLimits::maximum_bytes(self : LaunchFileLimits) -> Int64 {
  self.maximum_bytes
}

///|
pub fn LaunchFileLimits::maximum_json_depth(self : LaunchFileLimits) -> Int {
  self.maximum_json_depth
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
fn LaunchFileDigest::from_sha256(
  value : StringView,
) -> LaunchFileDigest raise LaunchFileError {
  if !is_lower_sha256(value) {
    raise InvalidExpectedDigest
  }
  LaunchFileDigest(value.to_owned())
}

///|
pub fn LaunchFileDigest::as_hex(self : LaunchFileDigest) -> String {
  self.0
}
