///|
const ABSOLUTE_MAX_POLICY_BYTES : Int64 = 1048576L

///|
const ABSOLUTE_MAX_JSON_DEPTH : Int = 16

///|
/// Validate an independently supplied policy digest.
pub fn InstancePolicyDigest::from_sha256(
  value : StringView,
) -> InstancePolicyDigest raise InstancePolicyFileError {
  if value.length() != 64 {
    raise InvalidExpectedDigest
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      raise InvalidExpectedDigest
    }
  }
  InstancePolicyDigest(value.to_owned())
}

///|
/// Return the validated digest text.
pub fn InstancePolicyDigest::as_hex(self : InstancePolicyDigest) -> String {
  self.0
}

///|
/// Construct bounded policy-file scan limits.
pub fn InstancePolicyFileLimits::new(
  max_policy_bytes~ : Int64,
  max_json_depth~ : Int,
) -> InstancePolicyFileLimits raise InstancePolicyFileError {
  if max_policy_bytes <= 0L || max_policy_bytes > ABSOLUTE_MAX_POLICY_BYTES {
    raise InvalidLimit(PolicyBytes)
  }
  if max_json_depth <= 0 || max_json_depth > ABSOLUTE_MAX_JSON_DEPTH {
    raise InvalidLimit(JsonDepth)
  }
  { max_policy_bytes, max_json_depth }
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyFileLimits::max_policy_bytes(
  self : InstancePolicyFileLimits,
) -> Int64 {
  self.max_policy_bytes
}

///|
/// Return or validate one exact admitted policy field.
pub fn InstancePolicyFileLimits::max_json_depth(
  self : InstancePolicyFileLimits,
) -> Int {
  self.max_json_depth
}
