///|
/// SHA-256 identity of immutable model artifact content.
pub struct ContentDigest {
  hex : String
} derive(Debug, Eq)

///|
/// SHA-256 identity of a canonical immutable model plan.
pub struct PlanDigest {
  hex : String
} derive(Debug, Eq)

///|
/// Pair of identities that salts model- and plan-specific runtime state.
pub struct ModelIdentity {
  content : ContentDigest
  plan : PlanDigest
} derive(Debug, Eq)

///|
fn is_lower_sha256(hex : StringView) -> Bool {
  hex.length() == 64 &&
  hex.all(c => c.is_ascii_digit() || ('a' <= c && c <= 'f'))
}

///|
/// Validates a lowercase hexadecimal SHA-256 content identity.
pub fn ContentDigest::from_sha256(
  hex : String,
) -> ContentDigest raise ModelSpecError {
  if !is_lower_sha256(hex) {
    raise InvalidDigest
  }
  { hex, }
}

///|
/// Returns the canonical lowercase SHA-256 representation.
pub fn ContentDigest::as_hex(self : ContentDigest) -> String {
  self.hex
}

///|
/// Validates a lowercase hexadecimal SHA-256 plan identity.
pub fn PlanDigest::from_sha256(hex : String) -> PlanDigest raise ModelSpecError {
  if !is_lower_sha256(hex) {
    raise InvalidDigest
  }
  { hex, }
}

///|
/// Returns the canonical lowercase SHA-256 representation.
pub fn PlanDigest::as_hex(self : PlanDigest) -> String {
  self.hex
}

///|
/// Constructs an immutable model identity from independently verified digests.
pub fn ModelIdentity::new(
  content : ContentDigest,
  plan : PlanDigest,
) -> ModelIdentity {
  { content, plan }
}

///|
pub fn ModelIdentity::content(self : ModelIdentity) -> ContentDigest {
  self.content
}

///|
pub fn ModelIdentity::plan(self : ModelIdentity) -> PlanDigest {
  self.plan
}
