///|
/// SHA-256 identity of immutable model artifact content.
pub struct ContentDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Syntactically validated SHA-256 claim about an immutable model plan.
///
/// This wire/persistence vocabulary carries no semantic authority by itself.
/// A trusted plan digest must be compared with one minted by `ModelPlan`.
pub struct PlanDigest {
  priv hex : String
} derive(Debug, Eq)

///|
/// Syntactically validated content/plan identity claim for wire persistence.
///
/// Construction proves only digest syntax. Execution admission must compare
/// the claim exactly with an identity minted by a rebuilt `ModelPlan`.
pub struct ModelIdentity {
  priv content : ContentDigest
  priv plan : PlanDigest
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
  let digest : ContentDigest = { hex, }
  digest.validate()
  digest
}

///|
fn ContentDigest::validate(self : ContentDigest) -> Unit raise ModelSpecError {
  if !is_lower_sha256(self.hex) {
    raise InvalidDigest
  }
}

///|
/// Returns the canonical lowercase SHA-256 representation.
pub fn ContentDigest::as_hex(self : ContentDigest) -> String {
  self.hex
}

///|
/// Validates lowercase SHA-256 syntax without authenticating plan semantics.
pub fn PlanDigest::from_sha256(hex : String) -> PlanDigest raise ModelSpecError {
  let digest : PlanDigest = { hex, }
  digest.validate()
  digest
}

///|
fn PlanDigest::validate(self : PlanDigest) -> Unit raise ModelSpecError {
  if !is_lower_sha256(self.hex) {
    raise InvalidDigest
  }
}

///|
/// Returns the canonical lowercase SHA-256 representation.
pub fn PlanDigest::as_hex(self : PlanDigest) -> String {
  self.hex
}

///|
/// Combines validated digest text without authenticating plan semantics.
pub fn ModelIdentity::new(
  content : ContentDigest,
  plan : PlanDigest,
) -> ModelIdentity raise ModelSpecError {
  let identity : ModelIdentity = { content, plan }
  identity.validate()
  identity
}

///|
pub fn ModelIdentity::content(self : ModelIdentity) -> ContentDigest {
  self.content
}

///|
pub fn ModelIdentity::plan(self : ModelIdentity) -> PlanDigest {
  self.plan
}

///|
fn ModelIdentity::validate(self : ModelIdentity) -> Unit raise ModelSpecError {
  self.content.validate()
  self.plan.validate()
}
