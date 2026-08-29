///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  let mut nonzero = false
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
    if unit != '0' {
      nonzero = true
    }
  }
  nonzero
}

///|
pub fn DeploymentBundleDigest::from_sha256(
  value : StringView,
) -> DeploymentBundleDigest raise DeploymentBundleError {
  if !is_lower_sha256(value) {
    raise InvalidExpectedDigest
  }
  DeploymentBundleDigest(value.to_owned())
}

///|
pub fn DeploymentBundleDigest::as_hex(self : DeploymentBundleDigest) -> String {
  self.0
}
