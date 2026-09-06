///|
pub(all) suberror ProjectionTuningError {
  InvalidSnapshot
  ScopeMismatch
} derive(Debug, Eq)

///|
/// Offline build scope, not device authority. The device identity is opaque to
/// the functional compiler; CUDA UUIDs belong to the caller's backend adapter.
pub struct ProjectionTuningScope {
  priv backend : @strategy.ProjectionBackendId
  priv architecture : @strategy.ProjectionTargetId
  priv device_identity : String
  priv toolchain_sha256 : String
} derive(Debug, Eq)

///|
pub struct ProjectionTuningSnapshot {
  priv scope : ProjectionTuningScope
  priv records : ReadOnlyArray[@strategy.ProjectionAutotuneRecord]
  priv canonical : Bytes
}

///|
fn valid_label(value : StringView) -> Bool {
  value.length() > 0 &&
  value.length() <= 128 &&
  value
  .iter()
  .all(c => c is ('a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | ':'))
}

///|
pub fn ProjectionTuningScope::new(
  backend~ : @strategy.ProjectionBackendId,
  architecture~ : @strategy.ProjectionTargetId,
  device_identity~ : String,
  toolchain_sha256~ : String,
) -> ProjectionTuningScope raise ProjectionTuningError {
  if !valid_label(backend.stable_name()) ||
    !valid_label(architecture.stable_name()) ||
    !valid_label(device_identity) ||
    toolchain_sha256.length() != 64 ||
    !toolchain_sha256.iter().all(c => c is ('0'..='9' | 'a'..='f')) {
    raise InvalidSnapshot
  }
  { backend, architecture, device_identity, toolchain_sha256 }
}

///|
pub fn ProjectionTuningSnapshot::canonical_bytes(
  self : ProjectionTuningSnapshot,
) -> Bytes {
  self.canonical
}

///|
pub fn ProjectionTuningSnapshot::records_for(
  self : ProjectionTuningSnapshot,
  expected : ProjectionTuningScope,
) -> ReadOnlyArray[@strategy.ProjectionAutotuneRecord] raise ProjectionTuningError {
  if self.scope != expected {
    raise ScopeMismatch
  }
  self.records
}
