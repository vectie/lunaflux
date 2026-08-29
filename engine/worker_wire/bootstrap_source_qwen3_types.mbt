///|
/// Exact digest-pinned dense-Qwen3 BF16 numeric source. Recipe v12 fixes the
/// model family, paged execution mode, and tied embedding/output storage.
pub struct WorkerQwen3BootstrapModelSource {
  priv model : WorkerBootstrapModelSource
  priv numeric_weight_artifact_digest : String
  priv weight_manifest_digest : String
}

///|
pub fn WorkerQwen3BootstrapModelSource::new(
  config_locator~ : StringView,
  numeric_weights_locator~ : StringView,
  config_digest_sha256~ : StringView,
  numeric_weight_artifact_sha256~ : StringView,
  weight_manifest_sha256~ : StringView,
  tied_embeddings~ : Bool,
) -> Self raise WorkerWireError {
  if !bootstrap_source_is_sha256(numeric_weight_artifact_sha256) ||
    !bootstrap_source_is_sha256(weight_manifest_sha256) ||
    !tied_embeddings {
    raise InvalidBootstrapSource(Recipe)
  }
  {
    model: WorkerBootstrapModelSource::new(
      config_locator~,
      weights_locator=numeric_weights_locator,
      config_digest_sha256~,
    ),
    numeric_weight_artifact_digest: numeric_weight_artifact_sha256.to_owned(),
    weight_manifest_digest: weight_manifest_sha256.to_owned(),
  }
}

///|
pub fn WorkerQwen3BootstrapModelSource::config_locator(self : Self) -> String {
  self.model.config_locator()
}

///|
pub fn WorkerQwen3BootstrapModelSource::numeric_weights_locator(
  self : Self,
) -> String {
  self.model.weights_locator()
}

///|
pub fn WorkerQwen3BootstrapModelSource::config_digest(self : Self) -> String {
  self.model.config_digest()
}

///|
pub fn WorkerQwen3BootstrapModelSource::numeric_weight_artifact_digest(
  self : Self,
) -> String {
  self.numeric_weight_artifact_digest
}

///|
pub fn WorkerQwen3BootstrapModelSource::weight_manifest_digest(
  self : Self,
) -> String {
  self.weight_manifest_digest
}

///|
pub fn WorkerQwen3BootstrapModelSource::tied_embeddings(_self : Self) -> Bool {
  true
}
