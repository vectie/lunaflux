///|
/// Exact digest-pinned Mistral numeric source. The family is fixed by recipe
/// v7; the optional sliding window is copied as zero-or-positive scalar
/// evidence and is rederived from authenticated config bytes in the child.
pub struct WorkerMistralBootstrapModelSource {
  priv model : WorkerBootstrapModelSource
  priv numeric_weight_artifact_digest : String
  priv weight_manifest_digest : String
  priv sliding_window_tokens : Int
}

///|
pub fn WorkerMistralBootstrapModelSource::new(
  config_locator~ : StringView,
  numeric_weights_locator~ : StringView,
  config_digest_sha256~ : StringView,
  numeric_weight_artifact_sha256~ : StringView,
  weight_manifest_sha256~ : StringView,
  sliding_window_tokens~ : Int,
) -> Self raise WorkerWireError {
  if !bootstrap_source_is_sha256(numeric_weight_artifact_sha256) ||
    !bootstrap_source_is_sha256(weight_manifest_sha256) ||
    sliding_window_tokens < 0 ||
    sliding_window_tokens > 1073741824 {
    raise InvalidBootstrapSource(Digest)
  }
  {
    model: WorkerBootstrapModelSource::new(
      config_locator~,
      weights_locator=numeric_weights_locator,
      config_digest_sha256~,
    ),
    numeric_weight_artifact_digest: numeric_weight_artifact_sha256.to_owned(),
    weight_manifest_digest: weight_manifest_sha256.to_owned(),
    sliding_window_tokens,
  }
}

///|
pub fn WorkerMistralBootstrapModelSource::config_locator(self : Self) -> String {
  self.model.config_locator()
}

///|
pub fn WorkerMistralBootstrapModelSource::numeric_weights_locator(
  self : Self,
) -> String {
  self.model.weights_locator()
}

///|
pub fn WorkerMistralBootstrapModelSource::config_digest(self : Self) -> String {
  self.model.config_digest()
}

///|
pub fn WorkerMistralBootstrapModelSource::numeric_weight_artifact_digest(
  self : Self,
) -> String {
  self.numeric_weight_artifact_digest
}

///|
pub fn WorkerMistralBootstrapModelSource::weight_manifest_digest(
  self : Self,
) -> String {
  self.weight_manifest_digest
}

///|
pub fn WorkerMistralBootstrapModelSource::sliding_window_tokens(
  self : Self,
) -> Int {
  self.sliding_window_tokens
}
