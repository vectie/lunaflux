///|
pub fn ArtifactIdentity::model_config(
  self : ArtifactIdentity,
) -> ArtifactDigest {
  self.model_config
}

///|
pub fn ArtifactIdentity::tokenizer(
  self : ArtifactIdentity,
) -> @tokenizer.TokenizerDigest {
  self.tokenizer
}

///|
pub fn ArtifactIdentity::model(self : ArtifactIdentity) -> @spec.ModelIdentity {
  self.model
}

///|
pub fn ReferenceArtifactBundle::model_config_bytes(
  self : ReferenceArtifactBundle,
) -> Bytes {
  self.model_config_bytes
}

///|
pub fn ReferenceArtifactBundle::tokenizer_bytes(
  self : ReferenceArtifactBundle,
) -> Bytes {
  self.tokenizer_bytes
}

///|
pub fn ReferenceArtifactBundle::weights_bytes(
  self : ReferenceArtifactBundle,
) -> Bytes {
  self.weights_bytes
}

///|
pub fn ReferenceArtifactBundle::model(
  self : ReferenceArtifactBundle,
) -> @spec.LlamaModelMetadata {
  self.model
}

///|
pub fn ReferenceArtifactBundle::tokenizer(
  self : ReferenceArtifactBundle,
) -> @tokenizer.TokenizerSpec {
  self.tokenizer
}

///|
pub fn ReferenceArtifactBundle::safetensors(
  self : ReferenceArtifactBundle,
) -> @safetensors.SafetensorsMetadata {
  self.safetensors
}

///|
pub fn ReferenceArtifactBundle::identity(
  self : ReferenceArtifactBundle,
) -> ArtifactIdentity {
  self.identity
}
