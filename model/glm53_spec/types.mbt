///|
/// The two official GLM-5.3 architecture profiles have different execution
/// semantics and must never be selected by a shared family-name branch.
pub(all) enum Glm53ProfileId {
  FullText
  FlashMultimodal
} derive(Debug, Eq)

///|
/// Per-layer attention semantics declared by an exact GLM-5.3 profile.
pub(all) enum Glm53AttentionKind {
  DeepSeekSparseAttention
  KimiDeltaAttention
} derive(Debug, Eq)

///|
/// Per-layer feed-forward semantics declared by an exact GLM-5.3 profile.
pub(all) enum Glm53FeedForwardKind {
  DenseSwiGlu
  RoutedMixtureOfExperts
} derive(Debug, Eq)

///|
/// Payload-safe profile validation failures.
pub(all) suberror Glm53SpecError {
  InvalidLayerIndex
} derive(Debug, Eq)

///|
/// Independently authenticated metadata for the full text-only profile.
pub struct Glm53FullMetadata {
  priv spec : Glm53FullSpec
  priv content_digest : @spec.ContentDigest
} derive(Debug, Eq)

///|
/// Independently authenticated metadata for the Flash multimodal profile.
pub struct Glm53FlashMetadata {
  priv spec : Glm53FlashSpec
  priv content_digest : @spec.ContentDigest
} derive(Debug, Eq)

///|
pub fn Glm53FullMetadata::from_verified_content(
  spec : Glm53FullSpec,
  content_digest : @spec.ContentDigest,
) -> Glm53FullMetadata {
  { spec, content_digest, }
}

///|
pub fn Glm53FullMetadata::spec(self : Self) -> Glm53FullSpec {
  self.spec
}

///|
pub fn Glm53FullMetadata::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn Glm53FlashMetadata::from_verified_content(
  spec : Glm53FlashSpec,
  content_digest : @spec.ContentDigest,
) -> Glm53FlashMetadata {
  { spec, content_digest, }
}

///|
pub fn Glm53FlashMetadata::spec(self : Self) -> Glm53FlashSpec {
  self.spec
}

///|
pub fn Glm53FlashMetadata::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}
