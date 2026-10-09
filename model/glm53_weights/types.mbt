///|
/// Closed official GLM-5.3 checkpoint encodings.
pub(all) enum Glm53WeightEncoding {
  BFloat16WithFloat32Routing
  Fp8E4M3Block128Dynamic
} derive(Debug, Eq)

///|
/// Text-layer tensor roles whose shapes are fixed by the architecture.
pub(all) enum Glm53TextTensorRole {
  InputNorm
  PostAttentionNorm
  DenseGate
  DenseUp
  DenseDown
  MoeRouter
  MoeCorrectionBias
  RoutedGateUp
  RoutedDown
  SharedGate
  SharedUp
  SharedDown
  QueryA
  QueryANorm
  QueryB
  KeyValueA
  KeyValueANorm
  KeyValueB
  AttentionOutput
  IndexQuery
  IndexKey
  IndexKeyNorm
  IndexKeyNormBias
  IndexWeights
  IndexPoolPosition
  IndexPoolGate
  LinearQuery
  LinearKey
  LinearValue
  LinearConvolution
  LinearBeta
  LinearGateA
  LinearGateB
  LinearOutputNorm
  LinearOutput
  LinearForgetA
  LinearForgetB
  LinearForgetBias
  LinearForgetLogA
  HyperAttentionFunction
  HyperAttentionBase
  HyperAttentionScale
  HyperFfnFunction
  HyperFfnBase
  HyperFfnScale
} derive(Debug, Eq)

///|
pub(all) enum Glm53GlobalTensorRole {
  TokenEmbedding
  FinalNorm
  LanguageModelHead
} derive(Debug, Eq)

///|
pub(all) suberror Glm53WeightsError {
  InvalidLayerIndex
  TensorRoleNotPresent(Glm53TextTensorRole)
  UnsupportedPhysicalEncoding(Glm53WeightEncoding)
  UnsupportedSafetensorsDType(ordinal~ : Int)
  TensorCountMismatch(expected~ : Int, actual~ : Int)
  DuplicateTensor(ordinal~ : Int)
  TensorNameMismatch(ordinal~ : Int)
  TensorDTypeMismatch(ordinal~ : Int)
  TensorShapeMismatch(ordinal~ : Int)
} derive(Debug, Eq)

///|
/// Startup-only exact weight layout. It grants no file, allocation, plan, or
/// execution authority.
pub struct Glm53WeightSchema {
  priv profile : @glm53_spec.Glm53ProfileId
  priv content_digest : @spec.ContentDigest
  priv encoding : Glm53WeightEncoding
} derive(Debug, Eq)

///|
pub fn Glm53WeightSchema::profile(self : Self) -> @glm53_spec.Glm53ProfileId {
  self.profile
}

///|
pub fn Glm53WeightSchema::content_digest(self : Self) -> @spec.ContentDigest {
  self.content_digest
}

///|
pub fn Glm53WeightSchema::encoding(self : Self) -> Glm53WeightEncoding {
  self.encoding
}

///|
pub fn Glm53WeightSchema::fp8_block_rows(self : Self) -> Int? {
  match self.encoding {
    BFloat16WithFloat32Routing => None
    Fp8E4M3Block128Dynamic => Some(128)
  }
}

///|
pub fn Glm53WeightSchema::fp8_block_columns(self : Self) -> Int? {
  self.fp8_block_rows()
}
