///|
/// Weight-bearing partition in the released MiniMax H3 pipeline.
///
/// FL2VA and Ref2VA denoisers are separate even though their schema-v1 tensor
/// names coincide: their authenticated content and conditioning semantics do
/// not. Text and VAE components are shared semantic roles.
pub(all) enum MiniMaxH3WeightComponent {
  Fl2vaDenoiser
  Ref2vaDenoiser
  TextConditioner
  VideoVae
  AudioVae
} derive(Debug, Eq)

///|
/// Operator-level role assigned by the schema-v1 vocabulary. This role is
/// deliberately independent of a device address or materialized tensor.
pub(all) enum MiniMaxH3TensorRole {
  DenoiserProjection
  DenoiserTimeEmbedding
  DenoiserAttention
  DenoiserFeedForward
  DenoiserNormalization
  TextEmbedding
  TextAttention
  TextFeedForward
  TextNormalization
  VisionPatchEmbedding
  VisionAttention
  VisionFeedForward
  VisionNormalization
  VideoEncoder
  VideoDecoder
  VideoLatentProjection
  AudioEncoder
  AudioDecoder
  AudioLatentProjection
} derive(Debug, Eq)

///|
/// Request-conditioning paths which consume one bound weight partition.
pub(all) enum MiniMaxH3ConditioningUse {
  PromptConditioning
  Fl2vaKeyframeConditioning
  Ref2vaImageConditioning
  Ref2vaVideoConditioning
  Ref2vaAudioConditioning
} derive(Debug, Eq)

///|
/// Payload-safe MiniMax H3 weight admission failure.
pub(all) suberror MiniMaxH3WeightsError {
  EmptyComponentShard(MiniMaxH3WeightComponent)
  ComponentVariantMismatch(MiniMaxH3WeightComponent)
  PlanContentMismatch
  PlanSemanticMismatch
  UnknownSemanticTensor(index~ : Int32)
  InvalidSemanticRank(index~ : Int32)
  InvalidSemanticDType(index~ : Int32)
  UnsupportedRawLayout(index~ : Int32)
  BindingComponentMismatch(index~ : Int32)
  BindingIdentityMismatch(index~ : Int32)
  DuplicateSemanticTensor(index~ : Int32)
  IncompleteComponent(expected~ : Int32, actual~ : Int32)
  MaterializationUnavailable(MiniMaxH3WeightComponent)
} derive(Debug, Eq)

///|
/// Immutable reference to validated safetensors metadata and its assigned
/// MiniMax H3 operator role. No payload bytes or device pointers are retained.
pub struct MiniMaxH3SemanticTensorBinding {
  priv source_index : Int32
  priv role : MiniMaxH3TensorRole
  priv tensor : @safetensors.TensorMetadata
} derive(Debug, Eq)

///|
/// One component shard bound to exact model content and execution-plan
/// identity. Completeness across all shards remains an artifact-manifest
/// concern; every tensor present in this shard has been classified.
pub struct MiniMaxH3PartialShardBinding {
  priv component : MiniMaxH3WeightComponent
  priv identity : @spec.ModelIdentity
  priv schema_version : Int32
  priv tensors : ReadOnlyArray[MiniMaxH3SemanticTensorBinding]
} derive(Debug, Eq)

///|
/// Exact all-shard component admission. Construction proves that every name
/// in the family-owned source vocabulary occurs exactly once.
pub struct MiniMaxH3CompleteComponentBinding {
  priv component : MiniMaxH3WeightComponent
  priv identity : @spec.ModelIdentity
  priv schema_version : Int32
  priv tensors : ReadOnlyArray[MiniMaxH3SemanticTensorBinding]
} derive(Debug, Eq)
