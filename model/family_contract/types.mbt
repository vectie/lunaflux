///|
/// Stable model-family identity used only during startup admission.
///
/// Runtime scheduling and KV ownership consume an admitted model plan and do
/// not inspect this value.
pub(all) enum ModelFamilyId {
  Llama
  Mistral
  Qwen3
  DeepSeekV4
  MiniMaxH3
  Glm53
} derive(Debug, Eq)

///|
/// Top-level execution topology required by a model family.
pub(all) enum ModelWorkload {
  DecoderOnlyText
  MultimodalConditionalGeneration
  AudioVideoDiffusion
} derive(Debug, Eq)

///|
/// Architecture-neutral semantic capabilities that a model plan/backend may
/// implement. These are startup capability names, never kernel IDs or request-
/// path dispatch tags.
pub(all) enum ModelSemanticCapability {
  DenseCausalAttention
  QueryKeyRmsNorm
  MixtureOfExperts
  MultiTokenPrediction
  HyperConnections
  CompressedSparseAttention
  HeavilyCompressedAttention
  DynamicSparseAttention
  LinearRecurrentAttention
  VisionEncoder
  MultimodalProjection
  DiffusionDenoising
  VideoVae
  AudioVae
  ReferenceConditioning
  TextConditioningEncoder
} derive(Debug, Eq)

///|
/// Invalid construction of a startup-only family capability contract.
pub(all) suberror FamilyContractError {
  EmptyArchitecture
  EmptyModelType
  InvalidSchemaVersion
  DuplicateCapability(ModelSemanticCapability)
} derive(Debug, Eq)

///|
/// Immutable requirements for one exact, versioned model-family profile.
pub struct ModelFamilyRequirements {
  priv family : ModelFamilyId
  priv architecture : String
  priv model_type : String
  priv schema_version : Int
  priv workload : ModelWorkload
  priv capabilities : ReadOnlyArray[ModelSemanticCapability]
} derive(Debug, Eq)

///|
/// Immutable set of semantic capabilities supplied by one execution backend.
pub struct ModelBackendCapabilities {
  priv supported : FixedArray[Bool]
} derive(Debug, Eq)

///|
/// Result of joining a family profile with one semantic backend declaration at
/// startup. This is not artifact, device, or execution readiness.
pub(all) enum ModelFamilyAdmission {
  SemanticCapabilitiesAvailable(ModelFamilyRequirements)
  MissingCapabilities(
    ModelFamilyRequirements,
    ReadOnlyArray[ModelSemanticCapability]
  )
} derive(Debug, Eq)
