///|
/// Stable selected-model configuration fields used by payload-safe errors.
pub(all) enum ModelConfigField {
  Architectures
  ModelType
  VocabularySize
  HiddenSize
  IntermediateSize
  LayerCount
  AttentionHeadCount
  KvHeadCount
  ContextLength
  RopeTheta
  TorchDType
  TieWordEmbeddings
  RopeScaling
  SlidingWindow
  HiddenActivation
  RmsNormEpsilon
  AttentionBias
  AttentionDropout
  MlpBias
  HeadDimension
  PretrainingTensorParallelism
  InertMetadata
} derive(Debug, Eq)

///|
/// Executable or remote-code metadata that is never admitted.
pub(all) enum ExecutableField {
  AutoMap
  TrustRemoteCode
  CustomPipelines
} derive(Debug, Eq)

///|
/// Bounded reasons why a selected-model field was rejected.
pub(all) enum ModelConfigValueIssue {
  ExpectedString
  ExpectedInteger
  ExpectedNumber
  ExpectedBoolean
  ExpectedArray
  ExpectedNull
  InvalidArrayElement
  UnsupportedValue
} derive(Debug, Eq)

///|
/// Payload-safe failures from selected-model architecture JSON admission.
///
/// Errors retain neither source text nor attacker-controlled field names or
/// values. All indices and limits are bounded by caller-supplied parse bounds.
pub(all) suberror ModelConfigReaderError {
  InvalidByteLimit
  InvalidDepthLimit
  ByteLimitExceeded(limit~ : Int)
  InvalidUtf8
  MalformedJson
  NonObjectRoot
  NestingLimitExceeded(limit~ : Int)
  DuplicateField(depth~ : Int, index~ : Int)
  UnknownField(index~ : Int)
  ExecutableMetadata(ExecutableField)
  MissingField(ModelConfigField)
  InvalidValue(
    field~ : ModelConfigField,
    issue~ : ModelConfigValueIssue,
    index~ : Int
  )
  Validation(@spec.ModelSpecError)
} derive(Debug, Eq)
