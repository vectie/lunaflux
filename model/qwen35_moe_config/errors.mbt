///|
/// Bounded area vocabulary for Qwen3.5-MoE config diagnostics.
pub(all) enum Qwen35ConfigArea {
  Root
  Text
  Rope
  Quantization
} derive(Debug, Eq)

///|
/// Bounded field vocabulary for Qwen3.5-MoE config diagnostics.
pub(all) enum Qwen35ConfigField {
  Architectures
  RootModelType
  TextConfig
  QuantizationConfig
  VisionConfig
  TieWordEmbeddings
  TokenId
  TransformersVersion
  TextModelType
  WeightDType
  VocabularySize
  HiddenSize
  LayerCount
  LayerTypes
  ContextLength
  AttentionHeadCount
  KvHeadCount
  HeadDimension
  AttentionBias
  AttentionDropout
  AttentionOutputGate
  FullAttentionInterval
  HiddenActivation
  InitializerRange
  LinearConvWidth
  LinearKeyHeadCount
  LinearKeyHeadDimension
  LinearValueHeadCount
  LinearValueHeadDimension
  RecurrentStateDType
  ExpertIntermediateSize
  RoutedExpertCount
  ExpertsPerToken
  SharedExpertIntermediateSize
  OutputRouterLogits
  RouterAuxLoss
  MtpLayerCount
  MtpDedicatedEmbeddings
  PartialRotaryFactor
  RmsNormEpsilon
  RopeParameters
  UseCache
  PadTokenId
  RopeInterleaved
  RopeSections
  RopeTheta
  RopeType
  ActivationScheme
  Fp8Format
  QuantizationMethod
  ModulesNotConverted
  WeightBlockSize
} derive(Debug, Eq)

///|
/// Payload-safe issue vocabulary. No variant contains source JSON or paths.
pub(all) enum Qwen35ConfigValueIssue {
  ExpectedObject
  ExpectedString
  ExpectedInteger
  ExpectedNumber
  ExpectedBoolean
  ExpectedArray
  ExpectedNull
  InvalidArrayElement
  MissingRequiredEntry
  UnsupportedValue
} derive(Debug, Eq)

///|
/// Payload-safe failures for the exact Qwen3.5-MoE config route.
pub(all) suberror Qwen35ConfigError {
  InvalidByteLimit
  InvalidDepthLimit
  ByteLimitExceeded(limit~ : Int)
  NestingLimitExceeded(limit~ : Int)
  InvalidUtf8
  MalformedJson
  NonObjectRoot
  DuplicateField(depth~ : Int, index~ : Int)
  UnknownField(area~ : Qwen35ConfigArea, index~ : Int)
  MissingField(area~ : Qwen35ConfigArea, field~ : Qwen35ConfigField)
  InvalidValue(
    area~ : Qwen35ConfigArea,
    field~ : Qwen35ConfigField,
    issue~ : Qwen35ConfigValueIssue,
    index~ : Int
  )
} derive(Debug, Eq)
