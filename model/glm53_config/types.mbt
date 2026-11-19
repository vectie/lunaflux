///|
/// Exact closed result of family-owned GLM-5.3 config admission.
pub(all) enum Glm53ConfigAdmission {
  Full(@glm53_spec.Glm53FullMetadata)
  Flash(@glm53_spec.Glm53FlashMetadata)
} derive(Debug, Eq)

///|
/// Stable semantic field groups used by payload-safe errors.
pub(all) enum Glm53ConfigField {
  Profile
  Architecture
  ModelType
  DecoderShape
  SparseAttention
  LayerSchedule
  ExpertRouting
  MultiTokenPrediction
  HyperConnection
  LinearAttention
  Vision
  ControlTokens
  NumericFormat
  InertMetadata
} derive(Debug, Eq)

///|
/// Payload-safe bounded parser failures. No attacker-controlled key, value,
/// source fragment, or filesystem label is retained.
pub(all) suberror Glm53ConfigError {
  InvalidByteLimit
  InvalidDepthLimit
  ByteLimitExceeded(limit~ : Int32)
  InvalidUtf8
  MalformedJson
  NonObjectRoot
  NestingLimitExceeded(limit~ : Int32)
  DuplicateField(depth~ : Int32, index~ : Int32)
  UnknownField(component~ : Glm53ConfigField, index~ : Int32)
  ExecutableMetadata
  MissingField(Glm53ConfigField)
  InvalidValue(Glm53ConfigField)
} derive(Debug, Eq)
