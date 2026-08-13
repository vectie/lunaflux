///|
/// Public field whose inference-contract value failed validation.
///
/// The vocabulary is intentionally payload-free: errors identify a location
/// and category without retaining request text, stop strings, identifiers, or
/// externally supplied numeric values.
pub(all) enum ContractField {
  ProtocolVersion
  RequestId
  RequestGeneration
  Limit
  InputToken
  InputText
  MaximumNewTokens
  ContextCeiling
  Temperature
  TopK
  TopP
  SamplingSeed
  StopToken
  StopString
  DeadlineBudget
  CacheScope
  TraceCorrelation
  DecodedDelta
  TokenPosition
  UsageInput
  UsageCachedInput
  UsageOutput
  PublicErrorCode
} derive(Debug, Eq)

///|
/// Payload-safe category of an inference-contract validation failure.
pub(all) enum ContractIssue {
  Empty
  TooLong
  TooMany
  InvalidFormat
  OutOfRange
  Duplicate
  NonFinite
  NotMonotonic
  Inconsistent
  Unsupported
} derive(Debug, Eq)

///|
/// Bounded public inference-contract failure.
///
/// `index` is a collection position, or zero for a scalar field. No variant
/// carries an offending value or raw request payload.
pub(all) suberror InferenceContractError {
  Invalid(field~ : ContractField, issue~ : ContractIssue, index~ : Int)
} derive(Debug, Eq)

///|
fn invalid(
  field : ContractField,
  issue : ContractIssue,
  index : Int,
) -> InferenceContractError {
  Invalid(field~, issue~, index~)
}
