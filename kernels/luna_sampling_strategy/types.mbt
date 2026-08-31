///|
/// Backend-neutral placement selected for one request row.
pub(all) enum SamplingRoute {
  DeviceGreedy
  DeviceRestrictiveTopK
  HostCanonical
} derive(Debug, Eq)

///|
/// Stable row-parameter ABI fields. Backends choose the physical encoding.
pub(all) enum SamplingParameterField {
  Mode
  TopK
  Temperature
  TopP
  Seed
  SampleIndex
  Flags
} derive(Debug, Eq)

///|
/// Failure dimensions for startup-only sampling strategy construction.
pub(all) enum SamplingStrategyRule {
  Shape
  CandidateBound
  Workspace
  Parameters
} derive(Debug, Eq)

///|
pub(all) suberror SamplingStrategyError {
  InvalidSamplingStrategy(SamplingStrategyRule)
} derive(Debug, Eq)

///|
/// Bounded device strategy for temperature/top-k/top-p selection.
///
/// The device route is deliberately restrictive-top-k: the canonical order is
/// descending finite logit then ascending token ID, temperature softmax is
/// evaluated over the first `top_k` entries, and top-p retains the smallest
/// prefix reaching the requested probability. A request without top-k uses the
/// existing full-vocabulary host route, preserving its established semantics.
pub struct SamplingDevicePlan {
  priv maximum_rows : Int
  priv vocabulary_size : Int
  priv maximum_top_k : Int
  priv parameter_row_bytes : Int64
  priv result_row_bytes : Int64
  priv candidate_token_offset_bytes : Int64
  priv candidate_weight_offset_bytes : Int64
  priv workspace_bytes : Int64
} derive(Debug, Eq)
