///|
/// Dimension governed by a sampling limit.
pub(all) enum LimitDimension {
  Rows
  TokensPerRow
} derive(Debug, Eq)

///|
/// Payload-safe deterministic sampling failures.
///
/// Non-finite positions are bounded by a validated `SamplingLimits` envelope;
/// logit values and input contents are deliberately omitted.
pub(all) suberror SamplingError {
  InvalidLimit(LimitDimension)
  EmptyInput
  InvalidShape
  LimitExceeded(LimitDimension)
  ScratchTooSmall(required~ : Int32, capacity~ : Int32)
  InvalidSamplingMode
  CandidateIndexOutOfRange(index~ : Int32, count~ : Int32)
  InvalidDistribution
  NonFiniteLogit(row~ : Int32, token_id~ : Int32)
} derive(Debug, Eq)

///|
/// Validated limits for one sampling invocation.
pub struct SamplingLimits {
  max_rows : Int32
  max_tokens_per_row : Int32
} derive(Debug, Eq)

///|
/// Constructs a bounded sampling envelope.
///
/// The reference executor accepts at most 4,096 rows and 1,048,576 tokens per
/// row. Smaller caller limits are encouraged and enforced by every selector.
pub fn SamplingLimits::new(
  max_rows : Int32,
  max_tokens_per_row : Int32,
) -> SamplingLimits raise SamplingError {
  if max_rows <= 0 || max_rows > 4096 {
    raise InvalidLimit(Rows)
  }
  if max_tokens_per_row <= 0 || max_tokens_per_row > 1048576 {
    raise InvalidLimit(TokensPerRow)
  }
  { max_rows, max_tokens_per_row, }
}

///|
pub fn SamplingLimits::max_rows(self : SamplingLimits) -> Int32 {
  self.max_rows
}

///|
pub fn SamplingLimits::max_tokens_per_row(self : SamplingLimits) -> Int32 {
  self.max_tokens_per_row
}

///|
/// Deterministic selected token and its finite source logit.
#valtype
pub struct Selection {
  token_id : Int32
  logit : Double
} derive(Debug, Eq)

///|
pub fn Selection::token_id(self : Selection) -> Int32 {
  self.token_id
}

///|
pub fn Selection::logit(self : Selection) -> Double {
  self.logit
}
