///|
/// Caller-owned reusable storage for stochastic selection.
///
/// Construction is the only point at which candidate storage is allocated.
/// Preparation and selection overwrite the fixed arrays without growing a
/// collection. The arrays remain private and are never returned.
pub struct SamplerScratch {
  priv token_ids : FixedArray[Int]
  priv weights : FixedArray[Double]
  priv mut candidate_count : Int
  priv mut total_weight : Double
}

///|
pub fn SamplerScratch::new(
  capacity : Int,
  limits : SamplingLimits,
) -> SamplerScratch raise SamplingError {
  if capacity <= 0 || capacity > limits.max_tokens_per_row() {
    raise InvalidLimit(TokensPerRow)
  }
  {
    token_ids: FixedArray::make(capacity, 0),
    weights: FixedArray::make(capacity, 0.0),
    candidate_count: 0,
    total_weight: 0.0,
  }
}

///|
pub fn SamplerScratch::capacity(self : SamplerScratch) -> Int {
  self.token_ids.length()
}

///|
/// Number of candidates in the most recently prepared distribution.
pub fn SamplerScratch::candidate_count(self : SamplerScratch) -> Int {
  self.candidate_count
}

///|
fn SamplerScratch::check_candidate(
  self : SamplerScratch,
  index : Int,
) -> Unit raise SamplingError {
  if index < 0 || index >= self.candidate_count {
    raise CandidateIndexOutOfRange(index~, count=self.candidate_count)
  }
}

///|
/// Reads one candidate from the most recently prepared distribution.
///
/// Candidates are ordered by descending source logit and then ascending token
/// ID. A later preparation overwrites this scalar-accessible snapshot.
pub fn SamplerScratch::candidate_token_id(
  self : SamplerScratch,
  index : Int,
) -> Int raise SamplingError {
  self.check_candidate(index)
  self.token_ids[index]
}

///|
/// Reads one normalized probability without exposing mutable scratch storage.
pub fn SamplerScratch::candidate_probability(
  self : SamplerScratch,
  index : Int,
) -> Double raise SamplingError {
  self.check_candidate(index)
  if self.total_weight <= 0.0 ||
    self.total_weight.is_nan() ||
    self.total_weight.is_inf() {
    raise InvalidDistribution
  }
  self.weights[index] / self.total_weight
}
