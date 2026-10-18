///|
/// Prepares the exact distribution consumed by `stochastic_sample`.
///
/// Processing order is: validate every logit; order by descending logit and
/// ascending token ID; retain top-k; compute max-shifted temperature softmax;
/// then retain the smallest leading nucleus whose mass reaches top-p. The
/// retained nucleus is renormalized during observation and selection.
///
/// The returned count is also available through `scratch.candidate_count()`.
/// Preparation does not read or advance an RNG.
pub fn prepare_distribution(
  logits : ArrayView[Double],
  parameters : @inference.SamplingParameters,
  limits : SamplingLimits,
  scratch : SamplerScratch,
) -> Int raise SamplingError {
  if parameters.mode() != Sample {
    raise InvalidSamplingMode
  }
  let top_k_option = parameters.top_k()
  let has_top_k = top_k_option is Some(_)
  let top_k = top_k_option.unwrap_or(0)
  let top_p_option = parameters.top_p()
  let has_top_p = top_p_option is Some(_)
  let top_p = top_p_option.unwrap_or(0.0)
  prepare_distribution_scalars(
    logits,
    parameters.temperature(),
    has_top_k,
    top_k,
    has_top_p,
    top_p,
    limits,
    scratch,
  )
}

///|
fn prepare_distribution_scalars(
  logits : ArrayView[Double],
  temperature : Double,
  has_top_k : Bool,
  top_k : Int,
  has_top_p : Bool,
  top_p : Double,
  limits : SamplingLimits,
  scratch : SamplerScratch,
) -> Int raise SamplingError {
  // Never leave a previous distribution observable after a failed refresh.
  scratch.candidate_count = 0
  scratch.total_weight = 0.0
  if logits.is_empty() {
    raise EmptyInput
  }
  if logits.length() > limits.max_tokens_per_row() {
    raise LimitExceeded(TokensPerRow)
  }
  if logits.length() > scratch.capacity() {
    raise ScratchTooSmall(required=logits.length(), capacity=scratch.capacity())
  }
  if temperature.is_nan() || temperature.is_inf() || temperature <= 0.0 {
    raise InvalidDistribution
  }
  if (has_top_k && top_k <= 0) ||
    (
      has_top_p &&
      (top_p.is_nan() || top_p.is_inf() || top_p <= 0.0 || top_p > 1.0)
    ) {
    raise InvalidDistribution
  }
  for token_id in 0..<logits.length() {
    let logit = logits[token_id]
    if logit.is_nan() || logit.is_inf() {
      raise NonFiniteLogit(row=0, token_id~)
    }
  }
  let token_count = logits.length()
  let top_k_count = if has_top_k { top_k.min(token_count) } else { token_count }
  prepare_candidate_order(
    logits,
    token_count,
    top_k_count,
    has_restrictive_top_k=has_top_k && top_k_count < token_count,
    scratch,
  )
  let maximum = logits[scratch.token_ids[0]]
  let total = for index in 0..<top_k_count; total = 0.0 {
    let token_id = scratch.token_ids[index]
    let weight = @math.exp((logits[token_id] - maximum) / temperature)
    scratch.weights[index] = weight
    continue total + weight
  } nobreak {
    total
  }
  if total <= 0.0 || total.is_nan() || total.is_inf() {
    scratch.candidate_count = 0
    scratch.total_weight = 0.0
    raise InvalidDistribution
  }
  let retained = if has_top_p {
    let threshold = top_p * total
    for count = 0, cumulative = 0.0; count < top_k_count &&
       cumulative < threshold; {
      continue count + 1, cumulative + scratch.weights[count]
    } nobreak {
      count.max(1)
    }
  } else {
    top_k_count
  }
  let retained_total = for index in 0..<retained; total = 0.0 {
    continue total + scratch.weights[index]
  } nobreak {
    total
  }
  if retained_total <= 0.0 || retained_total.is_nan() || retained_total.is_inf() {
    scratch.candidate_count = 0
    scratch.total_weight = 0.0
    raise InvalidDistribution
  }
  scratch.candidate_count = retained
  scratch.total_weight = retained_total
  retained
}

///|
/// Deterministically samples from validated scalar wire fields without
/// reconstructing a heap-backed request parameter object.
pub fn stochastic_sample_at_scalars(
  logits : ArrayView[Double],
  temperature~ : Double,
  has_top_k~ : Bool,
  top_k~ : Int,
  has_top_p~ : Bool,
  top_p~ : Double,
  limits : SamplingLimits,
  seed : @inference.SamplingSeed,
  sample_index : Int,
  scratch : SamplerScratch,
) -> Selection raise SamplingError {
  if sample_index < 0 {
    raise InvalidDistribution
  }
  let count = prepare_distribution_scalars(
    logits, temperature, has_top_k, top_k, has_top_p, top_p, limits, scratch,
  )
  select_prepared(logits, count, counter_unit(seed, sample_index), scratch)
}

///|
/// Draws one token from a bounded prepared stochastic distribution.
///
/// The RNG advances exactly once after preparation succeeds. A floating-point
/// boundary equal to cumulative mass belongs to the following interval; the
/// final candidate closes any representational rounding gap.
pub fn stochastic_sample(
  logits : ArrayView[Double],
  parameters : @inference.SamplingParameters,
  limits : SamplingLimits,
  rng : SamplingRng,
  scratch : SamplerScratch,
) -> Selection raise SamplingError {
  let count = prepare_distribution(logits, parameters, limits, scratch)
  select_prepared(logits, count, rng.next_unit(), scratch)
}

///|
fn select_prepared(
  logits : ArrayView[Double],
  count : Int,
  unit : Double,
  scratch : SamplerScratch,
) -> Selection {
  let threshold = unit * scratch.total_weight
  let selected = for index in 0..<count; cumulative = 0.0 {
    let next = cumulative + scratch.weights[index]
    if threshold < next {
      break index
    }
    continue next
  } nobreak {
    count - 1
  }
  let token_id = scratch.token_ids[selected]
  { token_id, logit: logits[token_id], }
}

///|
/// Deterministically samples one output position without shared mutable RNG
/// state. The same `(seed, sample_index, logits, parameters)` always selects
/// the same token, making worker retry and process replacement replayable.
pub fn stochastic_sample_at(
  logits : ArrayView[Double],
  parameters : @inference.SamplingParameters,
  limits : SamplingLimits,
  seed : @inference.SamplingSeed,
  sample_index : Int,
  scratch : SamplerScratch,
) -> Selection raise SamplingError {
  if sample_index < 0 {
    raise InvalidDistribution
  }
  let count = prepare_distribution(logits, parameters, limits, scratch)
  select_prepared(logits, count, counter_unit(seed, sample_index), scratch)
}
