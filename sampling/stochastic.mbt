///|
fn preferred_before(
  logits : ArrayView[Double],
  left_token : Int,
  right_token : Int,
) -> Bool {
  let left = logits[left_token]
  let right = logits[right_token]
  left > right || (left == right && left_token < right_token)
}

///|
fn worse_than(
  logits : ArrayView[Double],
  left_token : Int,
  right_token : Int,
) -> Bool {
  preferred_before(logits, right_token, left_token)
}

///|
fn sift_worst_down(
  token_ids : FixedArray[Int],
  logits : ArrayView[Double],
  start : Int,
  end : Int,
) -> Unit {
  let mut root = start
  while root * 2 + 1 <= end {
    let child = root * 2 + 1
    let mut selected = child
    if child + 1 <= end &&
      worse_than(logits, token_ids[child + 1], token_ids[child]) {
      selected = child + 1
    }
    if !worse_than(logits, token_ids[selected], token_ids[root]) {
      break
    }
    let swap = token_ids[root]
    token_ids[root] = token_ids[selected]
    token_ids[selected] = swap
    root = selected
  }
}

///|
/// In-place heap sort with a total order: greatest logit first, lowest token ID
/// first on ties. It uses no temporary collection or recursive call stack.
fn sort_candidates(
  token_ids : FixedArray[Int],
  logits : ArrayView[Double],
  count : Int,
) -> Unit {
  if count <= 1 {
    return
  }
  let mut start = (count - 2) / 2
  for ;; {
    sift_worst_down(token_ids, logits, start, count - 1)
    if start == 0 {
      break
    }
    start = start - 1
  }
  let mut end = count - 1
  while end > 0 {
    let swap = token_ids[end]
    token_ids[end] = token_ids[0]
    token_ids[0] = swap
    end = end - 1
    sift_worst_down(token_ids, logits, 0, end)
  }
}

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
  for token_id in 0..<token_count {
    scratch.token_ids[token_id] = token_id
  }
  sort_candidates(scratch.token_ids, logits, token_count)
  let top_k_count = if has_top_k { top_k.min(token_count) } else { token_count }
  let maximum = logits[scratch.token_ids[0]]
  let mut total = 0.0
  for index in 0..<top_k_count {
    let token_id = scratch.token_ids[index]
    let weight = @math.exp((logits[token_id] - maximum) / temperature)
    scratch.weights[index] = weight
    total = total + weight
  }
  if total <= 0.0 || total.is_nan() || total.is_inf() {
    scratch.candidate_count = 0
    scratch.total_weight = 0.0
    raise InvalidDistribution
  }
  let retained = if has_top_p {
    let threshold = top_p * total
    let mut cumulative = 0.0
    let mut count = 0
    while count < top_k_count && cumulative < threshold {
      cumulative = cumulative + scratch.weights[count]
      count = count + 1
    }
    count.max(1)
  } else {
    top_k_count
  }
  let mut retained_total = 0.0
  for index in 0..<retained {
    retained_total = retained_total + scratch.weights[index]
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
  let mut cumulative = 0.0
  let mut selected = count - 1
  for index in 0..<count {
    cumulative = cumulative + scratch.weights[index]
    if threshold < cumulative {
      selected = index
      break
    }
  }
  let token_id = scratch.token_ids[selected]
  { token_id, logit: logits[token_id] }
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
