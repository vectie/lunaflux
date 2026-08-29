///|
struct FrozenDistribution {
  token_ids : FixedArray[Int]
  weights : FixedArray[Double]
  count : Int
  total_weight : Double
}

///|
fn complexity_inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=32,
    max_token_id=511,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=16,
    max_deadline_millis=1000UL,
    max_top_k=512,
    max_temperature=100.0,
  )
}

///|
fn complexity_parameters(
  temperature : Double,
  top_k : Int?,
  top_p : Double?,
) -> @inference.SamplingParameters raise {
  @inference.SamplingParameters::sample(
    temperature~,
    top_k?,
    top_p?,
    complexity_inference_limits(),
  )
}

///|
/// Frozen independent oracle: insertion-sort the complete vocabulary, then
/// reproduce the original canonical softmax and nucleus accumulation order.
fn frozen_preferred_before(
  logits : ArrayView[Double],
  left_token : Int,
  right_token : Int,
) -> Bool {
  let left = logits[left_token]
  let right = logits[right_token]
  left > right || (left == right && left_token < right_token)
}

///|
fn frozen_prepare(
  logits : ArrayView[Double],
  temperature : Double,
  top_k : Int?,
  top_p : Double?,
) -> FrozenDistribution {
  let token_ids = FixedArray::make(logits.length(), 0)
  for token_id in 0..<logits.length() {
    token_ids[token_id] = token_id
  }
  for index in 1..<logits.length() {
    let candidate = token_ids[index]
    let mut cursor = index
    while cursor > 0 &&
          frozen_preferred_before(logits, candidate, token_ids[cursor - 1]) {
      token_ids[cursor] = token_ids[cursor - 1]
      cursor -= 1
    }
    token_ids[cursor] = candidate
  }
  let top_k_count = match top_k {
    Some(value) => value.min(logits.length())
    None => logits.length()
  }
  let weights = FixedArray::make(logits.length(), 0.0)
  let maximum = logits[token_ids[0]]
  let mut total = 0.0
  for index in 0..<top_k_count {
    let token_id = token_ids[index]
    let weight = @math.exp((logits[token_id] - maximum) / temperature)
    weights[index] = weight
    total += weight
  }
  let retained = match top_p {
    Some(value) => {
      let threshold = value * total
      let mut cumulative = 0.0
      let mut count = 0
      while count < top_k_count && cumulative < threshold {
        cumulative += weights[count]
        count += 1
      }
      count.max(1)
    }
    None => top_k_count
  }
  let mut retained_total = 0.0
  for index in 0..<retained {
    retained_total += weights[index]
  }
  { token_ids, weights, count: retained, total_weight: retained_total }
}

///|
fn frozen_counter_state(seed : UInt64, sample_index : Int) -> UInt64 {
  let mut value = seed + (sample_index + 1).to_uint64() * 0x9e3779b97f4a7c15UL
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9UL
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebUL
  value = value ^ (value >> 31)
  if value == 0UL {
    0x9e3779b97f4a7c15UL
  } else {
    value
  }
}

///|
fn frozen_select(
  distribution : FrozenDistribution,
  seed : UInt64,
  sample_index : Int,
) -> Int {
  let unit = (frozen_counter_state(seed, sample_index) >> 11).to_double() *
    1.1102230246251565e-16
  let threshold = unit * distribution.total_weight
  let mut cumulative = 0.0
  let mut selected = distribution.count - 1
  for index in 0..<distribution.count {
    cumulative += distribution.weights[index]
    if threshold < cumulative {
      selected = index
      break
    }
  }
  distribution.token_ids[selected]
}

///|
fn assert_exact_distribution(
  logits : ArrayView[Double],
  temperature : Double,
  top_k : Int?,
  top_p : Double?,
  limits : SamplingLimits,
  scratch : SamplerScratch,
) -> FrozenDistribution raise {
  let expected = frozen_prepare(logits, temperature, top_k, top_p)
  let actual_count = prepare_distribution(
    logits,
    complexity_parameters(temperature, top_k, top_p),
    limits,
    scratch,
  )
  assert_eq(actual_count, expected.count)
  assert_eq(scratch.candidate_count, expected.count)
  assert_eq(scratch.total_weight, expected.total_weight)
  for index in 0..<expected.count {
    assert_eq(scratch.token_ids[index], expected.token_ids[index])
    assert_eq(scratch.weights[index], expected.weights[index])
  }
  expected
}

///|
test "restrictive top-k sorts K rather than V and exactly replays the oracle" {
  let vocabulary = 257
  let top_k = 7
  let limits = SamplingLimits::new(1, vocabulary)
  let scratch = SamplerScratch::new(vocabulary, limits)
  let logits = FixedArray::make(vocabulary, 0.0)
  for token_id in 0..<vocabulary {
    // Many ties plus late high candidates exercise replacement and token-ID
    // tie precedence independently of the production heap implementation.
    logits[token_id] = (token_id * 37 % 29 - 14).to_double() / 3.0
  }
  let expected = assert_exact_distribution(
    logits[:],
    0.75,
    Some(top_k),
    Some(0.91),
    limits,
    scratch,
  )
  assert_eq(scratch.order_candidate_visits, vocabulary)
  assert_eq(scratch.order_heap_width, top_k)
  assert_eq(scratch.order_sorted_width, top_k)
  assert_true(scratch.order_replacement_sifts > 0)
  assert_true(scratch.order_replacement_sifts <= vocabulary - top_k)
  let parameters = complexity_parameters(0.75, Some(top_k), Some(0.91))
  for seed in [1UL, 42UL, 0x123456789abcdef0UL, 0xffffffffffffffffUL] {
    let typed_seed = @inference.SamplingSeed::new(seed)
    for sample_index in 0..<128 {
      let actual = stochastic_sample_at(
        logits[:],
        parameters,
        limits,
        typed_seed,
        sample_index,
        scratch,
      )
      assert_eq(actual.token_id(), frozen_select(expected, seed, sample_index))
    }
  }
}

///|
test "top-k boundaries ties and extreme finite logits match full-sort oracle" {
  let limits = SamplingLimits::new(1, 16)
  let scratch = SamplerScratch::new(16, limits)
  ignore(
    assert_exact_distribution(
      [5.0, 5.0, 5.0, 5.0, 4.0],
      1.0,
      Some(3),
      None,
      limits,
      scratch,
    ),
  )
  assert_eq(scratch.token_ids[0], 0)
  assert_eq(scratch.token_ids[1], 1)
  assert_eq(scratch.token_ids[2], 2)
  ignore(
    assert_exact_distribution(
      [1.7976931348623157e308, 1.0, 0.0, -1.7976931348623157e308],
      1.0e-300,
      Some(1),
      Some(1.0),
      limits,
      scratch,
    ),
  )
  assert_eq(scratch.order_sorted_width, 1)
  ignore(
    assert_exact_distribution(
      [3.0, 2.0, 1.0, 0.0],
      1.0,
      Some(16),
      Some(1.0),
      limits,
      scratch,
    ),
  )
  assert_eq(scratch.order_heap_width, 4)
  assert_eq(scratch.order_sorted_width, 4)
}

///|
test "top-p without top-k retains canonical full-order numerical replay" {
  let vocabulary = 97
  let limits = SamplingLimits::new(1, vocabulary)
  let scratch = SamplerScratch::new(vocabulary, limits)
  let logits = FixedArray::make(vocabulary, 0.0)
  for token_id in 0..<vocabulary {
    logits[token_id] = (token_id * 19 % 17 - 8).to_double() / 7.0
  }
  ignore(
    assert_exact_distribution(logits[:], 1.25, None, Some(1.0), limits, scratch),
  )
  assert_eq(scratch.order_candidate_visits, vocabulary)
  assert_eq(scratch.order_heap_width, vocabulary)
  assert_eq(scratch.order_sorted_width, vocabulary)
  assert_eq(scratch.order_replacement_sifts, 0)
  ignore(
    assert_exact_distribution(logits[:], 1.25, None, None, limits, scratch),
  )
  assert_eq(scratch.order_sorted_width, vocabulary)
}
