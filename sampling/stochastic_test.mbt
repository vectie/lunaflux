///|
fn inference_limits() -> @inference.InferenceLimits raise {
  @inference.InferenceLimits::new(
    max_text_bytes=64,
    max_input_tokens=16,
    max_new_tokens=16,
    max_context_tokens=32,
    max_token_id=31,
    max_stop_token_ids=4,
    max_stop_strings=4,
    max_stop_string_bytes=16,
    max_trace_bytes=16,
    max_cache_scope_bytes=16,
    max_decoded_delta_bytes=16,
    max_deadline_millis=1000UL,
    max_top_k=32,
    max_temperature=100.0,
  )
}

///|
fn sample_parameters(
  temperature : Double,
  top_k~ : Int?,
  top_p~ : Double?,
) -> @inference.SamplingParameters raise {
  @inference.SamplingParameters::sample(
    temperature~,
    top_k?,
    top_p?,
    inference_limits(),
  )
}

///|
fn close(actual : Double, expected : Double, tolerance : Double) -> Unit raise {
  assert_true((actual - expected).abs() <= tolerance)
}

///|
test "golden distribution applies top-k before top-p and orders ties" {
  let limits = @sampling.SamplingLimits::new(1, 8)
  let scratch = @sampling.SamplerScratch::new(8, limits)
  let filtered = sample_parameters(1.0, top_k=Some(3), top_p=Some(0.8))
  let count = @sampling.prepare_distribution(
    [1.3862943611198906, 0.6931471805599453, 0.0, -20.0],
    filtered,
    limits,
    scratch,
  )
  assert_eq(count, 2)
  assert_eq(scratch.candidate_count(), 2)
  assert_eq(scratch.candidate_token_id(0), 0)
  assert_eq(scratch.candidate_token_id(1), 1)
  close(scratch.candidate_probability(0), 2.0 / 3.0, 1.0e-14)
  close(scratch.candidate_probability(1), 1.0 / 3.0, 1.0e-14)
  let tied = sample_parameters(1.0, top_k=Some(2), top_p=None)
  ignore(@sampling.prepare_distribution([4.0, 4.0, 4.0], tied, limits, scratch))
  assert_eq(scratch.candidate_count(), 2)
  assert_eq(scratch.candidate_token_id(0), 0)
  assert_eq(scratch.candidate_token_id(1), 1)
  close(scratch.candidate_probability(0), 0.5, 1.0e-15)
  close(scratch.candidate_probability(1), 0.5, 1.0e-15)
}

///|
test "top-p threshold keeps the smallest exact equal-mass nucleus" {
  let limits = @sampling.SamplingLimits::new(1, 4)
  let scratch = @sampling.SamplerScratch::new(4, limits)
  let half = sample_parameters(1.0, top_k=None, top_p=Some(0.5))
  ignore(
    @sampling.prepare_distribution([0.0, 0.0, 0.0, 0.0], half, limits, scratch),
  )
  assert_eq(scratch.candidate_count(), 2)
  assert_eq(scratch.candidate_probability(0), 0.5)
  assert_eq(scratch.candidate_probability(1), 0.5)
  let above_half = sample_parameters(
    1.0,
    top_k=None,
    top_p=Some(0.5000000000000001),
  )
  ignore(
    @sampling.prepare_distribution(
      [0.0, 0.0, 0.0, 0.0],
      above_half,
      limits,
      scratch,
    ),
  )
  assert_eq(scratch.candidate_count(), 3)
  for index in 0..<3 {
    close(scratch.candidate_probability(index), 1.0 / 3.0, 1.0e-15)
  }
  let quarter = sample_parameters(1.0, top_k=None, top_p=Some(0.25))
  ignore(
    @sampling.prepare_distribution(
      [0.0, 0.0, 0.0, 0.0],
      quarter,
      limits,
      scratch,
    ),
  )
  assert_eq(scratch.candidate_count(), 1)
  assert_eq(scratch.candidate_probability(0), 1.0)
}

///|
test "golden seeded selections and replay state are stable" {
  let limits = @sampling.SamplingLimits::new(1, 4)
  let params = sample_parameters(1.0, top_k=None, top_p=None)
  let scratch = @sampling.SamplerScratch::new(4, limits)
  let low_rng = @sampling.SamplingRng::from_seed(
    @sampling.SamplingSeed::new(1UL),
  )
  let low = @sampling.stochastic_sample(
    [0.0, 0.0, 0.0],
    params,
    limits,
    low_rng,
    scratch,
  )
  assert_eq(low.token_id(), 0)
  assert_eq(low.logit(), 0.0)
  assert_eq(low_rng.state(), 1082269761UL)
  let high_rng = @sampling.SamplingRng::from_seed(
    @sampling.SamplingSeed::new(0x6c576fac43fd007cUL),
  )
  let high = @sampling.stochastic_sample(
    [0.0, 0.0, 0.0],
    params,
    limits,
    high_rng,
    scratch,
  )
  assert_eq(high.token_id(), 2)
  assert_eq(high_rng.state(), 0xc4992a768ecd657cUL)
}

///|
test "identical seeds reproduce a multi-step stream" {
  let limits = @sampling.SamplingLimits::new(1, 4)
  let params = sample_parameters(0.75, top_k=Some(3), top_p=Some(0.95))
  let left_scratch = @sampling.SamplerScratch::new(4, limits)
  let right_scratch = @sampling.SamplerScratch::new(4, limits)
  let seed = @sampling.SamplingSeed::new(0x123456789abcdef0UL)
  let left = @sampling.SamplingRng::from_seed(seed)
  let right = @sampling.SamplingRng::from_seed(seed)
  for _ in 0..<64 {
    let left_selection = @sampling.stochastic_sample(
      [2.0, 1.5, 1.0, -3.0],
      params,
      limits,
      left,
      left_scratch,
    )
    let right_selection = @sampling.stochastic_sample(
      [2.0, 1.5, 1.0, -3.0],
      params,
      limits,
      right,
      right_scratch,
    )
    assert_eq(left_selection.token_id(), right_selection.token_id())
    assert_eq(left.state(), right.state())
  }
}

///|
test "validation failures are bounded and do not advance RNG" {
  let limits = @sampling.SamplingLimits::new(1, 3)
  let params = sample_parameters(1.0, top_k=None, top_p=None)
  let scratch = @sampling.SamplerScratch::new(2, limits)
  let rng = @sampling.SamplingRng::from_seed(@sampling.SamplingSeed::new(42UL))
  try
    @sampling.stochastic_sample([1.0, 2.0, 3.0], params, limits, rng, scratch)
  catch {
    error =>
      debug_inspect(error, content="ScratchTooSmall(required=3, capacity=2)")
  } noraise {
    _ => fail("expected scratch rejection")
  }
  assert_eq(rng.state(), 42UL)
  try
    @sampling.stochastic_sample(
      [@double.not_a_number, @double.infinity],
      params,
      limits,
      rng,
      scratch,
    )
  catch {
    error => debug_inspect(error, content="NonFiniteLogit(row=0, token_id=0)")
  } noraise {
    _ => fail("expected all-invalid row rejection")
  }
  assert_eq(rng.state(), 42UL)
  try
    @sampling.stochastic_sample(
      [1.0],
      @inference.SamplingParameters::greedy(),
      limits,
      rng,
      scratch,
    )
  catch {
    error => debug_inspect(error, content="InvalidSamplingMode")
  } noraise {
    _ => fail("expected explicit mode rejection")
  }
  assert_eq(rng.state(), 42UL)
  try @sampling.SamplingSeed::new(0UL) catch {
    error => debug_inspect(error, content="InvalidSeed")
  } noraise {
    _ => fail("expected zero seed rejection")
  }
}

///|
test "NaN and both infinities report the exact bounded token position" {
  let limits = @sampling.SamplingLimits::new(1, 4)
  let params = sample_parameters(1.0, top_k=None, top_p=None)
  let scratch = @sampling.SamplerScratch::new(4, limits)
  for
    fixture in [
      ([0.0, @double.not_a_number], 1),
      ([0.0, @double.infinity], 1),
      ([@double.neg_infinity, 0.0], 0),
    ] {
    let (logits, position) = fixture
    try @sampling.prepare_distribution(logits, params, limits, scratch) catch {
      NonFiniteLogit(row=0, token_id~) => assert_eq(token_id, position)
      _ => fail("expected non-finite rejection")
    } noraise {
      _ => fail("expected non-finite rejection")
    }
  }
}

///|
test "max-shifted softmax handles extreme finite logits and temperature" {
  let limits = @sampling.SamplingLimits::new(1, 5)
  let scratch = @sampling.SamplerScratch::new(5, limits)
  let cold = sample_parameters(1.0e-300, top_k=None, top_p=Some(1.0))
  ignore(
    @sampling.prepare_distribution(
      [1.7976931348623157e308, -1.7976931348623157e308],
      cold,
      limits,
      scratch,
    ),
  )
  assert_eq(scratch.candidate_count(), 1)
  assert_eq(scratch.candidate_token_id(0), 0)
  assert_eq(scratch.candidate_probability(0), 1.0)
  let stable = sample_parameters(1.0, top_k=Some(5), top_p=Some(1.0))
  ignore(
    @sampling.prepare_distribution([10000.0, 9999.0], stable, limits, scratch),
  )
  assert_eq(scratch.candidate_count(), 2)
  close(
    scratch.candidate_probability(0),
    1.0 / (1.0 + @math.exp(-1.0)),
    1.0e-14,
  )
  let narrow = sample_parameters(100.0, top_k=None, top_p=Some(1.0e-300))
  ignore(@sampling.prepare_distribution([1.0, 1.0], narrow, limits, scratch))
  assert_eq(scratch.candidate_count(), 1)
  assert_eq(scratch.candidate_token_id(0), 0)
}

///|
test "candidate observation rejects missing and boundary indices" {
  let limits = @sampling.SamplingLimits::new(1, 2)
  let scratch = @sampling.SamplerScratch::new(2, limits)
  try scratch.candidate_probability(0) catch {
    error =>
      debug_inspect(error, content="CandidateIndexOutOfRange(index=0, count=0)")
  } noraise {
    _ => fail("expected unprepared observation rejection")
  }
  ignore(
    @sampling.prepare_distribution(
      [0.0, 0.0],
      sample_parameters(1.0, top_k=None, top_p=None),
      limits,
      scratch,
    ),
  )
  try scratch.candidate_token_id(-1) catch {
    error =>
      debug_inspect(
        error,
        content="CandidateIndexOutOfRange(index=-1, count=2)",
      )
  } noraise {
    _ => fail("expected negative index rejection")
  }
  try scratch.candidate_token_id(2) catch {
    error =>
      debug_inspect(error, content="CandidateIndexOutOfRange(index=2, count=2)")
  } noraise {
    _ => fail("expected upper index rejection")
  }
}

///|
test "candidate ordering is total across unsorted inputs and ties" {
  let limits = @sampling.SamplingLimits::new(1, 8)
  let scratch = @sampling.SamplerScratch::new(8, limits)
  let parameters = sample_parameters(1.0, top_k=None, top_p=None)
  ignore(
    @sampling.prepare_distribution(
      [2.0, -1.0, 4.0, 2.0, 9.0, 9.0, 0.0, 4.0],
      parameters,
      limits,
      scratch,
    ),
  )
  let expected = [4, 5, 2, 7, 0, 3, 6, 1]
  for index in 0..<expected.length() {
    assert_eq(scratch.candidate_token_id(index), expected[index])
  }
}

///|
test "failed preparation clears an earlier distribution" {
  let limits = @sampling.SamplingLimits::new(1, 2)
  let scratch = @sampling.SamplerScratch::new(2, limits)
  let parameters = sample_parameters(1.0, top_k=None, top_p=None)
  ignore(
    @sampling.prepare_distribution([1.0, 0.0], parameters, limits, scratch),
  )
  assert_eq(scratch.candidate_count(), 2)
  try
    @sampling.prepare_distribution(
      [1.0, @double.not_a_number],
      parameters,
      limits,
      scratch,
    )
  catch {
    NonFiniteLogit(_) => ()
    _ => fail("expected non-finite rejection")
  } noraise {
    _ => fail("expected non-finite rejection")
  }
  assert_eq(scratch.candidate_count(), 0)
  try scratch.candidate_token_id(0) catch {
    error =>
      debug_inspect(error, content="CandidateIndexOutOfRange(index=0, count=0)")
  } noraise {
    _ => fail("expected cleared distribution")
  }
}
