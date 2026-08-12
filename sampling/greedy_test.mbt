///|
test "greedy selects boundaries, negatives, and lowest tied token" {
  let limits = @sampling.SamplingLimits::new(4, 8)
  let first = @sampling.greedy([7.0, 2.0, -1.0], limits)
  assert_eq(first.token_id(), 0)
  assert_eq(first.logit(), 7.0)
  let last = @sampling.greedy([-8.0, -9.0, -2.0], limits)
  assert_eq(last.token_id(), 2)
  assert_eq(last.logit(), -2.0)
  let tied = @sampling.greedy([-3.0, 5.0, 5.0, 4.0], limits)
  assert_eq(tied.token_id(), 1)
  assert_eq(tied.logit(), 5.0)
}

///|
test "greedy rows validates shape and selects each borrowed row" {
  let limits = @sampling.SamplingLimits::new(3, 4)
  let selected = @sampling.greedy_rows(
    [-1.0, 3.0, 2.0, -4.0, -2.0, -3.0],
    2,
    3,
    limits,
  )
  assert_eq(selected.length(), 2)
  assert_eq(selected[0].token_id(), 1)
  assert_eq(selected[1].token_id(), 1)
  try @sampling.greedy_rows([1.0, 2.0, 3.0], 2, 2, limits) catch {
    error => debug_inspect(error, content="InvalidShape")
  } noraise {
    _ => fail("expected shape rejection")
  }
  try @sampling.greedy_rows([1.0], 0, 1, limits) catch {
    error => debug_inspect(error, content="InvalidShape")
  } noraise {
    _ => fail("expected zero-row shape rejection")
  }
}

///|
test "sampling rejects empty input and caller limit violations" {
  let upper = @sampling.SamplingLimits::new(4096, 1048576)
  assert_eq(upper.max_rows(), 4096)
  assert_eq(upper.max_tokens_per_row(), 1048576)
  let limits = @sampling.SamplingLimits::new(1, 2)
  try @sampling.greedy([], limits) catch {
    error => debug_inspect(error, content="EmptyInput")
  } noraise {
    _ => fail("expected empty input rejection")
  }
  try @sampling.greedy([1.0, 2.0, 3.0], limits) catch {
    error => debug_inspect(error, content="LimitExceeded(TokensPerRow)")
  } noraise {
    _ => fail("expected token limit rejection")
  }
  try @sampling.greedy_rows([1.0, 2.0], 2, 1, limits) catch {
    error => debug_inspect(error, content="LimitExceeded(Rows)")
  } noraise {
    _ => fail("expected row limit rejection")
  }
  try @sampling.SamplingLimits::new(0, 1) catch {
    error => debug_inspect(error, content="InvalidLimit(Rows)")
  } noraise {
    _ => fail("expected invalid limit rejection")
  }
  try @sampling.SamplingLimits::new(4097, 1) catch {
    error => debug_inspect(error, content="InvalidLimit(Rows)")
  } noraise {
    _ => fail("expected row ceiling rejection")
  }
  try @sampling.SamplingLimits::new(1, 1048577) catch {
    error => debug_inspect(error, content="InvalidLimit(TokensPerRow)")
  } noraise {
    _ => fail("expected token ceiling rejection")
  }
}

///|
test "sampling rejects NaN and both infinities with bounded positions" {
  let limits = @sampling.SamplingLimits::new(2, 3)
  try @sampling.greedy([0.0, @double.not_a_number], limits) catch {
    error => debug_inspect(error, content="NonFiniteLogit(row=0, token_id=1)")
  } noraise {
    _ => fail("expected NaN rejection")
  }
  try @sampling.greedy([-1.0, @double.infinity], limits) catch {
    error => debug_inspect(error, content="NonFiniteLogit(row=0, token_id=1)")
  } noraise {
    _ => fail("expected positive infinity rejection")
  }
  try
    @sampling.greedy_rows([1.0, 0.0, -1.0, @double.neg_infinity], 2, 2, limits)
  catch {
    error => debug_inspect(error, content="NonFiniteLogit(row=1, token_id=1)")
  } noraise {
    _ => fail("expected negative infinity rejection")
  }
}
