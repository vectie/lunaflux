///|
fn test_limits() -> ReferenceLimits {
  try! ReferenceLimits::new(max_values=20000, max_width=256, max_iterations=64)
}

///|
fn assert_float_close(actual : Float, expected : Float) -> Unit raise {
  assert_true(
    actual.is_close(
      expected,
      relative_tolerance=0.00001F,
      absolute_tolerance=0.00001F,
    ),
  )
}

///|
test "expert transforms are stable at zero and extreme finite logits" {
  let sqrt_softplus = transform_expert_scores(
    [0.0F, 100.0F, -100.0F],
    SqrtSoftplus,
    limits=test_limits(),
  )
  assert_float_close(sqrt_softplus[0], 0.83255464F)
  assert_float_close(sqrt_softplus[1], 10.0F)
  assert_true(sqrt_softplus[2] >= 0.0F)
  let sigmoid = transform_expert_scores(
    [0.0F, 100.0F, -100.0F],
    Sigmoid,
    limits=test_limits(),
  )
  assert_float_close(sigmoid[0], 0.5F)
  assert_float_close(sigmoid[1], 1.0F)
  assert_true(sigmoid[2] >= 0.0F && sigmoid[2] < 0.00001F)
}

///|
test "biased stable top-k keeps unbiased normalized weights" {
  let route = route_top_k(
    [0.0F, 0.0F, 0.0F],
    [0.0F, 1.0F, 0.0F],
    top_k=2,
    score_function=Sigmoid,
    normalize=true,
    scaling=2.0F,
    limits=test_limits(),
  )
  assert_eq(route.expert_indices(), [1, 0])
  assert_float_close(route.weights()[0], 1.0F)
  assert_float_close(route.weights()[1], 1.0F)
}

///|
test "stable top-k gives exact lower expert id order for complete ties" {
  let route = route_top_k(
    [0.0F, 0.0F, 0.0F, 0.0F],
    [1.0F, 1.0F, 1.0F, 1.0F],
    top_k=3,
    score_function=Sigmoid,
    normalize=true,
    scaling=1.5F,
    limits=test_limits(),
  )
  assert_eq(route.expert_indices(), [0, 1, 2])
  for weight in route.weights() {
    assert_float_close(weight, 0.5F)
  }
}

///|
test "token hash validates the whole table and preserves row order" {
  let route = route_token_hash(
    [0.0F, 0.0F, 0.0F],
    [2, 0, 1, 1, 2, 0],
    vocabulary_size=2,
    experts_per_token=3,
    token_id=1,
    score_function=Sigmoid,
    normalize=true,
    scaling=1.5F,
    limits=test_limits(),
  )
  assert_eq(route.expert_indices(), [1, 2, 0])
  for weight in route.weights() {
    assert_float_close(weight, 0.5F)
  }
  try
    route_token_hash(
      [0.0F, 0.0F, 0.0F],
      [0, 1, 2, 0, 0, 1],
      vocabulary_size=2,
      experts_per_token=3,
      token_id=0,
      score_function=Sigmoid,
      normalize=true,
      scaling=1.0F,
      limits=test_limits(),
    )
  catch {
    DuplicateExpertIndex(4) => ()
    _ => fail("unexpected invalid hash-table error")
  } noraise {
    _ => fail("duplicate hash expert was admitted")
  }
}

///|
test "routed and shared expert results combine in explicit order" {
  let route = route_top_k(
    [0.0F, 0.0F],
    [],
    top_k=2,
    score_function=Sigmoid,
    normalize=true,
    scaling=2.0F,
    limits=test_limits(),
  )
  let output = combine_expert_outputs(
    route,
    [1.0F, 2.0F, 3.0F, 4.0F],
    [10.0F, 20.0F],
    width=2,
    limits=test_limits(),
  )
  assert_float_close(output[0], 14.0F)
  assert_float_close(output[1], 26.0F)
}

///|
test "routing hostile bounds NaN and shape errors fail closed" {
  try
    transform_expert_scores(
      [@float.not_a_number],
      Sigmoid,
      limits=test_limits(),
    )
  catch {
    NonFiniteInput(0) => ()
    _ => fail("unexpected NaN error")
  } noraise {
    _ => fail("NaN expert score was admitted")
  }
  try
    route_top_k(
      [0.0F],
      [],
      top_k=2,
      score_function=Sigmoid,
      normalize=true,
      scaling=1.0F,
      limits=test_limits(),
    )
  catch {
    InvalidTopK => ()
    _ => fail("unexpected top-k error")
  } noraise {
    _ => fail("oversized top-k was admitted")
  }
  try
    route_top_k(
      [0.0F, 1.0F],
      [0.0F],
      top_k=1,
      score_function=Sigmoid,
      normalize=true,
      scaling=1.0F,
      limits=test_limits(),
    )
  catch {
    InvalidDimension => ()
    _ => fail("unexpected bias-shape error")
  } noraise {
    _ => fail("wrong bias shape was admitted")
  }
}
