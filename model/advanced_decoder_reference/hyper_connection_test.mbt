///|
test "Sinkhorn normalization and stream mixing match hand computation" {
  let limits = test_limits()
  let mixing = sinkhorn_normalize(
    [0.0F, 0.0F, 0.0F, 0.0F],
    width=2,
    iterations=3,
    epsilon=0.000001F,
    limits~,
  )
  assert_eq(mixing, [0.5F, 0.5F, 0.5F, 0.5F])
  let output = mix_hyper_connections(
    [1.0F, 2.0F, 3.0F, 4.0F],
    mixing,
    stream_count=2,
    feature_width=2,
    limits~,
  )
  assert_eq(output, [2.0F, 3.0F, 2.0F, 3.0F])
}

///|
test "Sinkhorn rejects NaN iterations and underflowed rows" {
  let limits = test_limits()
  try
    sinkhorn_normalize(
      [0.0F, @float.not_a_number, 0.0F, 0.0F],
      width=2,
      iterations=2,
      epsilon=0.000001F,
      limits~,
    )
  catch {
    NonFiniteInput(1) => ()
    _ => fail("unexpected Sinkhorn NaN error")
  } noraise {
    _ => fail("Sinkhorn admitted NaN")
  }
  try
    sinkhorn_normalize(
      [0.0F],
      width=1,
      iterations=0,
      epsilon=0.000001F,
      limits~,
    )
  catch {
    InvalidIterationCount => ()
    _ => fail("unexpected iteration error")
  } noraise {
    _ => fail("zero Sinkhorn iterations were admitted")
  }
  try
    sinkhorn_normalize(
      [0.0F, 0.0F, -1000.0F, -1000.0F],
      width=2,
      iterations=1,
      epsilon=0.000001F,
      limits~,
    )
  catch {
    ZeroNormalization => ()
    _ => fail("unexpected underflow error")
  } noraise {
    _ => fail("underflowed Sinkhorn row was admitted")
  }
}
