///|
test "ingress numerics are independent of fusion placement and head count" {
  let full = ProjectionAttentionIngressEpilogue::new(
    query_heads=16,
    kv_heads=8,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=1000000.0,
  )
  let partial = ProjectionAttentionIngressEpilogue::new(
    query_heads=4,
    kv_heads=2,
    head_dimension=128,
    norm_epsilon=0.000001,
    rope_theta=1000000.0,
  )
  let numerics = full.numerics(reduction_lanes=32)
  assert_eq(numerics, partial.numerics(reduction_lanes=32))
  assert_eq(numerics.shape(), (128, 32))
  assert_eq(numerics.rotary_basis().length(), 64)
  assert_eq(numerics.rotary_basis()[0], 1.0F)
  assert_eq(numerics.rotary_basis()[32], 0.001F)
  assert_true(numerics.rotary_basis()[63] > 0.000001F)
  assert_true(numerics != full.numerics(reduction_lanes=64))
}

///|
test "ingress numerics reject undefined Float lowering and reduction trees" {
  for
    item in [
      (127, 0.000001, 10000.0, 32),
      (128, 0.000001, 10000.0, 0),
      (128, 0.000001, 10000.0, 33),
      (128, 0.000001, 10000.0, 2048),
      (128, 1.0e-100, 10000.0, 32),
      (128, 0.000001, 1.0e100, 32),
    ] {
    let (dimension, epsilon, theta, lanes) = item
    let epilogue = ProjectionAttentionIngressEpilogue::new(
      query_heads=1,
      kv_heads=1,
      head_dimension=dimension,
      norm_epsilon=epsilon,
      rope_theta=theta,
    )
    try epilogue.numerics(reduction_lanes=lanes) catch {
      InvalidProjectionTileCompilation(InvalidProblem) => ()
      _ => fail("unexpected numerical-plan failure")
    } noraise {
      _ => fail("undefined numerical schedule accepted")
    }
  }
}
