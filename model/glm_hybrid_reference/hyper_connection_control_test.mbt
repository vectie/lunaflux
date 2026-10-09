///|
test "hyper function keeps post initial matrix and collapse as separate outputs" {
  let limits = try! GlmHybridReferenceLimits::new(
    maximum_sequence_length=2,
    maximum_head_count=4,
    maximum_head_dimension=4,
    maximum_channel_count=16,
    maximum_vision_patch_count=2,
  )
  let function = try! hyper_connection_function_and_collapse(
    [1.0, 2.0, 3.0, 4.0],
    Array::make(24 * 4, 0.0),
    Array::make(24, 0.0),
    [1.0, 1.0, 1.0],
    row_count=1,
    stream_count=4,
    hidden_width=1,
    control_epsilon=0.000001,
    normalization_epsilon=0.00001,
    limits~,
  )
  assert_eq(function.post(), [1.0, 1.0, 1.0, 1.0])
  assert_true((function.collapsed()[0] - 5.00001).abs() < 0.000001)
  assert_true((function.initial_combination()[0] - 0.250001).abs() < 0.000001)
  let sinkhorn = try! hyper_connection_positive_sinkhorn(
    function.initial_combination(),
    row_count=1,
    stream_count=4,
    iterations=20,
    epsilon=0.000001,
    limits~,
  )
  assert_true((sinkhorn.combination()[0] - 0.24999975).abs() < 0.00001)
}

///|
test "hyper Sinkhorn rejects zero and mismatched matrices" {
  let limits = try! GlmHybridReferenceLimits::new(
    maximum_sequence_length=2,
    maximum_head_count=4,
    maximum_head_dimension=4,
    maximum_channel_count=16,
    maximum_vision_patch_count=2,
  )
  try
    hyper_connection_positive_sinkhorn(
      Array::make(16, 0.0),
      row_count=1,
      stream_count=4,
      iterations=20,
      epsilon=0.000001,
      limits~,
    )
  catch {
    InvalidProbability => ()
    _ => fail("wrong hostile matrix rejection")
  } noraise {
    _ => fail("zero Sinkhorn matrix admitted")
  }
}
