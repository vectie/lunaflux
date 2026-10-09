///|
test "KDA gated RMSNorm normalizes each head before applying sigmoid gate" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=64,
    maximum_vision_patch_count=8,
  )
  let result = kda_sigmoid_gated_rms_norm(
    [3.0, 4.0, 0.0, 5.0],
    [0.0, 0.0, 0.0, 0.0],
    [2.0, 4.0],
    row_count=1,
    head_count=2,
    head_dimension=2,
    epsilon=0.00001,
    limits~,
  )
  assert_eq(result.row_count(), 1)
  assert_eq(result.values().length(), 4)
  assert_true(result.values()[0] > 0.84 && result.values()[0] < 0.86)
  assert_true(result.values()[1] > 2.26 && result.values()[1] < 2.27)
  assert_eq(result.values()[2], 0.0)
}

///|
test "KDA gated RMSNorm rejects mismatched gate and weight shapes" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=64,
    maximum_vision_patch_count=8,
  )
  try
    kda_sigmoid_gated_rms_norm(
      [1.0, 2.0],
      [0.0],
      [1.0, 1.0],
      row_count=1,
      head_count=1,
      head_dimension=2,
      epsilon=0.00001,
      limits~,
    )
  catch {
    InvalidShape => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("mismatched gate admitted")
  }
}
