///|
test "KDA decay control applies per-head exp rate and safe-bound sigmoid" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=64,
    maximum_vision_patch_count=8,
  )
  let result = kda_safe_decay_control(
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0],
    row_count=1,
    head_count=2,
    head_dimension=2,
    lower_bound=-5.0,
    limits~,
  )
  assert_eq(result.log_decay()[0], -2.5)
  assert_true(result.log_decay()[1] < -3.65)
  assert_eq(result.log_decay()[1], result.log_decay()[2])
}

///|
test "KDA decay control rejects mismatched per-head A_log" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=64,
    maximum_vision_patch_count=8,
  )
  try
    kda_safe_decay_control(
      [0.0, 0.0],
      [0.0, 0.0],
      [],
      row_count=1,
      head_count=1,
      head_dimension=2,
      lower_bound=-5.0,
      limits~,
    )
  catch {
    InvalidShape => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("missing A_log admitted")
  }
}
