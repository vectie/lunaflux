///|
test "MoE sigmoid correction oracle preserves uncorrected scores" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  let result = moe_sigmoid_and_correction(
    [0.0F, 0.0F, 0.0F, 0.0F],
    [0.25F, -0.25F],
    row_count=2,
    expert_count=2,
    limits~,
  )
  assert_eq(result.scores(), [0.5F, 0.5F, 0.5F, 0.5F])
  assert_eq(result.scores_for_choice(), [0.75F, 0.25F, 0.75F, 0.25F])
  assert_eq(result.row_count(), 2)
}

///|
test "MoE sigmoid correction oracle rejects mismatched bias" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  try
    moe_sigmoid_and_correction(
      [0.0F, 0.0F],
      [0.0F],
      row_count=1,
      expert_count=2,
      limits~,
    )
  catch {
    InvalidShape => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("mismatched correction bias admitted")
  }
}
