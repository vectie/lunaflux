///|
test "MoE router oracle computes only row-major no-bias logits" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  let result = moe_router_projection(
    [1.0, 2.0, 3.0],
    [1.0, 0.0, -1.0, 0.5, 1.0, 2.0],
    row_count=1,
    hidden_width=3,
    expert_count=2,
    limits~,
  )
  assert_eq(result.logits(), [-2.0, 8.5])
  assert_eq(result.expert_count(), 2)
}

///|
test "MoE router oracle rejects truncated expert weights" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  try
    moe_router_projection(
      [1.0, 2.0],
      [1.0],
      row_count=1,
      hidden_width=2,
      expert_count=1,
      limits~,
    )
  catch {
    InvalidShape => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("truncated router weight admitted")
  }
}
