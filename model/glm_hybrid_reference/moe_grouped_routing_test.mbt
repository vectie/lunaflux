///|
test "grouped route chooses with corrected scores and weights with uncorrected scores" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  let result = moe_grouped_route(
    [0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F],
    [10.0F, 9.0F, 1.0F, 8.0F, 7.0F, 6.0F],
    row_count=1,
    routed_expert_count=6,
    group_count=2,
    group_score_topk=2,
    selected_group_count=1,
    experts_per_token=2,
    normalize=true,
    scaling=2.0F,
    limits~,
  )
  assert_eq(result.selected_indices(), [0, 1])
  assert_eq(result.selected_weights(), [0.6666666F, 1.3333333F])
}

///|
test "grouped route breaks corrected-score ties by lower expert id" {
  let limits = GlmHybridReferenceLimits::new(
    maximum_sequence_length=8,
    maximum_head_count=8,
    maximum_head_dimension=8,
    maximum_channel_count=16,
    maximum_vision_patch_count=8,
  )
  let result = moe_grouped_route(
    [0.25F, 0.25F, 0.25F, 0.25F],
    [1.0F, 1.0F, 1.0F, 1.0F],
    row_count=1,
    routed_expert_count=4,
    group_count=1,
    group_score_topk=2,
    selected_group_count=1,
    experts_per_token=2,
    normalize=false,
    scaling=1.0F,
    limits~,
  )
  assert_eq(result.selected_indices(), [0, 1])
}
