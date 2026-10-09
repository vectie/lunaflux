///|
fn tiny_moe_limits() -> GlmHybridReferenceLimits {
  try! GlmHybridReferenceLimits::new(
    maximum_sequence_length=4,
    maximum_head_count=4,
    maximum_head_dimension=4,
    maximum_channel_count=16,
    maximum_vision_patch_count=4,
  )
}

///|
test "modular MoE oracle preserves selected weighting shared add and clamp" {
  let input = [2.0]
  let routed = moe_routed_expert_feed_forward(
    input,
    [1, 0],
    [0.25F, 0.75F],
    [1.0, 2.0],
    [1.0, 1.0],
    [1.0, 1.0],
    row_count=1,
    hidden_width=1,
    intermediate_width=1,
    expert_count=2,
    experts_per_token=2,
    activation=SwiGlu,
    limits=tiny_moe_limits(),
  )
  let shared = moe_shared_expert_feed_forward(
    input,
    [20.0],
    [20.0],
    [1.0],
    row_count=1,
    hidden_width=1,
    intermediate_width=1,
    activation=ClampedSwiGlu(10.0),
    limits=tiny_moe_limits(),
  )
  let combined = moe_routed_shared_combine(
    routed.values(),
    shared.values(),
    row_count=1,
    hidden_width=1,
    limits=tiny_moe_limits(),
  )
  assert_eq(combined.row_count(), 1)
  assert_eq(combined.hidden_width(), 1)
  assert_true(combined.values()[0] > routed.values()[0])
  assert_true(shared.values()[0] < 101.0)
}

///|
test "modular MoE oracle rejects duplicate and out-of-range selected experts" {
  for indices in [[0, 0], [0, 2]] {
    try
      moe_routed_expert_feed_forward(
        [1.0],
        indices,
        [0.5F, 0.5F],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        row_count=1,
        hidden_width=1,
        intermediate_width=1,
        expert_count=2,
        experts_per_token=2,
        activation=SwiGlu,
        limits=tiny_moe_limits(),
      )
    catch {
      DuplicateIndex | IndexOutOfBounds => ()
      _ => fail("wrong selected-expert error")
    } noraise {
      _ => fail("invalid selected experts admitted")
    }
  }
}
