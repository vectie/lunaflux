///|
fn tiny_identity_experts() -> Array[Float] {
  [1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 0.0F, 1.0F]
}

///|
fn tiny_block_input() -> TinyAdvancedDecoderBlockInput {
  try! TinyAdvancedDecoderBlockInput::new(
    stream_activations=[1.0F, 1.0F, 3.0F, 3.0F],
    query_weight=[1.0F, 0.0F, 0.0F, 1.0F],
    keys=[1.0F, 0.0F, 0.0F, 1.0F],
    values=[2.0F, 0.0F, 0.0F, 2.0F],
    output_weight=[1.0F, 0.0F, 0.0F, 1.0F],
    mixing_logits=[0.0F, 0.0F, 0.0F, 0.0F],
    router_weight=[1.0F, 0.0F, 0.0F, 1.0F],
    selection_bias=[],
    expert_up_weight=tiny_identity_experts(),
    expert_gate_weight=Array::make(8, 0.0F),
    expert_down_weight=tiny_identity_experts(),
    stream_count=2,
    width=2,
    key_count=2,
    index_top_k=2,
    expert_count=2,
    experts_per_token=1,
    expert_hidden_width=2,
    sinkhorn_iterations=3,
    epsilon=0.000001F,
    limits=test_limits(),
  )
}

///|
test "tiny block composes sparse attention hyper-connections and MoE order" {
  let output = run_tiny_advanced_decoder_block(
    tiny_block_input(),
    limits=test_limits(),
  )
  assert_eq(output.selected_key_indices(), [0, 1])
  assert_float_close(output.attention_weights()[0], 0.5F)
  assert_float_close(output.attention_weights()[1], 0.5F)
  assert_eq(output.routing().expert_indices(), [0])
  assert_float_close(output.routing().weights()[0], 1.0F)
  for value in output.mixed_streams() {
    assert_float_close(value, 2.5F)
  }
  assert_float_close(output.hidden()[0], 2.5F)
  assert_float_close(output.hidden()[1], 2.5F)
}

///|
test "tiny block rejects malformed and non-finite weights" {
  try
    TinyAdvancedDecoderBlockInput::new(
      stream_activations=[1.0F, 1.0F],
      query_weight=[1.0F],
      keys=[1.0F, 1.0F],
      values=[1.0F, 1.0F],
      output_weight=[1.0F],
      mixing_logits=[0.0F],
      router_weight=[1.0F, 1.0F],
      selection_bias=[],
      expert_up_weight=[1.0F, 1.0F],
      expert_gate_weight=[1.0F, 1.0F],
      expert_down_weight=[1.0F, 1.0F],
      stream_count=1,
      width=2,
      key_count=1,
      index_top_k=1,
      expert_count=1,
      experts_per_token=1,
      expert_hidden_width=1,
      sinkhorn_iterations=1,
      epsilon=0.000001F,
      limits=test_limits(),
    )
  catch {
    InvalidBlock => ()
    _ => fail("unexpected malformed-block rejection")
  } noraise {
    _ => fail("malformed block was admitted")
  }
}
