///|
test "full GLM-5.3 retains the exact official MoE DSA schedule" {
  let spec = @glm53_spec.Glm53FullSpec::official()
  assert_eq(spec.architecture(), "GlmMoeDsaForCausalLM")
  assert_eq(spec.model_type(), "glm_moe_dsa")
  assert_eq(spec.layer_count(), 78)
  assert_eq(spec.qk_head_dimension(), 256)
  assert_eq(spec.rms_norm_epsilon(), 0.00001)
  assert_eq(spec.routed_scaling_factor(), 2.5)
  assert_true(spec.uses_float32_moe_router())
  assert_true(spec.indexer_rope_interleave())
  assert_eq(spec.feed_forward_kind(0), DenseSwiGlu)
  assert_eq(spec.feed_forward_kind(2), DenseSwiGlu)
  assert_eq(spec.feed_forward_kind(3), RoutedMixtureOfExperts)
  assert_true(spec.computes_index_topk(0))
  assert_true(spec.computes_index_topk(2))
  assert_false(spec.computes_index_topk(3))
  assert_true(spec.computes_index_topk(6))
  assert_eq(spec.attention_kind(77), DeepSeekSparseAttention)
}

///|
test "Flash GLM-5.3 retains hybrid KDA DSA and multimodal geometry" {
  let spec = @glm53_spec.Glm53FlashSpec::official()
  assert_eq(spec.architecture(), "Glm5NextForConditionalGeneration")
  assert_eq(spec.model_type(), "glm5_next")
  assert_eq(spec.layer_count(), 45)
  assert_eq(spec.vision_layer_count(), 24)
  assert_eq(spec.hyper_connection_epsilon(), 0.000001)
  assert_eq(spec.linear_gate_lower_bound(), -5.0)
  assert_eq(spec.swiglu_limit(), 10.0)
  assert_eq(spec.vision_projection_intermediate_size(), 10240)
  assert_eq(spec.attention_kind(0), KimiDeltaAttention)
  assert_eq(spec.attention_kind(2), KimiDeltaAttention)
  assert_eq(spec.attention_kind(3), DeepSeekSparseAttention)
  assert_eq(spec.attention_kind(44), KimiDeltaAttention)
  assert_eq(spec.feed_forward_kind(2), DenseSwiGlu)
  assert_eq(spec.feed_forward_kind(3), RoutedMixtureOfExperts)
  assert_eq(spec.image_token_id(), 154854)
  assert_eq(spec.video_token_id(), 154855)
  assert_eq(spec.image_start_token_id(), 154830)
  assert_eq(spec.video_end_token_id(), 154833)
}

///|
test "GLM-5.3 profiles reject out-of-range layer selection" {
  try @glm53_spec.Glm53FullSpec::official().attention_kind(78) catch {
    InvalidLayerIndex => ()
  } noraise {
    _ => fail("out-of-range full layer was admitted")
  }
  try @glm53_spec.Glm53FlashSpec::official().feed_forward_kind(-1) catch {
    InvalidLayerIndex => ()
  } noraise {
    _ => fail("negative Flash layer was admitted")
  }
}
