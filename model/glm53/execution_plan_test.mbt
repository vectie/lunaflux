///|
test "full GLM-5.3 builds the exact DSA MoE MTP execution plan" {
  let first = @glm53.build_full_execution_plan(full_metadata())
  let second = @glm53.build_full_execution_plan(full_metadata())
  assert_eq(first, second)
  assert_eq(first.profile_identity(), "zai-org/GLM-5.3:glm_moe_dsa")
  assert_eq(first.decoder_layer_count(), 78)
  assert_eq(first.layers().length(), 78)
  assert_eq(first.layers()[0].attention(), DynamicSparse(ComputeTopK))
  assert_eq(first.layers()[2].feed_forward(), DenseSwiGlu)
  assert_eq(first.layers()[3].attention(), DynamicSparse(ReusePreviousTopK))
  assert_eq(first.layers()[3].feed_forward(), RoutedMoe)
  assert_eq(first.layers()[6].attention(), DynamicSparse(ComputeTopK))
  assert_true(first.kda() is None)
  assert_true(first.hyper_connection() is None)
  assert_true(first.vision_conditioning() is None)
  assert_eq(first.moe().routed_expert_count(), 256)
  assert_eq(first.moe().routed_scaling_factor(), 2.5)
  assert_eq(first.moe().score_function(), Sigmoid)
  assert_eq(first.moe().activation(), SwiGlu)
  assert_true(first.moe_router_float32())
  assert_true(first.moe_correction_bias())
  assert_eq(first.auxiliary_prediction(), MultiTokenPrediction(layer_count=1))
}

///|
test "Flash GLM-5.3 builds exact KDA DSA mHC vision and MTP plan" {
  let plan = @glm53.build_flash_execution_plan(flash_metadata())
  assert_eq(plan.profile_identity(), "zai-org/GLM-5.3-Flash-BF16:glm5_next")
  assert_eq(plan.decoder_layer_count(), 45)
  assert_eq(plan.layers().length(), 45)
  let mut kda_layers = 0
  let mut dsa_layers = 0
  for layer in plan.layers() {
    match layer.attention() {
      KimiDelta => kda_layers += 1
      DynamicSparse(ComputeTopK) => dsa_layers += 1
      DynamicSparse(ReusePreviousTopK) =>
        fail("Flash unexpectedly reused a DSA top-k selection")
    }
  }
  assert_eq(kda_layers, 34)
  assert_eq(dsa_layers, 11)
  guard plan.kda() is Some(kda) else { fail("Flash omitted KDA geometry") }
  assert_eq(kda.projected_width(), 8192)
  assert_eq(kda.recurrent_matrix_elements_per_sequence(), 1048576L)
  assert_eq(kda.convolution_history_tokens(), 3)
  assert_eq(kda.gate_lower_bound(), -5.0)
  guard plan.hyper_connection() is Some(hyper) else {
    fail("Flash omitted mHC geometry")
  }
  assert_eq(hyper.width(), 4)
  assert_eq(hyper.epsilon(), 0.000001)
  assert_eq(hyper.sinkhorn_iterations(), 20)
  assert_eq(plan.moe().routed_expert_count(), 288)
  assert_eq(plan.moe().activation(), ClampedSwiGlu(10.0))
  guard plan.vision_conditioning() is Some(vision) else {
    fail("Flash omitted vision conditioning")
  }
  assert_eq(vision.layer_count(), 24)
  assert_eq(vision.projection_intermediate_size(), 10240)
  assert_eq(vision.image_token_id(), 154854)
  assert_eq(vision.video_end_token_id(), 154833)
  guard plan.sparse_attention().index_pooling() is Some(pooling) else {
    fail("Flash omitted DSA index pooling")
  }
  assert_eq(pooling.kernel_size(), 4)
  assert_true(pooling.compress())
  assert_true(pooling.always_select_tail())
}

///|
test "GLM execution identity binds authenticated model content" {
  let first = @glm53.build_full_execution_plan(full_metadata())
  let alternate = @glm53.build_full_execution_plan(
    @glm53_spec.Glm53FullMetadata::from_verified_content(
      @glm53_spec.Glm53FullSpec::official(),
      try! @spec.ContentDigest::from_sha256(
        "f123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      ),
    ),
  )
  assert_true(first.identity() != alternate.identity())
  assert_true(first.identity().plan() != alternate.identity().plan())
}
