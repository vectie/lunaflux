///|
fn flash_0731_fixture_plan() -> @deepseek_v4.DeepSeekV4SemanticPlan raise {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "7777777777777777777777777777777777777777777777777777777777777777",
    ),
  )
  @deepseek_v4.build(metadata)
}

///|
fn fixture_limits() -> @advanced_decoder_reference.ReferenceLimits raise {
  @advanced_decoder_reference.ReferenceLimits::new(
    max_values=20000,
    max_width=256,
    max_iterations=32,
  )
}

///|
test "Flash-0731 fixture exercises exact routed, compressed, mHC, and DSpark semantics" {
  let plan = flash_0731_fixture_plan()
  let result = @deepseek_v4_reference_fixture.run_flash_0731(
    plan,
    limits=fixture_limits(),
  )
  assert_eq(result.plan_digest(), plan.execution_plan().plan_digest())
  assert_eq(result.top_k_routing().expert_indices(), [5, 7, 11, 13, 17, 19])
  assert_eq(result.hash_routing().expert_indices(), [19, 17, 13, 11, 7, 5])
  assert_eq(result.expert_output().length(), 2)
  assert_true(result.expert_output()[0] > 0.25F)
  assert_eq(result.expert_output()[1], result.expert_output()[0] * 2.0F)
  assert_eq(result.learned_indices().indices(), [1, 0])
  assert_eq(result.learned_indices().scores(), [2.0F, 1.0F])
  assert_eq(result.compressed_indices(), [7, 8])
  assert_eq(result.mixed_streams(), [
    12.000012397766113F, 15.000017166137695F, 12.000012397766113F, 15.000017166137695F,
    12.000012397766113F, 15.000017166137695F, 12.000012397766113F, 15.000017166137695F,
  ])
  assert_eq(result.auxiliary_stages(), [
    DsparkMainProjection,
    DsparkNoiseEmbedding,
    AuxiliaryDecoderStage(0),
    AuxiliaryDecoderStage(1),
    AuxiliaryDecoderStage(2),
    AuxiliaryOutputHead,
    DsparkMarkovAdjustment,
    DsparkConfidence,
  ])
}

///|
test "fixture rejects another closed DeepSeek profile" {
  let spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    spec,
    @spec.ContentDigest::from_sha256(
      "7777777777777777777777777777777777777777777777777777777777777777",
    ),
  )
  try
    @deepseek_v4_reference_fixture.run_flash_0731(
      @deepseek_v4.build(metadata),
      limits=fixture_limits(),
    )
  catch {
    @deepseek_v4_reference_fixture.UnsupportedFixtureProfile => ()
    _ => fail("wrong profile did not fail closed")
  } noraise {
    _ => fail("wrong profile was admitted")
  }
}

///|
test "fixture wraps hostile reference bounds without partial output" {
  let tiny = @advanced_decoder_reference.ReferenceLimits::new(
    max_values=128,
    max_width=256,
    max_iterations=32,
  )
  try
    @deepseek_v4_reference_fixture.run_flash_0731(
      flash_0731_fixture_plan(),
      limits=tiny,
    )
  catch {
    @deepseek_v4_reference_fixture.ReferenceFailure(InvalidDimension) => ()
    _ => fail("hostile bound produced the wrong typed error")
  } noraise {
    _ => fail("hostile bound was ignored")
  }
}
