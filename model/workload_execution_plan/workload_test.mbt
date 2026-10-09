///|
fn test_content(seed : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, seed))
}

///|
fn identity(
  content : @spec.ContentDigest,
  plan : @spec.PlanDigest,
) -> @spec.ModelIdentity {
  try! @spec.ModelIdentity::new(content, plan)
}

///|
test "requested families project into workload topology without family types" {
  let deepseek_spec = @deepseek_v4_spec.DeepSeekV4ModelSpec::from_official_model_id(
    "deepseek-ai/DeepSeek-V4-Flash",
  )
  let deepseek_metadata = @deepseek_v4_spec.DeepSeekV4ModelMetadata::from_verified_content(
    deepseek_spec,
    test_content('1'),
  )
  let deepseek_plan = @deepseek_v4.build(deepseek_metadata).execution_plan()
  let deepseek = @workload_execution_plan.WorkloadExecutionPlan::decoder_only(
    identity=identity(
      deepseek_plan.content_digest(),
      deepseek_plan.plan_digest(),
    ),
    schema_version=deepseek_plan.schema_version(),
    decoder_layer_count=deepseek_plan.shape().decoder_layer_count(),
    maximum_token_context=deepseek_plan.shape().maximum_position_count(),
  )
  assert_eq(deepseek.kind(), DecoderOnlyText)
  assert_eq(deepseek.decoder_layer_count(), Some(43))
  assert_eq(deepseek.maximum_token_context(), Some(1048576))
  assert_eq(deepseek.identity().content(), test_content('1'))

  let glm_full_metadata = @glm53_spec.Glm53FullMetadata::from_verified_content(
    @glm53_spec.Glm53FullSpec::official(),
    test_content('2'),
  )
  let glm_full_plan = @glm53.build_full_execution_plan(glm_full_metadata)
  let glm_full = @workload_execution_plan.WorkloadExecutionPlan::decoder_only(
    identity=glm_full_plan.identity(),
    schema_version=glm_full_plan.schema_version(),
    decoder_layer_count=glm_full_plan.decoder_layer_count(),
    maximum_token_context=glm_full_plan.context_length(),
  )
  assert_eq(glm_full.kind(), DecoderOnlyText)
  assert_eq(glm_full.decoder_layer_count(), Some(78))

  let glm_flash_metadata = @glm53_spec.Glm53FlashMetadata::from_verified_content(
    @glm53_spec.Glm53FlashSpec::official(),
    test_content('3'),
  )
  let glm_flash_plan = @glm53.build_flash_execution_plan(glm_flash_metadata)
  let glm_flash = @workload_execution_plan.WorkloadExecutionPlan::multimodal_conditional(
    identity=glm_flash_plan.identity(),
    schema_version=glm_flash_plan.schema_version(),
    decoder_layer_count=glm_flash_plan.decoder_layer_count(),
    maximum_token_context=glm_flash_plan.context_length(),
  )
  assert_eq(glm_flash.kind(), MultimodalConditionalGeneration)
  assert_eq(glm_flash.decoder_layer_count(), Some(45))

  let minimax_metadata = @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
    @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
      variant=Fl2va,
      checkpoint_format=DiffusersV1,
    ),
    test_content('4'),
  )
  let minimax_plan = @minimax_h3.build(minimax_metadata)
  let minimax = @workload_execution_plan.WorkloadExecutionPlan::joint_audio_video_diffusion(
    identity=identity(minimax_plan.content_digest(), minimax_plan.plan_digest()),
    schema_version=minimax_plan.schema_version(),
  )
  assert_eq(minimax.kind(), JointAudioVideoDiffusion)
  assert_eq(minimax.decoder_layer_count(), None)
  assert_eq(minimax.maximum_token_context(), None)
  assert_eq(minimax.identity().content(), test_content('4'))
}

///|
test "workload envelope rejects topology-shape confusion" {
  let model_identity = identity(
    test_content('5'),
    try! @spec.PlanDigest::from_sha256(String::make(64, '6')),
  )
  try
    @workload_execution_plan.WorkloadExecutionPlan::decoder_only(
      identity=model_identity,
      schema_version=1,
      decoder_layer_count=0,
      maximum_token_context=1024,
    )
  catch {
    InvalidDecoderContract => ()
    _ => fail("unexpected invalid decoder error")
  } noraise {
    _ => fail("zero-layer decoder was admitted")
  }
}
