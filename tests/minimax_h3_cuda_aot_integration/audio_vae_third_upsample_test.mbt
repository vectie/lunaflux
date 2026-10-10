///|
fn third_upsample_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is AudioVaeDecoderUpsampleTransition(2, _, _, _, _, _, _, _, _, _) {
      return ordinal
    }
  }
  abort("missing third AudioVAE upsampler")
}

///|
fn third_amp_activation_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is AudioVaeDecoderAmpActivation(2, _, _, _, _, _, _) {
      return ordinal
    }
  }
  abort("missing third AMP activation")
}

///|
test "official profiles bind exact decoder ups two and following AMP activation" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  let parameter_digests : Array[String] = []
  let activation_digests : Array[String] = []
  let upsample_sources : Array[String] = []
  let activation_sources : Array[String] = []
  let upsample_recipes : Array[String] = []
  let activation_recipes : Array[String] = []
  for variant in variants {
    let requirements = official_requirements(variant)
    let request = try! requirements
      .shape_contract()
      .resolve_request(
        height=768,
        width=1344,
        requested_frames=120,
        inference_steps=50,
      )
    let parameters = official_audio_vae_decoder_third_upsample(variant, request)
    assert_eq(parameters.stage_index(), 2)
    assert_eq(parameters.input_channels(), 256)
    assert_eq(parameters.output_channels(), 128)
    assert_eq(parameters.input_time(), 5175)
    assert_eq(parameters.output_time(), 10350)
    assert_eq(parameters.kernel_size(), 4)
    assert_eq(parameters.stride(), 2)
    assert_eq(parameters.padding(), 1)
    assert_eq(parameters.axes(), PerInputChannelOutputAndKernel)
    assert_eq(parameters.input_element_count(), 2649600L)
    assert_eq(parameters.weight_g_element_count(), 256L)
    assert_eq(parameters.weight_v_element_count(), 131072L)
    assert_eq(parameters.bias_element_count(), 128L)
    assert_eq(parameters.output_element_count(), 2649600L)
    let upsample = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=third_upsample_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(upsample.operands().length(), 6)
    assert_eq(upsample.operands()[0].role(), AudioDecoderAmpStageOutput)
    assert_eq(upsample.operands()[0].byte_count(), 10598400L)
    assert_eq(upsample.operands()[1].byte_count(), 1024L)
    assert_eq(upsample.operands()[2].byte_count(), 524288L)
    assert_eq(upsample.operands()[3].byte_count(), 512L)
    assert_eq(upsample.operands()[4].byte_count(), 524288L)
    assert_eq(
      upsample.operands()[5].role(),
      AudioDecoderUpsampleTransition2Output,
    )
    assert_eq(upsample.operands()[5].byte_count(), 10598400L)
    assert_eq(upsample.stages()[0].dimensions().grid_x(), 256)
    assert_eq(upsample.stages()[0].dimensions().block_x(), 1)
    assert_eq(upsample.stages()[1].dimensions().grid_x(), 10350)
    assert_eq(upsample.stages()[1].dimensions().block_x(), 256)
    assert_false(upsample.manifest_bindable())
    ignore(
      try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_upsample_transition(
        upsample,
      ),
    )
    let activation = official_audio_vae_decoder_third_amp_activation(
      variant, request,
    )
    assert_eq(activation.stage_index(), 2)
    assert_eq(activation.channel_count(), 128)
    assert_eq(activation.time_count(), 10350)
    assert_eq(activation.upsample_time(), 20700)
    assert_eq(activation.element_count(), 2649600L)
    assert_eq(activation.intermediate_element_count(), 5299200L)
    let amp = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_activation_cuda_aot_candidate(
      requirements,
      parameters=activation,
      requirement_ordinal=third_amp_activation_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(amp.operands()[0].role(), AudioDecoderUpsampleTransition2Output)
    assert_eq(amp.operands()[1].byte_count(), 512L)
    assert_eq(amp.operands()[2].byte_count(), 512L)
    assert_eq(amp.operands()[3].byte_count(), 48L)
    assert_eq(amp.operands()[4].byte_count(), 48L)
    assert_eq(amp.operands()[5].byte_count(), 21196800L)
    assert_eq(amp.operands()[6].byte_count(), 21196800L)
    assert_eq(amp.operands()[7].role(), AudioDecoderAmpActivation2Output)
    assert_eq(amp.operands()[7].byte_count(), 10598400L)
    assert_eq(amp.stages()[0].dimensions().grid_x(), 20700)
    assert_eq(amp.stages()[2].dimensions().grid_x(), 10350)
    assert_false(amp.manifest_bindable())
    ignore(
      try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_amp_activation(
        amp,
      ),
    )
    parameter_digests.push(parameters.digest().as_hex())
    activation_digests.push(activation.digest().as_hex())
    upsample_sources.push(upsample.source_digest().as_hex())
    activation_sources.push(amp.source_digest().as_hex())
    upsample_recipes.push(upsample.recipe_digest().as_hex())
    activation_recipes.push(amp.recipe_digest().as_hex())
  }
  debug_inspect(
    parameter_digests,
    content=(
      #|[
      #|  "8240f4f745ff7db292cda2a4d0e0e347cafe389c9f2208179e1e0cd3f8f20441",
      #|  "015307366fb8a0682e7406a705eb2109d1e8a4d69c8a5cd5bd43735a2ad35029",
      #|]
    ),
  )
  debug_inspect(
    activation_digests,
    content=(
      #|[
      #|  "a855d4ec0ebc7e38e704e9030870a09193b63c150a4b178e4b162b0e2927a4c2",
      #|  "fe00ad5d87b0a5688b9ae92cd8eea57ed8cf42d8eeee43bdb2eb213b725f5194",
      #|]
    ),
  )
  debug_inspect(
    upsample_sources,
    content=(
      #|[
      #|  "63be6e183d633cf5a10f7a812058cf2de05f688d46dc221bf172685a03905b77",
      #|  "63be6e183d633cf5a10f7a812058cf2de05f688d46dc221bf172685a03905b77",
      #|]
    ),
  )
  debug_inspect(
    activation_sources,
    content=(
      #|[
      #|  "130e2d23a6cb94b753ddb4f83186eaabfc650814c33d69148bf0692474c49494",
      #|  "130e2d23a6cb94b753ddb4f83186eaabfc650814c33d69148bf0692474c49494",
      #|]
    ),
  )
  debug_inspect(
    upsample_recipes,
    content=(
      #|[
      #|  "f4ae9131dbebca02936de5b317cc9c15b90197bc1362286adcd5fe2fa4f43ad7",
      #|  "a3f7a2a507585c570c8370bbd26a8ed93b849ee96ffa3abdbcffca9a1e620a3e",
      #|]
    ),
  )
  debug_inspect(
    activation_recipes,
    content=(
      #|[
      #|  "19ea19ae1efd0663d8425d39d05f9a6776dcc1de72d0918fda1d1a3a830ca2c9",
      #|  "dfd58c542afdf416323f5b61e3d320f91377f316e1db6ed0149e0e3482670247",
      #|]
    ),
  )
}

///|
test "third AudioVAE upsample and activation reject stage and limit substitutions" {
  let requirements = official_requirements(Fl2va)
  let request = try! requirements
    .shape_contract()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  let second = official_audio_vae_decoder_second_upsample(Fl2va, request)
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
      requirements,
      parameters=second,
      requirement_ordinal=third_upsample_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("wrong stage rejection")
  } noraise {
    _ => fail("stage-one upsampler substituted")
  }
  let activation = official_audio_vae_decoder_third_amp_activation(
    Fl2va,
    request,
  )
  let tiny = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=5299199L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_activation_cuda_aot_candidate(
      requirements,
      parameters=activation,
      requirement_ordinal=third_amp_activation_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=tiny,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong activation limit rejection")
  } noraise {
    _ => fail("oversized activation admitted")
  }
}
