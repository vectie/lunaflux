///|
fn audio_conv_pre_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      AudioVaeDecoderConvPre(_, _, _, _, _, _, _, _, _) => return ordinal
      _ => ()
    }
  }
  abort("missing audio decoder conv_pre")
}

///|
test "official profiles bind exact staged F32 AudioVAE decoder conv_pre" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  let parameter_digests : Array[String] = []
  let source_digests : Array[String] = []
  let recipe_digests : Array[String] = []
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
    let parameters = official_audio_vae_decoder_conv_pre(variant, request)
    assert_eq(parameters.decoder_batch_count(), 2)
    assert_eq(parameters.input_channels(), 2048)
    assert_eq(parameters.output_channels(), 1024)
    assert_eq(parameters.kernel_size(), 7)
    assert_eq(parameters.padding(), 3)
    assert_eq(parameters.stride(), 1)
    assert_eq(parameters.weight_g_name(), "decoder.conv_pre.weight_g")
    assert_eq(parameters.weight_v_name(), "decoder.conv_pre.weight_v")
    assert_eq(parameters.bias_name(), "decoder.conv_pre.bias")
    assert_eq(parameters.input_element_count(), 847872L)
    assert_eq(parameters.weight_g_element_count(), 1024L)
    assert_eq(parameters.weight_v_element_count(), 14680064L)
    assert_eq(parameters.bias_element_count(), 1024L)
    assert_eq(parameters.output_element_count(), 423936L)
    let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_conv_pre_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=audio_conv_pre_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      candidate.numerical_contract(),
      OrderedF32PerOutputWeightNormConv1dV1,
    )
    assert_eq(candidate.operands().length(), 6)
    assert_eq(candidate.operands()[0].byte_count(), 3391488L)
    assert_eq(candidate.operands()[1].byte_count(), 4096L)
    assert_eq(candidate.operands()[2].byte_count(), 58720256L)
    assert_eq(candidate.operands()[3].byte_count(), 4096L)
    assert_eq(candidate.operands()[4].byte_count(), 58720256L)
    assert_eq(candidate.operands()[5].byte_count(), 1695744L)
    assert_eq(candidate.stages().length(), 2)
    assert_eq(candidate.stages()[0].kind(), NormalizeWeight)
    assert_eq(candidate.stages()[0].dimensions().grid_x(), 1024)
    assert_eq(candidate.stages()[0].dimensions().block_x(), 1)
    assert_eq(
      candidate.stages()[0].function_symbol(),
      "lunaflux_joint_diffusion_audio_vae_conv_pre_normalize_v1",
    )
    assert_eq(candidate.stages()[1].kind(), Convolve)
    assert_eq(candidate.stages()[1].dimensions().grid_x(), 1656)
    assert_eq(candidate.stages()[1].dimensions().block_x(), 256)
    assert_eq(
      candidate.stages()[1].function_symbol(),
      "lunaflux_joint_diffusion_audio_vae_conv_pre_execute_v1",
    )
    assert_false(candidate.manifest_bindable())
    let _ = try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_conv_pre(
      candidate,
    )
    parameter_digests.push(parameters.digest().as_hex())
    source_digests.push(candidate.source_digest().as_hex())
    recipe_digests.push(candidate.recipe_digest().as_hex())
  }
  debug_inspect(
    parameter_digests,
    content=(
      #|[
      #|  "b2bd5ae57c92473018f11871c326f647030eea479488c1fef239958dda74d648",
      #|  "777d21817b68292d853898cf9dd2ee5ead026ad50889b95fa92e0ee3d524a067",
      #|]
    ),
  )
  debug_inspect(
    source_digests,
    content=(
      #|[
      #|  "a25706f882ce5368d0bfa34cec497849dc7ae3e52f488dd4814ebc7d2e116046",
      #|  "a25706f882ce5368d0bfa34cec497849dc7ae3e52f488dd4814ebc7d2e116046",
      #|]
    ),
  )
  debug_inspect(
    recipe_digests,
    content=(
      #|[
      #|  "e7baa84f40b293b0071aeaafa315f5402ef838ded268d194bc5abfb503ee82be",
      #|  "18d610718fba23940b71232813630b90844010638943e9da3810fc85ce55ed6c",
      #|]
    ),
  )
}

///|
test "AudioVAE decoder conv_pre rejects compiler ordinal identity and limit substitutions" {
  let variant : @minimax_h3_spec.MiniMaxH3Variant = Fl2va
  let requirements = official_requirements(variant)
  let request = try! requirements
    .shape_contract()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  let parameters = official_audio_vae_decoder_conv_pre(variant, request)
  let ordinal = audio_conv_pre_ordinal(requirements)
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_conv_pre_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal,
      target=integration_target(),
      compiler=unsafe_compiler,
      limits=integration_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler rejection")
  } noraise {
    _ => fail("FMA compiler admitted")
  }
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_conv_pre_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal + 1,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong remaining-body rejection")
  } noraise {
    _ => fail("remaining decoder body weakened into conv_pre")
  }
  let tiny = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=14680063L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_conv_pre_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=tiny,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong limit rejection")
  } noraise {
    _ => fail("oversized conv_pre admitted")
  }
  let foreign = official_requirements(Ref2va)
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_conv_pre_cuda_aot_candidate(
      foreign,
      parameters~,
      requirement_ordinal=audio_conv_pre_ordinal(foreign),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("wrong identity rejection")
  } noraise {
    _ => fail("foreign plan conv_pre admitted")
  }
}
