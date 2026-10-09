///|
fn audio_first_upsample_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      AudioVaeDecoderFirstUpsample(_, _, _, _, _, _, _, _, _) => return ordinal
      _ => ()
    }
  }
  abort("missing first AudioVAE upsampler")
}

///|
test "official profiles bind exact staged F32 first AudioVAE upsampler" {
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
    let parameters = official_audio_vae_decoder_first_upsample(variant, request)
    assert_eq(parameters.decoder_batch_count(), 2)
    assert_eq(parameters.input_channels(), 1024)
    assert_eq(parameters.output_channels(), 512)
    assert_eq(parameters.input_time(), 207)
    assert_eq(parameters.output_time(), 1035)
    assert_eq(parameters.kernel_size(), 9)
    assert_eq(parameters.stride(), 5)
    assert_eq(parameters.padding(), 2)
    assert_eq(parameters.axes(), PerInputChannelOutputAndKernel)
    assert_true(parameters.axes() != PerOutputChannelInputAndKernel)
    assert_eq(parameters.weight_g_element_count(), 1024L)
    assert_eq(parameters.weight_v_element_count(), 4718592L)
    assert_eq(parameters.bias_element_count(), 512L)
    assert_eq(parameters.output_element_count(), 1059840L)
    let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_upsample_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=audio_first_upsample_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      candidate.numerical_contract(),
      OrderedF32PerInputWeightNormConvTranspose1dV1,
    )
    assert_eq(candidate.operands().length(), 6)
    assert_eq(candidate.operands()[0].byte_count(), 1695744L)
    assert_eq(candidate.operands()[1].byte_count(), 4096L)
    assert_eq(candidate.operands()[2].byte_count(), 18874368L)
    assert_eq(candidate.operands()[3].byte_count(), 2048L)
    assert_eq(candidate.operands()[4].byte_count(), 18874368L)
    assert_eq(candidate.operands()[5].byte_count(), 4239360L)
    assert_eq(candidate.stages().length(), 2)
    assert_eq(candidate.stages()[0].kind(), NormalizeWeight)
    assert_eq(candidate.stages()[0].dimensions().grid_x(), 1024)
    assert_eq(candidate.stages()[0].dimensions().block_x(), 1)
    assert_eq(candidate.stages()[1].kind(), ConvolveTranspose)
    assert_eq(candidate.stages()[1].dimensions().grid_x(), 4140)
    assert_eq(candidate.stages()[1].dimensions().block_x(), 256)
    assert_false(candidate.manifest_bindable())
    let _ = try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_first_upsample(
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
      #|  "523acc3cf1416a3f7c68121a670d4b7b54b47cb70ce86f9802c6015316bef664",
      #|  "c7c66c5ebfba1d6a6e90f3f1134765e290bfb47ad3278b1bb176e2c235fc7a6f",
      #|]
    ),
  )
  debug_inspect(
    source_digests,
    content=(
      #|[
      #|  "8edb61586620a9491cd554f3e545b22bca515f3d2c21a193269d7782d741f2ef",
      #|  "8edb61586620a9491cd554f3e545b22bca515f3d2c21a193269d7782d741f2ef",
      #|]
    ),
  )
  debug_inspect(
    recipe_digests,
    content=(
      #|[
      #|  "21e2f6c6b2ac11abe08eeff7de57846aafc2871ecd0d297179dfdd11e4ff83c6",
      #|  "fa427879d2022bcd6aa81bc5a0eaae0a5e33187f316c38a1ab93c5bb34cc384b",
      #|]
    ),
  )
}

///|
test "first AudioVAE upsampler rejects compiler ordinal identity and limit substitutions" {
  let requirements = official_requirements(Fl2va)
  let request = try! requirements
    .shape_contract()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  let parameters = official_audio_vae_decoder_first_upsample(Fl2va, request)
  let ordinal = audio_first_upsample_ordinal(requirements)
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_upsample_cuda_aot_candidate(
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
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_upsample_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal - 1,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ConvPre ordinal rejection")
  } noraise {
    _ => fail("ConvPre requirement substituted")
  }
  let tiny = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=4718591L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_upsample_cuda_aot_candidate(
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
    _ => fail("oversized upsampler admitted")
  }
  let foreign = official_requirements(Ref2va)
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_upsample_cuda_aot_candidate(
      foreign,
      parameters~,
      requirement_ordinal=audio_first_upsample_ordinal(foreign),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("wrong identity rejection")
  } noraise {
    _ => fail("foreign-plan upsampler admitted")
  }
}
