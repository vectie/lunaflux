///|
fn audio_second_upsample_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      AudioVaeDecoderUpsampleTransition(_, _, _, _, _, _, _, _, _, _) =>
        return ordinal
      _ => ()
    }
  }
  abort("missing second AudioVAE upsampler")
}

///|
test "official profiles bind exact staged F32 second AudioVAE upsampler" {
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
    let parameters = official_audio_vae_decoder_second_upsample(
      variant, request,
    )
    assert_eq(parameters.stage_index(), 1)
    assert_eq(parameters.decoder_batch_count(), 2)
    assert_eq(parameters.input_channels(), 512)
    assert_eq(parameters.output_channels(), 256)
    assert_eq(parameters.input_time(), 1035)
    assert_eq(parameters.output_time(), 5175)
    assert_eq(parameters.kernel_size(), 9)
    assert_eq(parameters.stride(), 5)
    assert_eq(parameters.padding(), 2)
    assert_eq(parameters.axes(), PerInputChannelOutputAndKernel)
    assert_eq(parameters.weight_g_element_count(), 512L)
    assert_eq(parameters.weight_v_element_count(), 1179648L)
    assert_eq(parameters.bias_element_count(), 256L)
    assert_eq(parameters.output_element_count(), 2649600L)
    let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=audio_second_upsample_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      candidate.numerical_contract(),
      OrderedF32PerInputWeightNormConvTranspose1dV1,
    )
    assert_eq(candidate.operands().length(), 6)
    assert_eq(candidate.operands()[0].byte_count(), 4239360L)
    assert_eq(candidate.operands()[1].byte_count(), 2048L)
    assert_eq(candidate.operands()[2].byte_count(), 4718592L)
    assert_eq(candidate.operands()[3].byte_count(), 1024L)
    assert_eq(candidate.operands()[4].byte_count(), 4718592L)
    assert_eq(candidate.operands()[5].byte_count(), 10598400L)
    assert_eq(candidate.stages().length(), 2)
    assert_eq(candidate.stages()[0].kind(), NormalizeWeight)
    assert_eq(candidate.stages()[0].dimensions().grid_x(), 512)
    assert_eq(candidate.stages()[0].dimensions().block_x(), 1)
    assert_eq(candidate.stages()[1].kind(), ConvolveTranspose)
    assert_eq(candidate.stages()[1].dimensions().grid_x(), 10350)
    assert_eq(candidate.stages()[1].dimensions().block_x(), 256)
    assert_false(candidate.manifest_bindable())
    let _ = try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_upsample_transition(
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
      #|  "43f822d0c7def45937d1766b170f5b25b5e8fac023dbdca96c4418be98c4e70f",
      #|  "6f3995dd4a86c8ee6b6f6976d0a7861a44eaa551d900c71fd59a36f0bff47fd4",
      #|]
    ),
  )
  debug_inspect(
    source_digests,
    content=(
      #|[
      #|  "68e280c430c25b870c304b0dc5e5b005642a21ea9ceebbeeec5a6600669c782b",
      #|  "68e280c430c25b870c304b0dc5e5b005642a21ea9ceebbeeec5a6600669c782b",
      #|]
    ),
  )
  debug_inspect(
    recipe_digests,
    content=(
      #|[
      #|  "46c0bc264f04bb8302bd7d89836efb821bcaf85a0b73d86dca05bd7d86ab3fcc",
      #|  "d871e4c5cc515f774371c676601fbd7b45817a7d4d1bddb1c3a1741f65afacd0",
      #|]
    ),
  )
}

///|
test "second AudioVAE upsampler rejects compiler ordinal identity and limit substitutions" {
  let requirements = official_requirements(Fl2va)
  let request = try! requirements
    .shape_contract()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  let parameters = official_audio_vae_decoder_second_upsample(Fl2va, request)
  let ordinal = audio_second_upsample_ordinal(requirements)
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
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
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal - 1,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal rejection")
  } noraise {
    _ => fail("first AMP stage substituted")
  }
  let tiny = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=2649599L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
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
    _ => fail("oversized transition admitted")
  }
  let foreign = official_requirements(Ref2va)
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_upsample_transition_cuda_aot_candidate(
      foreign,
      parameters~,
      requirement_ordinal=audio_second_upsample_ordinal(foreign),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("wrong identity rejection")
  } noraise {
    _ => fail("foreign-plan transition admitted")
  }
}
