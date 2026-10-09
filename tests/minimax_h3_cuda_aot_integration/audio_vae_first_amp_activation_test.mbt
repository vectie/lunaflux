///|
fn audio_first_amp_activation_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      AudioVaeDecoderFirstAmpActivation(_, _, _, _, _, _) => return ordinal
      _ => ()
    }
  }
  abort("missing first AMP activation")
}

///|
test "official profiles bind exact alias-free first AMP activation" {
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
    let parameters = official_audio_vae_decoder_first_amp_activation(
      variant, request,
    )
    assert_eq(parameters.decoder_batch_count(), 2)
    assert_eq(parameters.channel_count(), 512)
    assert_eq(parameters.time_count(), 1035)
    assert_eq(parameters.upsample_time(), 2070)
    assert_eq(parameters.element_count(), 1059840L)
    assert_eq(parameters.intermediate_element_count(), 2119680L)
    assert_true(
      parameters.upsample_filter_name() != parameters.downsample_filter_name(),
    )
    let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_activation_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=audio_first_amp_activation_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(candidate.numerical_contract(), OrderedF32AliasFreeSnakeBetaV1)
    assert_eq(candidate.operands().length(), 8)
    assert_eq(candidate.operands()[0].byte_count(), 4239360L)
    assert_eq(candidate.operands()[1].byte_count(), 2048L)
    assert_eq(candidate.operands()[2].byte_count(), 2048L)
    assert_eq(candidate.operands()[3].byte_count(), 48L)
    assert_eq(candidate.operands()[4].byte_count(), 48L)
    assert_eq(candidate.operands()[5].byte_count(), 8478720L)
    assert_eq(candidate.operands()[6].byte_count(), 8478720L)
    assert_eq(candidate.operands()[7].byte_count(), 4239360L)
    assert_true(
      candidate.operands()[3].role() != candidate.operands()[4].role(),
    )
    assert_eq(candidate.stages().length(), 3)
    assert_eq(candidate.stages()[0].kind(), Upsample)
    assert_eq(candidate.stages()[0].dimensions().grid_x(), 8280)
    assert_eq(candidate.stages()[1].kind(), SnakeBeta)
    assert_eq(candidate.stages()[1].dimensions().grid_x(), 8280)
    assert_eq(candidate.stages()[2].kind(), Downsample)
    assert_eq(candidate.stages()[2].dimensions().grid_x(), 4140)
    assert_false(candidate.manifest_bindable())
    let _ = try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_first_amp_activation(
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
      #|  "d6747780e5d7e69aad6328f74dbe579125529c5c6030f3d0cd2cd374fd9ee6ee",
      #|  "f456cbdfe98e9293ab0c25504dd17c92a0d02562bb3f118b025961e6f556e8a4",
      #|]
    ),
  )
  debug_inspect(
    source_digests,
    content=(
      #|[
      #|  "f98628a23dbfa8c609bcd0e14caf09ed533389b863af56fdf6837872d553aae8",
      #|  "f98628a23dbfa8c609bcd0e14caf09ed533389b863af56fdf6837872d553aae8",
      #|]
    ),
  )
  debug_inspect(
    recipe_digests,
    content=(
      #|[
      #|  "24fd3b13cddee66e7f0ed42e64ef1b222677f25f2b4d1d08388d2ac46a720715",
      #|  "63bc6c510e9d71ffebcda7c5db6d2ca72529729b322555fdd7107c907657172b",
      #|]
    ),
  )
}

///|
test "first AMP activation rejects unsafe compiler and ordinal substitution" {
  let requirements = official_requirements(Fl2va)
  let request = try! requirements
    .shape_contract()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  let parameters = official_audio_vae_decoder_first_amp_activation(
    Fl2va,
    request,
  )
  let ordinal = audio_first_amp_activation_ordinal(requirements)
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_activation_cuda_aot_candidate(
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
    _ => fail("unsafe compiler admitted")
  }
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_activation_cuda_aot_candidate(
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
    _ => fail("wrong ordinal admitted")
  }
}
