///|
fn audio_input_projection_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      AudioVaeDecoderInputProjection(_, _, _, _, _, _) => return ordinal
      _ => ()
    }
  }
  abort("missing audio decoder input projection")
}

///|
test "official profiles bind exact F32 AudioVAE decoder input projection" {
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
    let parameters = official_audio_vae_decoder_input_projection(
      variant, request,
    )
    assert_eq(parameters.decoder_batch_count(), 2)
    assert_eq(parameters.input_channels(), 32)
    assert_eq(parameters.output_channels(), 2048)
    assert_eq(parameters.kernel_size(), 1)
    assert_eq(parameters.weight_name(), "dec_in_proj.weight")
    assert_eq(parameters.bias_name(), "dec_in_proj.bias")
    assert_eq(parameters.input_element_count(), 13248L)
    assert_eq(parameters.weight_element_count(), 65536L)
    assert_eq(parameters.bias_element_count(), 2048L)
    assert_eq(parameters.output_element_count(), 847872L)
    let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_input_projection_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=audio_input_projection_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      candidate.numerical_contract(),
      OrderedF32ChannelMajorPointwiseProjectionV1,
    )
    assert_eq(candidate.dimensions().grid_x(), 3312)
    assert_eq(candidate.operands()[0].byte_count(), 52992L)
    assert_eq(candidate.operands()[1].byte_count(), 262144L)
    assert_eq(candidate.operands()[2].byte_count(), 8192L)
    assert_eq(candidate.operands()[3].byte_count(), 3391488L)
    assert_false(candidate.manifest_bindable())
    parameter_digests.push(parameters.digest().as_hex())
    source_digests.push(candidate.source_digest().as_hex())
    recipe_digests.push(candidate.recipe_digest().as_hex())
  }
  debug_inspect(
    parameter_digests,
    content=(
      #|[
      #|  "5cf4069bfd4b955ea29c359a969a436ddddc26c188405c219a9a8cf1f3a38255",
      #|  "e1c2e7de529895f45ade249b1bfe248949c8ada7d6ac8d52074ac2950987daae",
      #|]
    ),
  )
  debug_inspect(
    source_digests,
    content=(
      #|[
      #|  "4c7667a82256e1139c0c801ef1ae6a89c603c9f12966dfbcd546656fdb497337",
      #|  "4c7667a82256e1139c0c801ef1ae6a89c603c9f12966dfbcd546656fdb497337",
      #|]
    ),
  )
  debug_inspect(
    recipe_digests,
    content=(
      #|[
      #|  "fb0c2ad25ebc813b8de49cd6fb47481ba7745dd40255beb6b063ed36b2dc6551",
      #|  "a6dbd96006ef1cf45c014043460c8af2b4f98fdbaba38bf5f90aff310a40a01c",
      #|]
    ),
  )
}

///|
test "AudioVAE decoder input projection rejects compiler ordinal identity and limit substitutions" {
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
  let parameters = official_audio_vae_decoder_input_projection(variant, request)
  let ordinal = audio_input_projection_ordinal(requirements)
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_input_projection_cuda_aot_candidate(
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
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_input_projection_cuda_aot_candidate(
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
    _ => fail("remaining decoder body weakened into projection")
  }
  let tiny = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=847871L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_input_projection_cuda_aot_candidate(
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
    _ => fail("oversized projection admitted")
  }
  let foreign_variant : @minimax_h3_spec.MiniMaxH3Variant = Ref2va
  let foreign = official_requirements(foreign_variant)
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_input_projection_cuda_aot_candidate(
      foreign,
      parameters~,
      requirement_ordinal=audio_input_projection_ordinal(foreign),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("wrong identity rejection")
  } noraise {
    _ => fail("foreign plan projection admitted")
  }
}
