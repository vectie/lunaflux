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
      #|  "dba7ace424f3ee08aee35139ac9679ae4d32c65ca834dae799e634bb5f18ac03",
      #|  "831accb54acf0d35dae9b5989b6b1cf80d40af10a70a2734166379c52f68d4ec",
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
      #|  "f7483be29b804ae5e561363d7bd8c2b46ab4412826df5f77123e6b669459fde7",
      #|  "a40c8afbb604e1ddbe680c3de90fa3e70b12c792ce387cdd39a05336ca5e5ca9",
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
