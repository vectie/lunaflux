///|
fn audio_destandardization_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      AudioVaeLatentDestandardization(_, _, _, _) => return ordinal
      _ => ()
    }
  }
  abort("missing audio latent destandardization")
}

///|
test "FL2VA and Ref2VA bind exact official audio latent destandardization" {
  let parameter_digests : Array[String] = []
  let source_digests : Array[String] = []
  let recipe_digests : Array[String] = []
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
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
    let parameters = official_audio_vae_destandardization(variant, request)
    assert_eq(parameters.batch_count(), 2)
    assert_eq(parameters.channel_count(), 32)
    assert_eq(parameters.elements_per_channel(), 207L)
    assert_eq(parameters.element_count(), 13248L)
    assert_eq(parameters.layout(), BatchChannelTime)
    assert_eq(parameters.storage(), Float32InputScaleMeanOutput)
    let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=audio_destandardization_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(candidate.decoder_batch_count(), 2)
    assert_eq(candidate.channel_count(), 32)
    assert_eq(candidate.elements_per_channel(), 207L)
    assert_eq(candidate.element_count(), 13248L)
    assert_eq(candidate.dimensions().grid_x(), 52)
    assert_eq(candidate.operands()[0].byte_count(), 52992L)
    assert_eq(candidate.operands()[1].byte_count(), 128L)
    assert_eq(candidate.operands()[2].byte_count(), 128L)
    assert_eq(candidate.operands()[3].byte_count(), 52992L)
    assert_false(candidate.manifest_bindable())
    parameter_digests.push(parameters.digest().as_hex())
    source_digests.push(candidate.source_digest().as_hex())
    recipe_digests.push(candidate.recipe_digest().as_hex())
  }
  debug_inspect(
    parameter_digests,
    content=(
      #|[
      #|  "9b7eb4925c22f1293fcdba4123549c2db7faf0af8b71db03655687cc64bb8254",
      #|  "9912bfadda55ec1d2f701779fd5a16441c752f1534bc6804341c364cbcde900a",
      #|]
    ),
  )
  debug_inspect(
    source_digests,
    content=(
      #|[
      #|  "2e91ffe2cb293b4162214f491896ffd87831581c9664c069d4c409b1b5f0ca7d",
      #|  "2e91ffe2cb293b4162214f491896ffd87831581c9664c069d4c409b1b5f0ca7d",
      #|]
    ),
  )
  debug_inspect(
    recipe_digests,
    content=(
      #|[
      #|  "c4c58aeb6ed42533f679f8fca749d87a513c5d17b74e1f665ad91601c05ab3a6",
      #|  "b54afa1c1f36ae647dec7ff30f7724cdeda123fcbef53340aff5947e5994a161",
      #|]
    ),
  )
}

///|
test "audio latent scalar reference repeats coefficients across decoder batch" {
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
  let parameters = official_audio_vae_destandardization(variant, request)
  let normalized = Array::make(13248, 0.0F)
  normalized[0] = 1.0F
  normalized[32 * 207] = 1.0F
  let output = try! @joint_diffusion_vae_reference.destandardize_audio_latents(
    parameters, normalized,
  )
  assert_eq(
    output[0].reinterpret_as_uint(),
    output[32 * 207].reinterpret_as_uint(),
  )
  try
    @joint_diffusion_vae_reference.destandardize_audio_latents(
      parameters,
      normalized[:13247],
    )
  catch {
    InvalidDestandardizationInput => ()
    _ => fail("wrong hostile input rejection")
  } noraise {
    _ => fail("short audio latent input admitted")
  }
}

///|
test "audio latent candidate rejects compiler ordinal identity and limit substitutions" {
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
  let parameters = official_audio_vae_destandardization(variant, request)
  let ordinal = audio_destandardization_ordinal(requirements)
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal,
      target=integration_target(),
      compiler=unsafe_compiler,
      limits=integration_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong hostile compiler rejection")
  } noraise {
    _ => fail("FMA compiler admitted")
  }
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal + 1,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong decoder-body ordinal rejection")
  } noraise {
    _ => fail("decoder body weakened into prefix")
  }
  let tiny_limits = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=13247L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_latent_destandardization_cuda_aot_candidate(
      requirements,
      parameters~,
      requirement_ordinal=ordinal,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=tiny_limits,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong hostile limit rejection")
  } noraise {
    _ => fail("oversized audio request admitted")
  }
  let foreign_variant : @minimax_h3_spec.MiniMaxH3Variant = Ref2va
  let foreign_requirements = official_requirements(foreign_variant)
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_latent_destandardization_cuda_aot_candidate(
      foreign_requirements,
      parameters~,
      requirement_ordinal=audio_destandardization_ordinal(foreign_requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("wrong foreign identity rejection")
  } noraise {
    _ => fail("foreign plan parameters admitted")
  }
}
