///|
fn second_amp_activation_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is AudioVaeDecoderAmpActivation(_, _, _, _, _, _, _) {
      return ordinal
    }
  }
  abort("missing second AMP activation")
}

///|
fn second_amp_stage_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation()
      is AudioVaeDecoderAmpStage(_, _, _, _, _, _, _, _) {
      return ordinal
    }
  }
  abort("missing second AMP stage")
}

///|
test "official profiles bind exact second alias-free activation and 97-launch AMP stage" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  let activation_recipes : Array[String] = []
  let activation_parameters : Array[String] = []
  let stage_recipes : Array[String] = []
  let activation_sources : Array[String] = []
  let stage_sources : Array[String] = []
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
    let activation = official_audio_vae_decoder_second_amp_activation(
      variant, request,
    )
    assert_eq(activation.stage_index(), 1)
    assert_eq(activation.channel_count(), 256)
    assert_eq(activation.time_count(), 5175)
    assert_eq(activation.upsample_time(), 10350)
    assert_eq(activation.element_count(), 2649600L)
    assert_eq(activation.intermediate_element_count(), 5299200L)
    let activation_candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_activation_cuda_aot_candidate(
      requirements,
      parameters=activation,
      requirement_ordinal=second_amp_activation_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(activation_candidate.operands().length(), 8)
    assert_eq(activation_candidate.operands()[0].byte_count(), 10598400L)
    assert_eq(activation_candidate.operands()[1].byte_count(), 1024L)
    assert_eq(activation_candidate.operands()[2].byte_count(), 1024L)
    assert_eq(activation_candidate.operands()[3].byte_count(), 48L)
    assert_eq(activation_candidate.operands()[4].byte_count(), 48L)
    assert_eq(activation_candidate.operands()[5].byte_count(), 21196800L)
    assert_eq(activation_candidate.operands()[6].byte_count(), 21196800L)
    assert_eq(activation_candidate.operands()[7].byte_count(), 10598400L)
    assert_eq(activation_candidate.stages().length(), 3)
    assert_eq(activation_candidate.stages()[0].dimensions().grid_x(), 20700)
    assert_eq(activation_candidate.stages()[2].dimensions().grid_x(), 10350)
    assert_false(activation_candidate.manifest_bindable())
    ignore(
      try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_amp_activation(
        activation_candidate,
      ),
    )
    let parameters = try! @minimax_h3_vae_reference.bind_second_amp_parameter_layout()
    let workspace = try! @joint_audio_amp_workspace_layout.amp_workspace_layout(
      element_count=2649600L,
      intermediate_element_count=5299200L,
      normalized_weight_element_count=720896L,
    )
    assert_eq(parameters.total_arena_bytes(), 33106176L)
    assert_eq(workspace.workspace_bytes(), 119465984L)
    let stage_candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_stage_cuda_aot_candidate(
      requirements,
      request~,
      parameter_layout=parameters,
      workspace_layout=workspace,
      requirement_ordinal=second_amp_stage_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      stage_candidate.numerical_contract(),
      OrderedF32AmpParallelResidualAverageV1,
    )
    assert_eq(stage_candidate.operands().length(), 5)
    assert_eq(stage_candidate.operands()[0].byte_count(), 10598400L)
    assert_eq(stage_candidate.operands()[1].byte_count(), 10598400L)
    assert_eq(stage_candidate.operands()[2].byte_count(), 33106176L)
    assert_eq(stage_candidate.operands()[3].byte_count(), 119465984L)
    assert_eq(stage_candidate.operands()[4].byte_count(), 10598400L)
    assert_eq(stage_candidate.stages().length(), 97)
    assert_eq(stage_candidate.stages()[0].dimensions().grid_x(), 256)
    assert_eq(stage_candidate.stages()[1].dimensions().grid_x(), 10350)
    match stage_candidate.stages()[0].identity() {
      Convolution(0, 0, 0, Normalize) => ()
      _ => fail("precomputed activation was not consumed first")
    }
    match stage_candidate.stages()[96].identity() {
      ParallelAverage => ()
      _ => fail("parallel average missing")
    }
    assert_false(stage_candidate.manifest_bindable())
    ignore(
      try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_amp_stage(
        stage_candidate,
      ),
    )
    activation_recipes.push(activation_candidate.recipe_digest().as_hex())
    activation_parameters.push(activation.digest().as_hex())
    stage_recipes.push(stage_candidate.recipe_digest().as_hex())
    activation_sources.push(activation_candidate.source_digest().as_hex())
    stage_sources.push(stage_candidate.source_digest().as_hex())
  }
  debug_inspect(
    activation_recipes,
    content=(
      #|[
      #|  "b2615b4470dc0b789f5f1cde9105c64b9928238c6b2d28b45473f1e0ecf59b89",
      #|  "baed3c4741c0f526c93aa6b849498630146bdc823a2194acf303832e1839ba95",
      #|]
    ),
  )
  debug_inspect(
    stage_recipes,
    content=(
      #|[
      #|  "a5c111895206952cc5d7de9812512aaf550d00f392a15742c572c36333b01c2d",
      #|  "0004dbe578dfbdea0a1bc7609b64016b7b2d168a4b780a1fefb9ceb70f5317ec",
      #|]
    ),
  )
  debug_inspect(
    activation_parameters,
    content=(
      #|[
      #|  "e9d387b4d68a1acaae1b49d3519cc0dc4e5bbc584b925e1780402963bd3e2897",
      #|  "5ff548a7045e9ef27f2cfa037b37ee92c4c72326bf6c4df916f8685b1febc7bf",
      #|]
    ),
  )
  debug_inspect(
    activation_sources,
    content=(
      #|[
      #|  "b8a499dac1b6349f7f2e071a786fcd3110b489674a7e10b7ebc9f5d2b4faaf4e",
      #|  "b8a499dac1b6349f7f2e071a786fcd3110b489674a7e10b7ebc9f5d2b4faaf4e",
      #|]
    ),
  )
  debug_inspect(
    stage_sources,
    content=(
      #|[
      #|  "3dbd945bc921adc75d36fb74d3df3394bcf49c414447c3a3578d825829819208",
      #|  "3dbd945bc921adc75d36fb74d3df3394bcf49c414447c3a3578d825829819208",
      #|]
    ),
  )
}

///|
test "second AMP stage rejects ordinal compiler and parameter-geometry substitutions" {
  let requirements = official_requirements(Fl2va)
  let request = try! requirements
    .shape_contract()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  let activation = official_audio_vae_decoder_second_amp_activation(
    Fl2va,
    request,
  )
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_activation_cuda_aot_candidate(
      requirements,
      parameters=activation,
      requirement_ordinal=second_amp_activation_ordinal(requirements),
      target=integration_target(),
      compiler=unsafe_compiler,
      limits=integration_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler rejection")
  } noraise {
    _ => fail("FMA AMP activation admitted")
  }
  let first_parameters = try! @minimax_h3_vae_reference.bind_first_amp_parameter_layout()
  let workspace = try! @joint_audio_amp_workspace_layout.amp_workspace_layout(
    element_count=2649600L,
    intermediate_element_count=5299200L,
    normalized_weight_element_count=720896L,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_stage_cuda_aot_candidate(
      requirements,
      request~,
      parameter_layout=first_parameters,
      workspace_layout=workspace,
      requirement_ordinal=second_amp_stage_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Requirement) => ()
    _ => fail("wrong parameter-layout rejection")
  } noraise {
    _ => fail("512-channel parameter layout admitted")
  }
  let exact = try! @minimax_h3_vae_reference.bind_second_amp_parameter_layout()
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_amp_stage_cuda_aot_candidate(
      requirements,
      request~,
      parameter_layout=exact,
      workspace_layout=workspace,
      requirement_ordinal=second_amp_stage_ordinal(requirements) - 1,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal rejection")
  } noraise {
    _ => fail("AMP activation ordinal substituted")
  }
}
