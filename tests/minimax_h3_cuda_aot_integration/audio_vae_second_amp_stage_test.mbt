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
      #|  "59b6d3c7401eb78e8bef020cf25e490990134a339c2f3b82b0dbeceee0aa3206",
      #|  "c5cfa906fcce9d203c716a6d3162e6476ff484a53a80670749703456859ce9ca",
      #|]
    ),
  )
  debug_inspect(
    stage_recipes,
    content=(
      #|[
      #|  "fefdda7bca4f87ec40b5378f3f2176b5c1dc502a4e3606eebad569b6296d80c4",
      #|  "2c4080d47d11ab0f2991cfe82385fceabe3d5ac23d9c006b13e3d2023dcc7888",
      #|]
    ),
  )
  debug_inspect(
    activation_parameters,
    content=(
      #|[
      #|  "79b0e97327491737de8507dba3d8e1fad3fae2c9051ecbe47d9e48259aa82c7e",
      #|  "eb8f7b026b64a1033cedea736afd95c11ca2832989beec79b6f581ba1e6a7ee5",
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
