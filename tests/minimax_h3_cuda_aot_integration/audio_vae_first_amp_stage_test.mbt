///|
fn audio_first_amp_stage_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      AudioVaeDecoderFirstAmpStage(_, _, _, _, _, _) => return ordinal
      _ => ()
    }
  }
  abort("missing complete first AMP stage")
}

///|
test "official profiles bind exact 97-launch first AMP stage" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  let recipes : Array[String] = []
  let sources : Array[String] = []
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
    let parameters = try! @minimax_h3_vae_reference.bind_first_amp_parameter_layout()
    let workspace = try! @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
      element_count=1059840L,
      intermediate_element_count=2119680L,
      normalized_weight_element_count=2883584L,
    )
    let candidate = try! @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_stage_cuda_aot_candidate(
      requirements,
      request~,
      parameter_layout=parameters,
      workspace_layout=workspace,
      requirement_ordinal=audio_first_amp_stage_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      candidate.numerical_contract(),
      OrderedF32AmpParallelResidualAverageV1,
    )
    assert_eq(candidate.stages().length(), 97)
    assert_eq(candidate.operands().length(), 5)
    assert_eq(candidate.operands()[0].byte_count(), 4239360L)
    assert_eq(candidate.operands()[1].byte_count(), 4239360L)
    assert_eq(candidate.operands()[2].byte_count(), 132270336L)
    assert_eq(candidate.operands()[3].byte_count(), workspace.workspace_bytes())
    assert_eq(candidate.operands()[4].byte_count(), 4239360L)
    assert_eq(
      candidate.workspace_layout_digest().as_hex(),
      "fcf8e6679a166fea460a7baae27ebd2828f311d042bd76f468a71cd05a20d268",
    )
    match candidate.stages()[0].identity() {
      Convolution(0, 0, 0, Normalize) => ()
      _ => fail("activation0 was recomputed")
    }
    match candidate.stages()[29].identity() {
      Residual(0, 2) => ()
      _ => fail("block0 launch count changed")
    }
    match candidate.stages()[62].identity() {
      Residual(1, 2) => ()
      _ => fail("block1 launch count changed")
    }
    match candidate.stages()[95].identity() {
      Residual(2, 2) => ()
      _ => fail("block2 launch count changed")
    }
    match candidate.stages()[96].identity() {
      ParallelAverage => ()
      _ => fail("average missing")
    }
    assert_eq(candidate.stages()[0].dimensions().grid_x(), 512)
    assert_eq(candidate.stages()[0].dimensions().block_x(), 1)
    assert_eq(candidate.stages()[1].dimensions().grid_x(), 4140)
    assert_false(candidate.manifest_bindable())
    ignore(
      try! @joint_diffusion_candidate_evidence.from_audio_vae_decoder_first_amp_stage(
        candidate,
      ),
    )
    recipes.push(candidate.recipe_digest().as_hex())
    sources.push(candidate.source_digest().as_hex())
  }
  assert_eq(recipes.length(), 2)
  assert_eq(
    recipes[0],
    "42d32891f4a010f74b1d07aa7bce25e299a7650c2a1d234cae28fc13cf0b73b3",
  )
  assert_eq(
    recipes[1],
    "618125455f602757972ac82bcb85466b202db600d02d0e387399117d9f43aea5",
  )
  assert_eq(
    sources[0],
    "fd62910395586a261e6778edf86a587a37dcdf4ffcf6d07adef8434bae2a0a1f",
  )
  assert_eq(sources[1], sources[0])
}

///|
test "first AMP stage rejects wrong ordinal and FMA compiler" {
  let requirements = official_requirements(Fl2va)
  let request = try! requirements
    .shape_contract()
    .resolve_request(
      height=768,
      width=1344,
      requested_frames=120,
      inference_steps=50,
    )
  let parameters = try! @minimax_h3_vae_reference.bind_first_amp_parameter_layout()
  let workspace = try! @joint_audio_amp_workspace_layout.first_stage_amp_workspace_layout(
    element_count=1059840L,
    intermediate_element_count=2119680L,
    normalized_weight_element_count=2883584L,
  )
  let ordinal = audio_first_amp_stage_ordinal(requirements)
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_stage_cuda_aot_candidate(
      requirements,
      request~,
      parameter_layout=parameters,
      workspace_layout=workspace,
      requirement_ordinal=ordinal - 1,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("wrong ordinal admitted")
  }
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_audio_vae_decoder_first_amp_stage_cuda_aot_candidate(
      requirements,
      request~,
      parameter_layout=parameters,
      workspace_layout=workspace,
      requirement_ordinal=ordinal,
      target=integration_target(),
      compiler=unsafe_compiler,
      limits=integration_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("unsafe compiler admitted")
  }
}
