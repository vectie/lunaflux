///|
fn timestep_mlp_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      TimestepEmbeddingMlpRows(_, _, _) => return ordinal
      _ => ()
    }
  }
  abort("timestep embedding MLP missing")
}

///|
fn distinct_timestep_adaln_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      DistinctTimestepAdaLnParameterRows(_, _) => return ordinal
      _ => ()
    }
  }
  abort("distinct timestep AdaLN parameter rows missing")
}

///|
fn assert_timestep_mlp_candidate(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
  values : @joint_diffusion_io_plan.JointDistinctTimestepValues,
  frequency : @joint_diffusion_cuda_aot.JointDiffusionTimestepFrequencyCudaCandidate,
) -> Unit raise {
  let candidate = @joint_diffusion_cuda_aot.lower_timestep_mlp_cuda_aot_candidate(
    requirements,
    values~,
    requirement_ordinal=timestep_mlp_ordinal(requirements),
    target=integration_target(),
    compiler=integration_compiler(),
    limits=integration_limits(),
  )
  assert_eq(candidate.numerical_contract(), OrderedFloat32DenseSiluDenseV1)
  assert_eq(candidate.operation(), TimestepEmbeddingMlpRows(256, 5376, 2688))
  assert_eq(candidate.capability(), TimestepEmbeddingMlpRowsAot)
  assert_eq(candidate.values_digest(), values.digest())
  assert_eq(candidate.row_count(), 3)
  assert_eq(candidate.input_width(), 256)
  assert_eq(candidate.hidden_width(), 5376)
  assert_eq(candidate.output_width(), 2688)
  assert_eq(candidate.dimensions().grid_x(), 32)
  assert_eq(candidate.operands().length(), 6)
  assert_eq(candidate.operands()[0].byte_count(), 3072L)
  assert_eq(candidate.operands()[1].byte_count(), 5505024L)
  assert_eq(candidate.operands()[2].byte_count(), 21504L)
  assert_eq(candidate.operands()[3].byte_count(), 57802752L)
  assert_eq(candidate.operands()[4].byte_count(), 10752L)
  assert_eq(candidate.operands()[5].byte_count(), 32256L)
  assert_false(candidate.manifest_bindable())
  let adaln = @joint_diffusion_cuda_aot.lower_distinct_timestep_adaln_cuda_aot_candidate(
    requirements,
    values~,
    requirement_ordinal=distinct_timestep_adaln_ordinal(requirements),
    target=integration_target(),
    compiler=integration_compiler(),
    limits=integration_limits(),
  )
  assert_eq(adaln.operation(), DistinctTimestepAdaLnParameterRows(2688, 5376))
  assert_eq(adaln.values_digest(), values.digest())
  assert_eq(adaln.row_count(), 3)
  assert_eq(adaln.input_width(), 2688)
  assert_eq(adaln.hidden_size(), 5376)
  assert_eq(adaln.weight_element_count(), 28901376L)
  assert_eq(adaln.dimensions().grid_x(), 32256)
  assert_eq(adaln.operands()[0].byte_count(), 32256L)
  assert_eq(adaln.operands()[1].byte_count(), 57802752L)
  assert_eq(adaln.operands()[2].byte_count(), 21504L)
  assert_eq(adaln.operands()[3].byte_count(), 32256L)
  assert_eq(adaln.operands()[4].byte_count(), 32256L)
  assert_false(adaln.manifest_bindable())
  let composition = @joint_diffusion_cuda_aot.compose_timestep_conditioning_candidates(
    frequency, candidate, adaln,
  )
  assert_eq(composition.source_identity(), requirements.source_identity())
  assert_eq(composition.requirements_digest(), requirements.digest())
  assert_eq(composition.values_digest(), values.digest())
  assert_eq(composition.row_count(), 3)
  assert_eq(composition.frequency_width(), 256)
  assert_eq(composition.mlp_hidden_width(), 5376)
  assert_eq(composition.conditioning_width(), 2688)
  assert_eq(composition.adaln_hidden_size(), 5376)
  assert_false(composition.manifest_bindable())
  try
    @joint_diffusion_cuda_aot.lower_timestep_mlp_cuda_aot_candidate(
      requirements,
      values~,
      requirement_ordinal=0,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected timestep MLP ordinal rejection")
  } noraise {
    _ => fail("conditioning ordinal admitted as timestep MLP")
  }
}

///|
test "timestep MLP rejects foreign values unsafe compiler and undersized limits" {
  let fl = official_requirements(Fl2va)
  let reference = official_requirements(Ref2va)
  let (_, _, values) = packed_selection_fixture(Fl2va)
  let (_, _, reference_values) = packed_selection_fixture(Ref2va)
  try
    @joint_diffusion_cuda_aot.lower_timestep_mlp_cuda_aot_candidate(
      reference,
      values~,
      requirement_ordinal=timestep_mlp_ordinal(reference),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected foreign timestep-value rejection")
  } noraise {
    _ => fail("foreign timestep values admitted")
  }
  try
    @joint_diffusion_cuda_aot.lower_distinct_timestep_adaln_cuda_aot_candidate(
      reference,
      values~,
      requirement_ordinal=distinct_timestep_adaln_ordinal(reference),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected foreign AdaLN-value rejection")
  } noraise {
    _ => fail("foreign values admitted for AdaLN tables")
  }
  let fl_frequency = try! @joint_diffusion_cuda_aot.lower_timestep_frequency_cuda_aot_candidate(
    fl,
    values~,
    requirement_ordinal=timestep_frequency_ordinal(fl),
    target=integration_target(),
    compiler=integration_compiler(),
    limits=integration_limits(),
  )
  let reference_mlp = try! @joint_diffusion_cuda_aot.lower_timestep_mlp_cuda_aot_candidate(
    reference,
    values=reference_values,
    requirement_ordinal=timestep_mlp_ordinal(reference),
    target=integration_target(),
    compiler=integration_compiler(),
    limits=integration_limits(),
  )
  let reference_adaln = try! @joint_diffusion_cuda_aot.lower_distinct_timestep_adaln_cuda_aot_candidate(
    reference,
    values=reference_values,
    requirement_ordinal=distinct_timestep_adaln_ordinal(reference),
    target=integration_target(),
    compiler=integration_compiler(),
    limits=integration_limits(),
  )
  try
    @joint_diffusion_cuda_aot.compose_timestep_conditioning_candidates(
      fl_frequency, reference_mlp, reference_adaln,
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected mixed-composition rejection")
  } noraise {
    _ => fail("mixed request identities composed")
  }
  let unsafe_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_timestep_mlp_cuda_aot_candidate(
      fl,
      values~,
      requirement_ordinal=timestep_mlp_ordinal(fl),
      target=integration_target(),
      compiler=unsafe_compiler,
      limits=integration_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected timestep MLP compiler rejection")
  } noraise {
    _ => fail("unsafe timestep MLP compiler admitted")
  }
  let tiny_limits = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=1L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_timestep_mlp_cuda_aot_candidate(
      fl,
      values~,
      requirement_ordinal=timestep_mlp_ordinal(fl),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=tiny_limits,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected timestep MLP size rejection")
  } noraise {
    _ => fail("oversized timestep MLP admitted")
  }
  try
    @joint_diffusion_cuda_aot.lower_distinct_timestep_adaln_cuda_aot_candidate(
      fl,
      values~,
      requirement_ordinal=distinct_timestep_adaln_ordinal(fl),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=tiny_limits,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected AdaLN table size rejection")
  } noraise {
    _ => fail("oversized AdaLN parameter tables admitted")
  }
  let non_bf16_target = try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=false,
    supports_cublas_lt=true,
  )
  try
    @joint_diffusion_cuda_aot.lower_distinct_timestep_adaln_cuda_aot_candidate(
      fl,
      values~,
      requirement_ordinal=distinct_timestep_adaln_ordinal(fl),
      target=non_bf16_target,
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected AdaLN target rejection")
  } noraise {
    _ => fail("non-BF16 AdaLN target admitted")
  }
}
