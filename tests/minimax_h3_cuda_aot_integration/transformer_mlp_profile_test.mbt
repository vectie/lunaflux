///|
fn transformer_mlp_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is JointTransformerMlp(_, _, _, _) {
      return ordinal
    }
  }
  abort("transformer MLP requirement missing")
}

///|
test "official MiniMax profiles lower exact packed and refiner SwiGLU geometry" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  for variant in variants {
    let requirements = official_requirements(variant)
    let (io, timestep, _, _) = packed_selection_fixture_with_io(variant)
    let candidate = @joint_diffusion_cuda_aot.lower_transformer_mlp_cuda_aot_candidate(
      requirements,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_mlp_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      candidate.numerical_contract(),
      OrderedF32AccumulationBf16SwiGluDownBf16V1,
    )
    assert_eq(candidate.operation(), JointTransformerMlp(50, 2, 5376, 14336))
    assert_eq(candidate.capability(), JointTransformerMlpAot)
    assert_eq(candidate.request_digest(), io.request_digest())
    assert_eq(candidate.layer_count(), 50)
    assert_eq(candidate.refiner_layer_count(), 2)
    assert_eq(candidate.packed_row_count(), 37847)
    assert_eq(candidate.refiner_row_count(), 137)
    assert_eq(candidate.hidden_size(), 5376)
    assert_eq(candidate.intermediate_size(), 14336)
    assert_eq(candidate.packed_weight_element_count(), 154140672L)
    assert_eq(candidate.down_weight_element_count(), 77070336L)
    assert_eq(candidate.packed_dimensions().grid_x(), 37847 * 21)
    assert_eq(candidate.refiner_dimensions().grid_x(), 2877)
    assert_eq(candidate.packed_operands()[0].byte_count(), 37847L * 5376L * 2L)
    assert_eq(candidate.packed_operands()[1].byte_count(), 308281344L)
    assert_eq(candidate.packed_operands()[2].byte_count(), 154140672L)
    assert_eq(candidate.packed_operands()[3].byte_count(), 37847L * 5376L * 2L)
    assert_eq(candidate.refiner_operands()[0].byte_count(), 1473024L)
    assert_eq(candidate.refiner_operands()[3].byte_count(), 1473024L)
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "transformer MLP rejects foreign request unsafe compiler limit and ordinal" {
  let fl = official_requirements(Fl2va)
  let reference = official_requirements(Ref2va)
  let (fl_io, fl_timestep, _, _) = packed_selection_fixture_with_io(Fl2va)
  try
    @joint_diffusion_cuda_aot.lower_transformer_mlp_cuda_aot_candidate(
      reference,
      request=fl_io,
      packed_timestep=fl_timestep,
      requirement_ordinal=transformer_mlp_ordinal(reference),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected foreign MLP request rejection")
  } noraise {
    _ => fail("foreign MLP request admitted")
  }
  let hostile_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=true,
    max_registers=128,
  )
  for hostile in [hostile_compiler] {
    try
      @joint_diffusion_cuda_aot.lower_transformer_mlp_cuda_aot_candidate(
        fl,
        request=fl_io,
        packed_timestep=fl_timestep,
        requirement_ordinal=transformer_mlp_ordinal(fl),
        target=integration_target(),
        compiler=hostile,
        limits=integration_limits(),
      )
    catch {
      Invalid(Compiler) => ()
      _ => fail("unexpected MLP compiler rejection")
    } noraise {
      _ => fail("unsafe MLP compiler admitted")
    }
  }
  let tiny = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=154140671L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_transformer_mlp_cuda_aot_candidate(
      fl,
      request=fl_io,
      packed_timestep=fl_timestep,
      requirement_ordinal=transformer_mlp_ordinal(fl),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=tiny,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected MLP limit rejection")
  } noraise {
    _ => fail("oversized MLP admitted")
  }
  try
    @joint_diffusion_cuda_aot.lower_transformer_mlp_cuda_aot_candidate(
      fl,
      request=fl_io,
      packed_timestep=fl_timestep,
      requirement_ordinal=0,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("unexpected MLP ordinal rejection")
  } noraise {
    _ => fail("conditioning requirement lowered as MLP")
  }
}
