///|
fn transformer_attention_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is JointTransformerAttention(_, _, _, _, _) {
      return ordinal
    }
  }
  abort("transformer attention requirement missing")
}

///|
test "official MiniMax profiles lower exact eight-stage attention geometry" {
  let variants : Array[@minimax_h3_spec.MiniMaxH3Variant] = [Fl2va, Ref2va]
  for variant in variants {
    let requirements = official_requirements(variant)
    let (io, timestep, _, _) = packed_selection_fixture_with_io(variant)
    let candidate = @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
      requirements,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_attention_ordinal(requirements),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
    assert_eq(
      candidate.operation(),
      JointTransformerAttention(50, 2, 56, 128, 5376),
    )
    assert_eq(candidate.capability(), JointTransformerAttentionAot)
    assert_eq(candidate.layer_count(), 50)
    assert_eq(candidate.refiner_layer_count(), 2)
    assert_eq(candidate.packed_row_count(), 37847)
    assert_eq(candidate.refiner_row_count(), 137)
    assert_eq(candidate.head_count(), 56)
    assert_eq(candidate.head_dimension(), 128)
    assert_eq(candidate.inner_size(), 7168)
    assert_eq(candidate.rotary_dimension(), 96)
    assert_eq(candidate.qk_epsilon_numerator(), 1)
    assert_eq(candidate.qk_epsilon_denominator(), 100000)
    assert_eq(candidate.projection_weight_element_count(), 38535168L)
    assert_eq(candidate.norm_weight_element_count(), 128L)
    assert_eq(candidate.stages().length(), 8)
    assert_eq(candidate.stages()[0].kind(), QkvProjection)
    assert_eq(candidate.stages()[0].scope(), PackedRows)
    assert_eq(candidate.stages()[0].dimensions().grid_x(), 37847 * 28)
    assert_eq(
      candidate.stages()[0].operands()[0].byte_count(),
      37847L * 5376L * 2L,
    )
    assert_eq(candidate.stages()[0].operands()[1].byte_count(), 77070336L)
    assert_eq(
      candidate.stages()[0].operands()[4].byte_count(),
      37847L * 7168L * 2L,
    )
    assert_eq(candidate.stages()[1].kind(), QkNormRotary)
    assert_eq(candidate.stages()[1].dimensions().grid_x(), 37847 * 56)
    assert_eq(
      candidate.stages()[1].operands()[4].byte_count(),
      37847L * 96L * 4L,
    )
    assert_eq(candidate.stages()[2].kind(), FullAttention)
    assert_eq(candidate.stages()[3].kind(), OutputProjection)
    assert_eq(candidate.stages()[3].dimensions().grid_x(), 37847 * 21)
    assert_eq(candidate.stages()[4].scope(), RefinerRows)
    assert_eq(candidate.stages()[4].dimensions().grid_x(), 3836)
    assert_eq(candidate.stages()[5].kind(), QkNorm)
    assert_eq(candidate.stages()[5].dimensions().grid_x(), 7672)
    assert_eq(candidate.stages()[6].dimensions().grid_x(), 3836)
    assert_eq(candidate.stages()[7].dimensions().grid_x(), 2877)
    assert_false(candidate.manifest_bindable())
  }
}

///|
test "attention rejects foreign request ordinal compiler and element limit" {
  let fl = official_requirements(Fl2va)
  let reference = official_requirements(Ref2va)
  let (io, timestep, _, _) = packed_selection_fixture_with_io(Fl2va)
  try
    @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
      reference,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_attention_ordinal(reference),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(RequestShape) => ()
    _ => fail("unexpected foreign attention request rejection")
  } noraise {
    _ => fail("foreign attention request admitted")
  }
  try
    @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
      fl,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_attention_ordinal(fl) + 1,
      target=integration_target(),
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("unexpected attention ordinal rejection")
  } noraise {
    _ => fail("non-attention requirement ordinal admitted")
  }
  let non_bf16_target = try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=0,
    supports_bf16=false,
    supports_cublas_lt=true,
  )
  try
    @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
      fl,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_attention_ordinal(fl),
      target=non_bf16_target,
      compiler=integration_compiler(),
      limits=integration_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("unexpected attention target rejection")
  } noraise {
    _ => fail("non-BF16 attention target admitted")
  }
  let hostile_compiler = try! @luna_specialization.LunaCompilerPolicy::new(
    identity=integration_compiler().identity(),
    optimization_level=3,
    fused_multiply_add=true,
    reassociate=false,
    max_registers=128,
  )
  try
    @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
      fl,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_attention_ordinal(fl),
      target=integration_target(),
      compiler=hostile_compiler,
      limits=integration_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("unexpected attention compiler rejection")
  } noraise {
    _ => fail("unsafe attention compiler admitted")
  }
  let tiny = try! @joint_diffusion_cuda_aot.JointDiffusionCudaAotLimits::new(
    maximum_elements=269803519L,
    maximum_source_bytes=65536,
    maximum_recipe_bytes=65536,
  )
  try
    @joint_diffusion_cuda_aot.lower_transformer_attention_cuda_aot_candidate(
      fl,
      request=io,
      packed_timestep=timestep,
      requirement_ordinal=transformer_attention_ordinal(fl),
      target=integration_target(),
      compiler=integration_compiler(),
      limits=tiny,
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("unexpected attention limit rejection")
  } noraise {
    _ => fail("oversized attention admitted")
  }
}
