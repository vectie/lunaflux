///|
fn parameter_row_ordinal(
  requirements : @joint_diffusion_kernel_requirements.JointDiffusionKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      FinalAdaLnParameterRow(_, _) => return ordinal
      _ => ()
    }
  }
  abort("parameter row missing")
}

///|
test "final AdaLN parameter row authenticates exact one-row ABI" {
  let requirements = diffusion_aot_requirements()
  let candidate = lower_final_adaln_parameter_row_cuda_aot_candidate(
    requirements,
    requirement_ordinal=parameter_row_ordinal(requirements),
    target=diffusion_aot_target(),
    compiler=diffusion_aot_compiler(),
    limits=diffusion_aot_limits(),
  )
  assert_eq(candidate.operation(), FinalAdaLnParameterRow(2, 4))
  assert_eq(candidate.input_width(), 2)
  assert_eq(candidate.hidden_size(), 4)
  assert_eq(candidate.weight_element_count(), 16L)
  assert_eq(candidate.dimensions().grid_x(), 8)
  assert_eq(candidate.dimensions().block_x(), 1)
  assert_eq(candidate.operands().map(op => (op.role(), op.byte_count())), [
    (FinalAdaLnConditioningInput, 8L),
    (FinalAdaLnWeight, 32L),
    (FinalAdaLnBias, 16L),
    (FinalAdaLnShiftOutput, 8L),
    (FinalAdaLnScaleOutput, 8L),
  ])
  assert_false(candidate.manifest_bindable())
}

///|
test "parameter row rejects ordinal compiler target and overflow ceilings" {
  let requirements = diffusion_aot_requirements()
  let ordinal = parameter_row_ordinal(requirements)
  try
    lower_final_adaln_parameter_row_cuda_aot_candidate(
      requirements,
      requirement_ordinal=0,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    UnsupportedRequirement(0) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("wrong ordinal accepted")
  }
  try
    lower_final_adaln_parameter_row_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(bf16=false),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target error")
  } noraise {
    _ => fail("wrong target accepted")
  }
  try
    lower_final_adaln_parameter_row_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(reassociate=true),
      limits=diffusion_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("wrong compiler accepted")
  }
  try
    lower_final_adaln_parameter_row_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      target=diffusion_aot_target(),
      compiler=diffusion_aot_compiler(),
      limits=diffusion_aot_limits(maximum_elements=15L),
    )
  catch {
    Invalid(Limit) => ()
    _ => fail("wrong limit error")
  } noraise {
    _ => fail("overflow ceiling accepted")
  }
}
