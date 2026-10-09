///|
fn joint_kernel_content(seed : Char) -> @spec.ContentDigest {
  try! @spec.ContentDigest::from_sha256(String::make(64, seed))
}

///|
fn joint_kernel_plan(
  variant? : @minimax_h3_spec.MiniMaxH3Variant = Fl2va,
) -> @joint_diffusion_plan.JointDiffusionPlan {
  try! @minimax_h3.build(
    @minimax_h3_spec.MiniMaxH3ModelMetadata::from_verified_content(
      @minimax_h3_spec.MiniMaxH3TransformerSpec::official(
        variant~,
        checkpoint_format=DiffusersV1,
      ),
      joint_kernel_content('a'),
    ),
  )
}

///|
fn joint_kernel_limits(
  maximum_layers? : Int = 64,
  maximum_requirements? : Int = 33,
) -> JointDiffusionKernelRequirementLimits {
  try! JointDiffusionKernelRequirementLimits::new(
    maximum_transformer_layers=maximum_layers,
    maximum_requirements~,
  )
}

///|
test "requirements are deterministic ordered and source bound" {
  let plan = joint_kernel_plan()
  let first = try! derive_requirements(plan, limits=joint_kernel_limits())
  let repeated = try! derive_requirements(plan, limits=joint_kernel_limits())
  assert_eq(first, repeated)
  assert_eq(first.source_identity().content(), plan.content_digest())
  assert_eq(first.source_identity().plan(), plan.plan_digest())
  assert_eq(first.source_schema_version(), 1)
  assert_eq(first.denoiser_shape(), plan.denoiser())
  assert_eq(first.shape_contract(), plan.shapes())
  assert_eq(first.conditioning_limits(), plan.conditioning())
  assert_eq(first.requirements().length(), 33)
  assert_eq(first.capabilities().length(), 30)
}

///|
test "decoder generic capability projection rejects the first diffusion op" {
  let requirements = try! derive_requirements(
    joint_kernel_plan(),
    limits=joint_kernel_limits(),
  )
  try requirements.require_current_generic_decoder_capabilities() catch {
    UnsupportedGenericDecoderKernelOperation(0, TokenTextConditioning(5120)) =>
      ()
    _ => fail("generic decoder projection returned the wrong rejection")
  } noraise {
    _ => fail("diffusion requirements were weakened into decoder kernels")
  }
}

///|
test "derivation limits fail before partial requirements escape" {
  for hostile in [(0, 1), (4097, 1), (1, 0), (1, 65)] {
    let (layers, requirements) = hostile
    try
      JointDiffusionKernelRequirementLimits::new(
        maximum_transformer_layers=layers,
        maximum_requirements=requirements,
      )
    catch {
      InvalidLimit => ()
      _ => fail("invalid limit returned the wrong error")
    } noraise {
      _ => fail("invalid requirement limit was admitted")
    }
  }
  try
    derive_requirements(
      joint_kernel_plan(),
      limits=joint_kernel_limits(maximum_layers=51),
    )
  catch {
    PlanExceedsLimits => ()
    _ => fail("transformer limit returned the wrong error")
  } noraise {
    _ => fail("transformer layer ceiling was bypassed")
  }
  try
    derive_requirements(
      joint_kernel_plan(),
      limits=joint_kernel_limits(maximum_requirements=9),
    )
  catch {
    PlanExceedsLimits => ()
    _ => fail("requirement limit returned the wrong error")
  } noraise {
    _ => fail("requirement count ceiling was bypassed")
  }
}
