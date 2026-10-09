///|
fn kda_output_projection_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is KdaOutputProjection(_, _) {
      return ordinal
    }
  }
  abort("missing KDA output projection")
}

///|
test "KDA output projection binds exact BF16 row-major no-bias geometry" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let candidate = lower_kda_output_projection_cuda_aot_candidate(
    requirements,
    requirement_ordinal=kda_output_projection_ordinal(requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(candidate.operation(), KdaOutputProjection(8192, 4096))
  assert_eq(candidate.family(), KdaOutputProjectionBf16)
  assert_eq(candidate.operands()[1].role(), KdaOutputProjectionInput)
  assert_eq(candidate.operands()[1].byte_count(), 524288L)
  assert_eq(candidate.operands()[2].role(), KdaOutputProjectionWeights)
  assert_eq(candidate.operands()[2].byte_count(), 67108864L)
  assert_eq(candidate.operands()[3].role(), KdaOutputProjectionOutput)
  assert_eq(candidate.operands()[3].byte_count(), 262144L)
  assert_eq(candidate.dimensions().grid_x(), 512)
  assert_false(candidate.manifest_bindable())
}

///|
test "KDA output projection rejects adjacent ordinal compiler and target" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=64,
  )
  let ordinal = kda_output_projection_ordinal(requirements)
  try
    lower_kda_output_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal - 1,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal error")
  } noraise {
    _ => fail("gated norm admitted as output projection")
  }
  try
    lower_kda_output_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler error")
  } noraise {
    _ => fail("reassociating compiler admitted")
  }
  try
    lower_kda_output_projection_cuda_aot_candidate(
      requirements,
      requirement_ordinal=ordinal,
      maximum_tokens=32,
      target=glm_aot_target(bf16=false),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target error")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  let hostile_shape = glm_aot_requirements(
    pooled=true,
    official_index=true,
    official_sparse=true,
    kda_head_count=8,
  )
  try
    lower_kda_output_projection_cuda_aot_candidate(
      hostile_shape,
      requirement_ordinal=kda_output_projection_ordinal(hostile_shape),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape error")
  } noraise {
    _ => fail("non-official KDA output width admitted")
  }
}
