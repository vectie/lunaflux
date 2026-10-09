///|
fn moe_router_ordinal(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> Int {
  for ordinal, requirement in requirements.requirements() {
    if requirement.operation() is MoeRouterProjection(_, _, _) {
      return ordinal
    }
  }
  abort("missing MoE router projection")
}

///|
test "MoE router candidates bind distinct official Full and Flash geometry" {
  let full_requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=false,
    routed_expert_count=256,
  )
  let flash_requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
  )
  let full = lower_moe_router_projection_cuda_aot_candidate(
    full_requirements,
    requirement_ordinal=moe_router_ordinal(full_requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  let flash = lower_moe_router_projection_cuda_aot_candidate(
    flash_requirements,
    requirement_ordinal=moe_router_ordinal(flash_requirements),
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(full.family(), FullFloat32Router256)
  assert_eq(flash.family(), FlashFloat32Router288)
  assert_eq(full.operation(), MoeRouterProjection(6144, 256, true))
  assert_eq(flash.operation(), MoeRouterProjection(4096, 288, true))
  assert_eq(full.operands()[1].role(), MoeRouterFloat32Input)
  assert_eq(full.operands()[1].byte_count(), 786432L)
  assert_eq(full.operands()[2].role(), MoeRouterBf16Weights)
  assert_eq(full.operands()[2].byte_count(), 3145728L)
  assert_eq(full.operands()[3].role(), MoeRouterFloat32LogitsOutput)
  assert_eq(full.operands()[3].byte_count(), 32768L)
  assert_eq(full.dimensions().grid_x(), 32)
  assert_eq(flash.operands()[2].byte_count(), 2359296L)
  assert_eq(flash.operands()[3].byte_count(), 36864L)
  assert_eq(flash.dimensions().grid_x(), 36)
  let source = @utf8.decode(flash.source_bytes())
  assert_true(source.contains("const float *input"))
  assert_true(source.contains("const __nv_bfloat16 *weight"))
  assert_true(source.contains("float *output"))
  assert_false(source.contains("sigmoid"))
  assert_false(source.contains("bias"))
  assert_false(full.manifest_bindable())
  assert_false(flash.manifest_bindable())
}

///|
test "MoE router rejects ordinal dtype shape compiler and target mismatches" {
  let requirements = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
  )
  let ordinal = moe_router_ordinal(requirements)
  try
    lower_moe_router_projection_cuda_aot_candidate(
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
    _ => fail("neighboring requirement admitted")
  }
  let non_f32 = glm_aot_requirements(
    official_sparse=true,
    pooled=true,
    routed_expert_count=288,
    moe_router_float32=false,
  )
  try
    lower_moe_router_projection_cuda_aot_candidate(
      non_f32,
      requirement_ordinal=moe_router_ordinal(non_f32),
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong dtype error")
  } noraise {
    _ => fail("non-F32 router admitted")
  }
  for hostile_experts in [287, 289] {
    let hostile = glm_aot_requirements(
      official_sparse=true,
      pooled=true,
      routed_expert_count=hostile_experts,
    )
    try
      lower_moe_router_projection_cuda_aot_candidate(
        hostile,
        requirement_ordinal=moe_router_ordinal(hostile),
        maximum_tokens=32,
        target=glm_aot_target(),
        compiler=glm_aot_compiler(),
        limits=glm_aot_limits(),
      )
    catch {
      Invalid(Shape) => ()
      _ => fail("wrong shape error")
    } noraise {
      _ => fail("neighboring expert geometry admitted")
    }
  }
  try
    lower_moe_router_projection_cuda_aot_candidate(
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
    lower_moe_router_projection_cuda_aot_candidate(
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
    _ => fail("target without BF16 admitted")
  }
}
