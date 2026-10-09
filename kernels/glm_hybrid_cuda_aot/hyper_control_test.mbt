///|
fn hyper_control_ordinals(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int) {
  let mut function = -1
  let mut sinkhorn = -1
  for ordinal, requirement in requirements.requirements() {
    match requirement.operation() {
      HyperConnectionFunction(_, _, _, _) => function = ordinal
      HyperConnectionSinkhorn(_, _, _) => sinkhorn = ordinal
      _ => ()
    }
  }
  if function < 0 || sinkhorn < 0 {
    abort("missing hyper control requirements")
  }
  (function, sinkhorn)
}

///|
test "Flash hyper function and Sinkhorn candidates expose staged exact ABIs" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_sparse=true,
    routed_expert_count=288,
    hyper_connection=true,
  )
  let (function_ordinal, sinkhorn_ordinal) = hyper_control_ordinals(
    requirements,
  )
  let function = lower_hyper_connection_function_cuda_aot_candidate(
    requirements,
    requirement_ordinal=function_ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  let sinkhorn = lower_hyper_connection_sinkhorn_cuda_aot_candidate(
    requirements,
    requirement_ordinal=sinkhorn_ordinal,
    maximum_tokens=32,
    target=glm_aot_target(),
    compiler=glm_aot_compiler(),
    limits=glm_aot_limits(),
  )
  assert_eq(function.family(), FlashFunctionAndCollapse4x4096)
  assert_eq(sinkhorn.family(), FlashPositiveSinkhorn4x20)
  assert_eq(
    function.operation(),
    HyperConnectionFunction(4, 4096, 0.000001, 0.00001),
  )
  assert_eq(sinkhorn.operation(), HyperConnectionSinkhorn(4, 20, 0.000001))
  assert_eq(function.operands().length(), 8)
  assert_eq(function.operands()[0].role(), StepCountsInput)
  assert_eq(function.operands()[1].role(), HyperResidualStreamsInput)
  assert_eq(function.operands()[1].byte_count(), 1048576L)
  assert_eq(function.operands()[2].role(), HyperFunctionWeightsInput)
  assert_eq(function.operands()[2].byte_count(), 786432L)
  assert_eq(function.operands()[3].role(), HyperBaseInput)
  assert_eq(function.operands()[3].byte_count(), 96L)
  assert_eq(function.operands()[4].role(), HyperScaleInput)
  assert_eq(function.operands()[5].role(), HyperPostControlInput)
  assert_eq(function.operands()[6].role(), HyperInitialCombinationOutput)
  assert_eq(function.operands()[6].byte_count(), 2048L)
  assert_eq(function.operands()[7].role(), HyperCollapsedOutput)
  assert_eq(function.operands()[7].byte_count(), 262144L)
  assert_eq(sinkhorn.operands().length(), 3)
  assert_eq(sinkhorn.operands()[1], function.operands()[6])
  assert_eq(sinkhorn.operands()[2].role(), HyperCombineControlInput)
  assert_eq(function.dimensions().block_x(), 1)
  assert_eq(sinkhorn.dimensions().grid_x(), 32)
  let function_source = @utf8.decode(function.source_bytes())
  let sinkhorn_source = @utf8.decode(sinkhorn.source_bytes())
  assert_true(function_source.contains("const __nv_bfloat16 *function_weights"))
  assert_true(function_source.contains("__float2bfloat16_rn(sum)"))
  assert_false(function_source.contains("iteration = 1"))
  assert_true(sinkhorn_source.contains("iteration = 1; iteration < 20"))
  assert_false(sinkhorn_source.contains("function_weights"))
  assert_false(function.manifest_bindable())
  assert_false(sinkhorn.manifest_bindable())
}

///|
test "hyper control lowerers reject neighboring ordinals shape target and compiler" {
  let requirements = glm_aot_requirements(
    pooled=true,
    official_sparse=true,
    routed_expert_count=288,
    hyper_connection=true,
  )
  let (function_ordinal, sinkhorn_ordinal) = hyper_control_ordinals(
    requirements,
  )
  try
    lower_hyper_connection_function_cuda_aot_candidate(
      requirements,
      requirement_ordinal=sinkhorn_ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    UnsupportedRequirement(_) => ()
    _ => fail("wrong ordinal rejection")
  } noraise {
    _ => fail("Sinkhorn ordinal admitted as function")
  }
  try
    lower_hyper_connection_sinkhorn_cuda_aot_candidate(
      requirements,
      requirement_ordinal=sinkhorn_ordinal,
      maximum_tokens=32,
      target=glm_aot_target(bf16=false),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Target) => ()
    _ => fail("wrong target rejection")
  } noraise {
    _ => fail("non-BF16 target admitted")
  }
  try
    lower_hyper_connection_function_cuda_aot_candidate(
      requirements,
      requirement_ordinal=function_ordinal,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(reassociate=true),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Compiler) => ()
    _ => fail("wrong compiler rejection")
  } noraise {
    _ => fail("reassociated compiler admitted")
  }
  let wrong_shape = glm_aot_requirements(hyper_connection=true)
  let (wrong_function, _) = hyper_control_ordinals(wrong_shape)
  try
    lower_hyper_connection_function_cuda_aot_candidate(
      wrong_shape,
      requirement_ordinal=wrong_function,
      maximum_tokens=32,
      target=glm_aot_target(),
      compiler=glm_aot_compiler(),
      limits=glm_aot_limits(),
    )
  catch {
    Invalid(Shape) => ()
    _ => fail("wrong shape rejection")
  } noraise {
    _ => fail("nonofficial hidden width admitted")
  }
}
