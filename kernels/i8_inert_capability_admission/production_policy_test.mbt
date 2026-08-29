///|
fn production_i8_metadata() -> @spec.LlamaModelMetadata {
  try! @spec.LlamaModelMetadata::from_verified_content(
    @spec.LlamaModelSpec::new(
      vocabulary_size=8,
      hidden_size=4,
      intermediate_size=6,
      layer_count=1,
      attention_head_count=2,
      kv_head_count=1,
      context_length=8,
      rope_theta=10000.0,
      weight_dtype=BFloat16,
      kv_dtype=BFloat16,
    ),
    @spec.ContentDigest::from_sha256(
      "f123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    ),
  )
}

///|
fn rebuild_production_numeric_plan(
  base : @plan.ModelPlan,
  selected : ArrayView[@plan.OperationId],
  fp8? : Bool = false,
) -> @plan.ModelPlan {
  let numeric_schema = if fp8 {
    @plan_test_fixture.synthetic_fp8_w8a8_numeric_schema(
      base.operations(),
      selected,
    )
  } else {
    @plan_test_fixture.synthetic_symmetric_i8_weight_only_numeric_schema_v1(
      base.operations(),
      selected,
    )
  }
  try! @plan.ModelPlan::new_with_kv_execution(
    content_digest=base.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=base.shape_constraints(),
    kv_cache_geometry=base.kv_cache_geometry(),
    workspace_bounds=base.workspace_bounds(),
    operations=base.operations(),
    final_output=base.final_output(),
    kv_execution=base.kv_execution(),
    numeric_schema~,
  )
}

///|
fn production_i8_eligible(
  plan : @plan.ModelPlan,
  omit? : @plan.OperationId? = None,
) -> Array[@plan.OperationId] {
  let selected : Array[@plan.OperationId] = []
  for operation in plan.operations() {
    let eligible = operation.kind()
      is (QkvProjection | OutputProjection | GatedMlp | LanguageModelHead)
    if eligible && Some(operation.id()) != omit {
      selected.push(operation.id())
    }
  }
  selected
}

///|
test "production policy preflight admits only exact full Llama I8 coverage" {
  let base = try! @llama.build_paged(production_i8_metadata())
  let plan = rebuild_production_numeric_plan(base, production_i8_eligible(base))
  preflight_production_llama_symmetric_i8_weight_only_v1(plan)
}

///|
test "production policy rejects partial reviewed-operation coverage" {
  let base = try! @llama.build_paged(production_i8_metadata())
  let omitted = try! @plan.OperationId::new(5)
  let plan = rebuild_production_numeric_plan(
    base,
    production_i8_eligible(base, omit=Some(omitted)),
  )
  try preflight_production_llama_symmetric_i8_weight_only_v1(plan) catch {
    I8InertKernelCapabilityRejected(ProductionExecution) => ()
    _ => fail("unexpected partial production-I8 rejection")
  } noraise {
    _ => fail("partial production-I8 coverage was admitted")
  }
}

///|
test "production policy rejects mixed FP8 execution before catalog work" {
  let base = try! @llama.build_paged(production_i8_metadata())
  let fp8 = rebuild_production_numeric_plan(
    base,
    production_i8_eligible(base),
    fp8=true,
  )
  try preflight_production_llama_symmetric_i8_weight_only_v1(fp8) catch {
    I8InertKernelCapabilityRejected(MixedFp8Execution) => ()
    _ => fail("unexpected mixed-FP8 production rejection")
  } noraise {
    _ => fail("FP8 execution entered production-I8 policy")
  }
}

///|
test "production policy rejects an arbitrary paged graph" {
  let plan = i8_admission_plan(1)
  try preflight_production_llama_symmetric_i8_weight_only_v1(plan) catch {
    I8InertKernelCapabilityRejected(ProductionGraph) => ()
    _ => fail("unexpected non-Llama graph rejection")
  } noraise {
    _ => fail("arbitrary paged graph entered production Llama policy")
  }
}
