///|
// The current model builder requests row-end logits. These FP8 reference
// lowerers implement only the separately supported full-token head contract.
// Construct that contract explicitly instead of silently dropping an operand
// in production, or making an identity test depend on an unsupported builder.
fn fp8_full_token_head_fixture(
  source : @plan.ModelPlan,
) -> @plan.ModelPlan raise {
  let operations : Array[@plan.PlanOperation] = []
  for operation in source.operations() {
    if operation.kind() == LanguageModelHead {
      operations.push(
        @plan.PlanOperation::new_with_runtime_inputs(
          id=operation.id(),
          kind=operation.kind(),
          inputs=operation.inputs(),
          outputs=operation.outputs(),
          shape=operation.shape(),
          required_capability=operation.required_capability(),
          workspace_bytes=operation.workspace_bytes(),
          runtime_inputs=[StepCounts],
        ),
      )
    } else {
      operations.push(operation)
    }
  }
  @plan.ModelPlan::new_with_kv_execution(
    content_digest=source.identity().content(),
    limits=@plan.ModelPlanLimits::production(),
    shape_constraints=source.shape_constraints(),
    kv_cache_geometry=source.kv_cache_geometry(),
    workspace_bounds=source.workspace_bounds(),
    operations~,
    final_output=source.final_output(),
    kv_execution=PagedKeyValue,
    numeric_schema=source.numeric_binding().schema(),
  )
}

///|
test "FP8 v1 and v2 full-token head fixtures own complete output storage" {
  let profile = fp8_aot_profile()
  let v1_plan = fp8_full_token_head_fixture(fp8_aot_plan())
  let v2_plan = fp8_full_token_head_fixture(fp8_aot_plan_v2())
  let head = @plan.OperationId::new(11)
  let v1 = fp8_aot_lower(11, model_plan=v1_plan)
  let v2 = fp8_aot_lower_v2(11, model_plan=v2_plan)
  assert_eq(v1.family(), LanguageModelHead)
  assert_eq(v2.family(), LanguageModelHead)
  for plan in [v1_plan, v2_plan] {
    assert_eq(plan.operations()[11].runtime_inputs(), [StepCounts])
    assert_eq(plan.operations()[11].outputs().length(), 1)
  }
  for
    operands in [
      derive_fp8_projection_operands(v1_plan, operation=head, profile),
      derive_fp8_projection_operands_v2(v2_plan, operation=head, profile),
    ] {
    assert_eq(operands.length(), 6)
    assert_eq(operands[0].role(), RuntimeInput(StepCounts))
    assert_true(operands[4].role() is ActivationOutput(_))
    assert_eq(operands[4].byte_count(), 8L * 32L * 2L)
    assert_eq(operands[5].role(), Workspace)
    assert_eq(operands[5].byte_count(), 4L)
  }
  for source in [v1.source_bytes(), v2.source_bytes()] {
    assert_false(@utf8.decode(source).contains("row_offsets"))
  }
}

///|
test "current selected-row FP8 heads reject rather than shift full-token operands" {
  let head = @plan.OperationId::new(11)
  let profile = fp8_aot_profile()
  let limits = Fp8ProjectionAotLimits::new(
    max_source_bytes=65536,
    max_recipe_bytes=65536,
  )
  let v1 = fp8_aot_plan()
  let v2 = fp8_aot_plan_v2()
  for plan in [v1, v2] {
    assert_eq(plan.operations()[11].runtime_inputs(), [
      StepCounts,
      QueryRowOffsets,
    ])
    assert_not_eq(
      plan.identity().plan(),
      fp8_full_token_head_fixture(plan).identity().plan(),
    )
  }
  try derive_fp8_projection_operands(v1, operation=head, profile) catch {
    Mismatch(Operation) => ()
    _ => fail("unexpected v1 selected-row rejection")
  } noraise {
    _ => fail("v1 selected-row head was silently admitted")
  }
  try derive_fp8_projection_operands_v2(v2, operation=head, profile) catch {
    Mismatch(Operation) => ()
    _ => fail("unexpected v2 selected-row rejection")
  } noraise {
    _ => fail("v2 selected-row head was silently admitted")
  }
  let v1_operands = derive_fp8_projection_operands(
    fp8_full_token_head_fixture(v1),
    operation=head,
    profile,
  )
  let v2_operands = derive_fp8_projection_operands_v2(
    fp8_full_token_head_fixture(v2),
    operation=head,
    profile,
  )
  try
    lower_fp8_projection_cuda_aot_candidate(
      v1,
      operation=head,
      profile,
      fp8_aot_target(),
      fp8_aot_compiler(),
      v1_operands,
      limits,
    )
  catch {
    Mismatch(Operation) => ()
    _ => fail("unexpected v1 full-token operand replay rejection")
  } noraise {
    _ => fail("v1 shifted head ABI was emitted")
  }
  try
    lower_fp8_projection_cuda_aot_candidate_v2(
      v2,
      operation=head,
      profile,
      fp8_aot_target(),
      fp8_aot_compiler(),
      v2_operands,
      limits,
    )
  catch {
    Mismatch(Operation) => ()
    _ => fail("unexpected v2 full-token operand replay rejection")
  } noraise {
    _ => fail("v2 shifted head ABI was emitted")
  }
}
