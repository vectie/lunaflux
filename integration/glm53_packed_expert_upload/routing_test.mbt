///|
test "GLM routing keeps checkpoint BF16 weight and F32 score semantics" {
  let plan = @glm53_packed_expert_upload.routing_compute_plan(
    maximum_rows=2,
    maximum_workspace_bytes=7040L,
    accuracy=@precision_ir.Accuracy::new(absolute=0.02, relative=0.02),
  )
  assert_eq(plan.geometry(), (2, 4096, 288, 8))
  assert_eq(plan.grouping(), (1, 2, 1))
  assert_eq(
    plan.weight().representation().format(),
    @precision_format.Format::Bf16,
  )
  assert_eq(plan.weight().representation().payload_bytes(), 2359296L)
  assert_eq(plan.workspace_bytes(), 7040L)
  assert_eq(plan.score(), @precision_ir.MoeRoutingScore::Sigmoid)
  assert_eq(plan.scaling(), 2.5)
  assert_true(plan.normalize())
}
