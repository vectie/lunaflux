///|
test "GLM packed execution plan maps device expert partition without BF16 expansion" {
  let plan = @glm53_packed_expert_upload.expert_compute_plan(
    layer_index=10,
    expert_indices=[287, 0],
    maximum_rows=1,
    accuracy=@precision_ir.Accuracy::new(absolute=0.02, relative=0.02),
    memory_ceiling=40000000L,
  )
  assert_eq(plan.geometry(), (1, 4096, 2048, 288, 8))
  assert_eq(plan.expert_map()[287], 0)
  assert_eq(plan.expert_map()[0], 1)
  assert_eq(plan.expert_map()[1], -1)
  assert_eq(plan.bank_bytes(), 28311648L)
  assert_eq(plan.workspace_bytes(), (32768L, 65536L))
  assert_eq(plan.clamp(), Some(10.0))
}
