///|
test "semantic ingress cuts preserve materializations and KV write ownership" {
  let full = @luna_fusion_plan.AttentionIngressCutPlan::new(
    FullyFused,
    qkv_operation=2,
  )
  let partial = @luna_fusion_plan.AttentionIngressCutPlan::new(
    ProducerSeparated,
    qkv_operation=2,
  )
  let none = @luna_fusion_plan.AttentionIngressCutPlan::new(
    Unfused,
    qkv_operation=2,
  )
  assert_eq(full.module_span(), Some((2, 4)))
  assert_eq(partial.module_span(), Some((3, 3)))
  assert_eq(none.module_span(), None)
  assert_eq(full.materialized_values(), (false, false, true))
  assert_eq(partial.materialized_values(), (true, false, true))
  assert_eq(none.materialized_values(), (true, true, true))
  assert_true(full.permits_readonly_attention())
  assert_true(partial.permits_readonly_attention())
  assert_false(none.permits_readonly_attention())
  for origin in [-1, 2147483645, 2147483647] {
    try
      @luna_fusion_plan.AttentionIngressCutPlan::new(
        FullyFused,
        qkv_operation=origin,
      )
    catch {
      @luna_fusion_plan.InvalidFusionPlan(OperationOrder) => ()
      _ => fail("wrong origin error")
    } noraise {
      _ => fail("invalid span origin accepted")
    }
  }
}
