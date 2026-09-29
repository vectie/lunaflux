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

///|
fn alternative(
  schedule : @luna_fusion_strategy.FusionSchedule,
  ns : Int64,
  local_bytes? : Int64 = 0L,
) -> @luna_fusion_strategy.FusionAlternative raise {
  @luna_fusion_strategy.FusionAlternative::new(
    schedule~,
    cost=@luna_fusion_strategy.FusionSpanCost::measured(
      median_nanoseconds=ns,
      samples=3,
    ),
    workspace_bytes=0L,
    local_bytes~,
  )
}

///|
test "ingress whole-region comparison shares global deterministic partition selection" {
  for winner in [@luna_fusion_strategy.Unfused, ProducerSeparated, FullyFused] {
    let choices = [@luna_fusion_strategy.Unfused, ProducerSeparated, FullyFused].map(schedule => {
        alternative(schedule, if schedule == winner { 10L } else { 20L })
      },
    )
    for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
      let cut = @luna_fusion_plan.select_attention_ingress_cut(
        order.map(index => choices[index]),
        qkv_operation=2,
        maximum_workspace_bytes=0L,
        maximum_local_bytes=0L,
      )
      assert_eq(cut.schedule(), winner)
      assert_eq(
        cut.partition().regions().length(),
        match winner {
          Unfused => 4
          ProducerSeparated => 2
          FullyFused => 1
        },
      )
    }
  }
  assert_eq(
    @luna_fusion_plan.select_attention_ingress_cut(
      [
        alternative(FullyFused, 1L, local_bytes=64L),
        alternative(ProducerSeparated, 10L),
      ],
      qkv_operation=2,
      maximum_workspace_bytes=0L,
      maximum_local_bytes=32L,
    ).schedule(),
    @luna_fusion_strategy.ProducerSeparated,
  )
  assert_eq(
    @luna_fusion_plan.select_attention_ingress_cut(
      [alternative(FullyFused, 10L), alternative(Unfused, 10L)],
      qkv_operation=2,
      maximum_workspace_bytes=0L,
      maximum_local_bytes=0L,
    ).schedule(),
    @luna_fusion_strategy.Unfused,
  )
}

///|
test "ingress comparisons reject duplicates mixed cost bases and resource exhaustion" {
  let full = alternative(FullyFused, 1L, local_bytes=64L)
  let estimated = @luna_fusion_strategy.FusionAlternative::new(
    schedule=Unfused,
    cost=@luna_fusion_strategy.FusionSpanCost::estimated(
      compute_nanoseconds=1L,
      traffic_nanoseconds=0L,
      synchronization_nanoseconds=0L,
      launch_nanoseconds=0L,
    ),
    workspace_bytes=0L,
    local_bytes=0L,
  )
  for alternatives in [[full, full], [full, estimated], [full]] {
    try
      @luna_fusion_plan.select_attention_ingress_cut(
        alternatives,
        qkv_operation=2,
        maximum_workspace_bytes=0L,
        maximum_local_bytes=32L,
      )
    catch {
      @luna_fusion_plan.InvalidFusionPlan(InvalidSelection | NoFeasibleCut) =>
        ()
      _ => fail("wrong comparison error")
    } noraise {
      _ => fail("invalid comparison accepted")
    }
  }
}
