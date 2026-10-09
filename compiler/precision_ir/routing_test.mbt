///|
test "routing precision owns exact compact memory and portable score policy" {
  for format in [@format.Format::Bf16, F32] {
    let representation = @format.Representation::new(
      format~,
      rows=8,
      columns=32,
    )
    let weight = @precision_ir.plan_operand(
      role=Weight,
      representation~,
      placement=NativeOperand,
      supported_placements=[NativeOperand],
      accuracy=@precision_ir.Accuracy::new(absolute=0.01, relative=0.01),
      memory_ceiling=representation.payload_bytes(),
    )
    let plan = @precision_ir.MoeRoutingPrecision::new(
      weight~,
      maximum_rows=3,
      group_count=2,
      group_score_topk=2,
      selected_group_count=1,
      selected_experts=2,
      score=SqrtSoftplus,
      normalize=false,
      scaling=1.5,
      maximum_workspace_bytes=336L,
    )
    assert_eq(plan.geometry(), (3, 32, 8, 2))
    assert_eq(plan.grouping(), (2, 2, 1))
    assert_eq(plan.score(), @precision_ir.MoeRoutingScore::SqrtSoftplus)
    assert_eq(plan.buffer_bytes()[:], [96L, 96L, 96L, 24L, 24L][:])
    assert_eq(plan.workspace_bytes(), 336L)
    try
      @precision_ir.MoeRoutingPrecision::new(
        weight~,
        maximum_rows=3,
        group_count=2,
        group_score_topk=2,
        selected_group_count=1,
        selected_experts=2,
        score=Sigmoid,
        normalize=true,
        scaling=1.0,
        maximum_workspace_bytes=335L,
      )
    catch {
      MemoryBudgetExceeded => ()
      _ => fail("wrong workspace error")
    } noraise {
      _ => fail("routing memory excluded")
    }
  }
}
