///|
fn quantized_activation(
  rows : Int,
  columns : Int,
) -> @precision_ir.QuantizationPlan raise {
  @precision_ir.plan_operand(
    role=Activation,
    representation=@format.Representation::new(
      format=MxFp8E4M3,
      rows~,
      columns~,
      grouping=Block(rows=1, columns=128),
    ),
    placement=QuantizeBeforeConsumer,
    supported_placements=[QuantizeBeforeConsumer],
    accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
    memory_ceiling=1000000L,
  )
  .quantization()
  .with_amax_floor(0.0001)
}

///|
test "expert quantization and routing order are explicit immutable numeric semantics" {
  let qx = quantized_activation(3, 64)
  let qa = quantized_activation(6, 32)
  let plan = @precision_ir.ExpertMlpPrecision::new(
    gate=expert_operand(32, 64),
    up=expert_operand(32, 64),
    down=expert_operand(64, 32),
    expert_count=7,
    local_experts=[5, 1],
    selected_experts=2,
    maximum_rows=3,
    activation_rounding=F32Product,
    routing_placement=BeforeDownInputRound,
    activation_quantization=(qx, qa),
    memory_ceiling=100000L,
  )
  assert_eq(plan.activation_rounding(), F32Product)
  assert_eq(plan.routing_placement(), BeforeDownInputRound)
  assert_eq(plan.quantized_workspace_bytes(), (195L, 198L))
  assert_eq(plan.total_bytes(), 9805L)
  assert_eq(qx.amax_floor(), 0.0001)
  assert_true(qx.canonical().contains("amax-floor:"))
  assert_eq(qx.operand().quantization().amax_floor(), 0.0)
}

///|
test "expert activation precision cannot silently use the wrong row extent" {
  try
    @precision_ir.ExpertMlpPrecision::new(
      gate=expert_operand(32, 64),
      up=expert_operand(32, 64),
      down=expert_operand(64, 32),
      expert_count=7,
      local_experts=[5, 1],
      selected_experts=2,
      maximum_rows=3,
      activation_quantization=(
        quantized_activation(3, 64),
        quantized_activation(3, 32),
      ),
      memory_ceiling=100000L,
    )
  catch {
    @precision_ir.PrecisionPlanError::IllegalPlacement => ()
    _ => fail("wrong numeric shape error")
  } noraise {
    _ => fail("quantized products lost the expert dimension")
  }
}
