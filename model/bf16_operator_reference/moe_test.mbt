///|
test "router, packed experts, dense shared MLP and gate compose in BF16" {
  let operand = fn(height, width) raise {
    @precision.plan_operand(
      role=Weight,
      representation=@format.Representation::new(
        format=Bf16,
        rows=height,
        columns=width,
      ),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision.Accuracy::new(absolute=0.0, relative=0.0),
      memory_ceiling=100000L,
      tile_elements=32,
    )
  }
  let expert_plan = @precision.ExpertMlpPrecision::new(
    gate=operand(1, 2),
    up=operand(1, 2),
    down=operand(2, 1),
    expert_count=2,
    local_experts=[0, 1],
    selected_experts=2,
    maximum_rows=2,
    combine_rounding=Bf16EachAdd,
    memory_ceiling=100000L,
  )
  let router = @bf16_operator_reference.Router::new(
    @precision.SoftmaxTopKPrecision::new(
      expert_count=2,
      experts_per_token=2,
      maximum_rows=2,
      weight_storage=Float32CarryingBf16,
    ),
  )
  let experts = @bf16_operator_reference.Experts::new(expert_plan)
  let shared = @bf16_operator_reference.DenseMlp::new(
    @precision.DenseSwiGluPrecision::new(
      hidden_width=2,
      intermediate_width=1,
      maximum_rows=2,
    ),
  )
  let indices = FixedArray::make(4, -1)
  let routes = FixedArray::make(4, 0.0F)
  let routed_output = FixedArray::make(4, 0.0F)
  let shared_output = FixedArray::make(4, 0.0F)
  let output = FixedArray::make(4, 0.0F)
  let input = [1.0F, 0.0F, 0.0F, 1.0F]
  router.run_into([0.0F, 0.0F, 0.0F, 0.0F], indices, routes, rows=2)
  assert_eq(routes[:].to_owned(), [0.5F, 0.5F, 0.5F, 0.5F])
  experts.run_into(
    input,
    indices[:],
    routes[:],
    [16.0F, 0.0F, 1.0F, 0.0F, 16.0F, 0.0F, 1.0F, 0.0F],
    [1.0F, 0.0F, 0.0F, 1.0F],
    routed_output,
    rows=2,
  )
  shared.run_into(
    input,
    [16.0F, 0.0F],
    [1.0F, 0.0F],
    [2.0F, -2.0F],
    shared_output,
    rows=2,
  )
  assert_eq(routed_output[:].to_owned(), [8.0F, 8.0F, 0.0F, 0.0F])
  assert_eq(shared_output[:].to_owned(), [32.0F, -32.0F, 0.0F, 0.0F])
  @bf16_operator_reference.shared_expert_into(
    @precision.SharedExpertPrecision::new(hidden_width=2, maximum_rows=2),
    routed_output[:],
    shared_output[:],
    [0.0F, 0.0F],
    output,
    rows=2,
  )
  assert_eq(output[:].to_owned(), [24.0F, -8.0F, 0.0F, 0.0F])
}
