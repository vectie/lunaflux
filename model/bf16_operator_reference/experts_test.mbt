///|
fn tiny_experts(
  rounding : @precision.ExpertCombineRounding,
) -> @precision.ExpertMlpPrecision raise {
  let operand = @precision.plan_operand(
    role=Weight,
    representation=@format.Representation::new(format=Bf16, rows=2, columns=2),
    placement=DecodeInOperandTile,
    supported_placements=[DecodeInOperandTile],
    accuracy=@precision.Accuracy::new(absolute=0.0, relative=0.0),
    memory_ceiling=100000L,
    tile_elements=32,
  )
  @precision.ExpertMlpPrecision::new(
    gate=operand,
    up=operand,
    down=operand,
    expert_count=3,
    local_experts=[0, 1, 2],
    selected_experts=3,
    maximum_rows=2,
    combine_rounding=rounding,
    memory_ceiling=100000L,
  )
}

///|
test "packed experts combine by ascending ID with BF16 rounding after each contribution" {
  let staged = @bf16_operator_reference.Experts::new(tiny_experts(Bf16EachAdd))
  let float_sum = @bf16_operator_reference.Experts::new(
    tiny_experts(Float32Combine),
  )
  // SiLU(16) rounds to 16; one active intermediate gives 256, 1, -256.
  let gate_up = [
    16.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 16.0F, 0.0F, 0.0F, 0.0F, 1.0F,
    0.0F, 0.0F, 0.0F, 16.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
  ]
  let down = [
    16.0F,
    0.0F,
    0.0F,
    0.0F,
    0.0625F,
    0.0F,
    0.0F,
    0.0F,
    -16.0F,
    0.0F,
    0.0F,
    0.0F,
  ]
  let indices = [2, 1, 0, 2, 1, 0]
  let routes = [1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F]
  let actual = FixedArray::make(4, 0.0F)
  let other = FixedArray::make(4, 0.0F)
  staged.run_into(
    [1.0F, 0.0F, 0.0F, 1.0F],
    indices,
    routes,
    gate_up,
    down,
    actual,
    rows=2,
  )
  float_sum.run_into(
    [1.0F, 0.0F, 0.0F, 1.0F],
    indices,
    routes,
    gate_up,
    down,
    other,
    rows=2,
  )
  assert_eq(actual[:].to_owned(), [0.0F, 0.0F, 0.0F, 0.0F])
  assert_eq(other[:].to_owned(), [1.0F, 0.0F, 0.0F, 0.0F])
  // Reuse product scratch and preserve association of weights with selection slots.
  staged.run_into(
    [1.0F, 0.0F, 0.0F, 1.0F],
    indices,
    [0.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F],
    gate_up,
    down,
    actual,
    rows=2,
  )
  assert_eq(actual[:].to_owned(), [256.0F, 0.0F, 0.0F, 0.0F])
}

///|
test "shared sigmoid and product publish BF16 before routed addition" {
  let plan = @precision.SharedExpertPrecision::new(
    hidden_width=2,
    maximum_rows=2,
  )
  let output = FixedArray::make(4, 0.0F)
  @bf16_operator_reference.shared_expert_into(
    plan,
    [1.0F, -1.0F, 0.25F, 0.25F],
    [2.0F, 4.0F, 3.5F, -3.5F],
    [0.0F, 1.0F],
    output,
    rows=2,
  )
  assert_eq(output[:].to_owned(), [2.0F, 1.0F, 2.8125F, -2.3125F])
}
