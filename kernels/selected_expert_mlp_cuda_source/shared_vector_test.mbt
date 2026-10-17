///|
test "export selected BF16 shared expert transport fixture" {
  guard @env.get_env_var("LUNA_EXPORT_SHARED_VECTOR") is Some(name) else {
    return
  }
  guard name is ("old" | "next") else { fail("diagnostic symbol") }
  fn operand(n : Int, k : Int) -> @precision_ir.OperandPlan raise {
    @precision_ir.plan_operand(
      role=Weight,
      representation=@format.Representation::new(format=Bf16, rows=n, columns=k),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision_ir.Accuracy::new(absolute=0.02, relative=0.02),
      memory_ceiling=1073741824L,
      tile_elements=32,
    )
  }
  let plan = @precision_ir.ExpertMlpPrecision::new(
    gate=operand(1024, 4096),
    up=operand(1024, 4096),
    down=operand(4096, 1024),
    expert_count=1,
    local_experts=[0],
    selected_experts=1,
    maximum_rows=64,
    clamp=10.0,
    dot_law=ShapeAwareFloat32,
    down_publication=ReduceFloat32BeforeRound,
    memory_ceiling=1073741824L,
  )
  let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    plan,
    symbol=name,
    grouping?=match @env.get_env_var("LUNA_SHARED_COLUMNS") {
      None => None
      Some(value) =>
        Some(
          @matrix.ExpertGroupingPlan::new(
            maximum_rows=64,
            selected_experts=1,
            shared_input_staging=true,
            matrix_columns=if value == "16" {
              16
            } else if value == "32" {
              32
            } else {
              64
            },
          ),
        )
    },
  )
  println("LUNA_SOURCE_BEGIN")
  println(@utf8.decode(source.source_bytes()))
  for i, (grid, block) in source.launches() {
    println(
      "static constexpr int \{name}_grid\{i}=\{grid},\{name}_block\{i}=\{block};",
    )
  }
  println(
    "static constexpr size_t \{name}_activation=\{source.activation_bytes()}ULL;",
  )
  println("LUNA_SOURCE_END")
}
