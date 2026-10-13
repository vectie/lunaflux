///|
test "tensor parallel source delays expert rounding until after reduction" {
  fn operand(n : Int, k : Int) -> @precision_ir.OperandPlan raise {
    @precision_ir.plan_operand(
      role=Weight,
      representation=@format.Representation::new(
        format=NvFp4,
        rows=n,
        columns=k,
      ),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision_ir.Accuracy::new(absolute=0.02, relative=0.02),
      memory_ceiling=1000000L,
      tile_elements=32,
    )
  }
  for law in [@precision_ir.DenseDotLaw::ScalarFloat32, MatrixFloat32] {
    let full = @precision_ir.ExpertMlpPrecision::new(
      gate=operand(64, 64),
      up=operand(64, 64),
      down=operand(64, 64),
      expert_count=2,
      local_experts=[0, 1],
      selected_experts=2,
      maximum_rows=3,
      dot_law=law,
      memory_ceiling=1000000L,
    )
    let partition = @precision_ir.ExpertTensorPartition::new(
      full,
      rank=0,
      world_size=2,
      memory_ceiling=1000000L,
    )
    let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      partition.rank_plan(),
      symbol="partition",
    )
    let text = @utf8.decode(source.source_bytes())
    assert_true(text.contains("float *weighted"))
    assert_false(text.contains("__nv_bfloat16 *weighted"))
    if law == ScalarFloat32 {
      assert_true(text.contains("weighted[cell]=sum"))
    } else {
      assert_true(text.contains("local<0?0.0f:result[i]"))
    }
    assert_true(text.contains("selected_weights[row*2ULL+slot]"))
    assert_eq(partition.rank_plan().workspace_bytes().1, 1536L)
    if law == MatrixFloat32 &&
      @env.get_env_var("LUNA_EXPORT_EXPERT_TP") == Some("1") {
      for (plan, name) in [(full, "whole"), (partition.rank_plan(), "shard")] {
        let export = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
          plan,
          symbol=name,
        )
        println("LUNA_SOURCE_BEGIN")
        println(@utf8.decode(export.source_bytes()))
        for i, launch in export.launches() {
          println(
            "static constexpr int \{name}_grid\{i}=\{launch.0},\{name}_block\{i}=\{launch.1};",
          )
        }
        let offsets = plan.weight_offsets()
        println(
          "static constexpr size_t \{name}_offsets[3]={\{offsets[0]}ULL,\{offsets[1]}ULL,\{offsets[2]}ULL};",
        )
        println(
          "static constexpr size_t \{name}_stride=\{plan.expert_stride()}ULL,\{name}_activation=\{export.activation_bytes()}ULL;",
        )
        println("LUNA_SOURCE_END")
      }
    }
  }
}
