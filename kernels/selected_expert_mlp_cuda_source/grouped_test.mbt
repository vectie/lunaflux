///|
fn grouped_fixture(
  format : @format.Format,
  law : @precision_ir.DenseDotLaw,
) -> @precision_ir.ExpertMlpPrecision raise {
  fn operand(n : Int, k : Int) -> @precision_ir.OperandPlan raise {
    @precision_ir.plan_operand(
      role=Weight,
      representation=@format.Representation::new(format~, rows=n, columns=k),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision_ir.Accuracy::new(absolute=0.02, relative=0.02),
      memory_ceiling=100000000L,
      tile_elements=32,
    )
  }
  @precision_ir.ExpertMlpPrecision::new(
    gate=operand(32, 64),
    up=operand(32, 64),
    down=operand(64, 32),
    expert_count=7,
    local_experts=[5, 1],
    selected_experts=2,
    maximum_rows=17,
    clamp=10.0,
    dot_law=law,
    memory_ceiling=100000000L,
  )
}

///|
test "export full-width grouped expert performance fixture" {
  if @env.get_env_var("LUNA_EXPORT_GROUPED_PERFORMANCE") != Some("1") {
    return
  }
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
      memory_ceiling=1073741824L,
      tile_elements=32,
    )
  }
  for
    (law, name) in [
      (@precision_ir.DenseDotLaw::ScalarFloat32, "scalar"),
      (MatrixFloat32, "matrix"),
    ] {
    let plan = @precision_ir.ExpertMlpPrecision::new(
      gate=operand(2048, 4096),
      up=operand(2048, 4096),
      down=operand(4096, 2048),
      expert_count=288,
      local_experts=Array::makei(32, i => i),
      selected_experts=8,
      maximum_rows=16,
      clamp=10.0,
      dot_law=law,
      memory_ceiling=1073741824L,
    )
    let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      plan,
      symbol=name,
      grouping?=if law == MatrixFloat32 &&
        @env.get_env_var("LUNA_GROUPED_SHARED_INPUT") == Some("1") {
        Some(
          @matrix.ExpertGroupingPlan::new(
            maximum_rows=16,
            selected_experts=8,
            shared_input_staging=@env.get_env_var("LUNA_GROUPED_SHARED_INPUT") ==
              Some("1"),
          ),
        )
      } else {
        None
      },
    )
    println("LUNA_SOURCE_BEGIN")
    println(@utf8.decode(source.source_bytes()))
    for i in 0..<source.symbols().length() {
      let (grid, block) = source.launches()[i]
      println(
        "static constexpr int \{name}_grid\{i}=\{grid},\{name}_block\{i}=\{block};",
      )
    }
    println(
      "static constexpr size_t \{name}_stride=\{plan.expert_stride()}ULL;",
    )
    let offsets = plan.weight_offsets()
    println(
      "static constexpr size_t \{name}_offsets[3]={\{offsets[0]}ULL,\{offsets[1]}ULL,\{offsets[2]}ULL};",
    )
    println(
      "static constexpr size_t \{name}_activation=\{source.activation_bytes()}ULL;",
    )
    println("LUNA_SOURCE_END")
  }
}

///|
test "grouped expert actual source and bounded startup scratch" {
  for (format, name) in [(@format.Format::NvFp4, "nv"), (Bf16, "bf")] {
    for
      (law, suffix) in [
        (@precision_ir.DenseDotLaw::ScalarFloat32, "scalar"),
        (MatrixFloat32, "matrix"),
        (ShapeAwareFloat32, "shape"),
      ] {
      let plan = grouped_fixture(format, law)
      let name = name + "_" + suffix
      let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
        plan,
        symbol=name,
      )
      let text = @utf8.decode(source.source_bytes())
      if law != ScalarFloat32 {
        assert_eq(source.symbols().length(), 4)
        assert_eq(source.activation_bytes(), 4492L)
        assert_true(text.contains("mma_sync"))
        assert_true(text.contains("groups+=(matched+15)/16"))
        assert_false(text.contains("cudaMalloc"))
      } else {
        assert_eq(source.activation_bytes(), 2176L)
      }
      if @env.get_env_var("LUNA_EXPORT_GROUPED_MATRIX") == Some("1") ||
        @env.get_env_var("LUNA_EXPORT_SHAPE_AWARE") == Some("1") {
        println("LUNA_SOURCE_BEGIN \{name}")
        println(text)
        for i in 0..<source.symbols().length() {
          let (grid, block) = source.launches()[i]
          println(
            "static constexpr int \{name}_grid\{i}=\{grid},\{name}_block\{i}=\{block};",
          )
        }
        println(
          "static constexpr size_t \{name}_stride=\{plan.expert_stride()}ULL;",
        )
        let offsets = plan.weight_offsets()
        println(
          "static constexpr size_t \{name}_offsets[3]={\{offsets[0]}ULL,\{offsets[1]}ULL,\{offsets[2]}ULL};",
        )
        println(
          "static constexpr size_t \{name}_activation=\{source.activation_bytes()}ULL;",
        )
        println("LUNA_SOURCE_END")
      }
    }
  }
}
