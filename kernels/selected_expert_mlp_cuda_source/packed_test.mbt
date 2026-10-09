///|
fn packed_plan(
  format : @format.Format,
) -> @precision_ir.ExpertMlpPrecision raise {
  fn operand() -> @precision_ir.OperandPlan raise {
    @precision_ir.plan_operand(
      role=Weight,
      representation=@format.Representation::new(format~, rows=32, columns=32),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision_ir.Accuracy::new(absolute=0.02, relative=0.02),
      memory_ceiling=100000L,
      tile_elements=32,
    )
  }
  @precision_ir.ExpertMlpPrecision::new(
    gate=operand(),
    up=operand(),
    down=operand(),
    expert_count=4,
    local_experts=[3, 1],
    selected_experts=2,
    maximum_rows=2,
    clamp=10.0,
    memory_ceiling=100000L,
  )
}

///|
test "export compact expert execution fixtures" {
  if @env.get_env_var("LUNA_EXPORT_PACKED_EXPERT") == Some("1") {
    println("LUNA_SOURCE_BEGIN packed-expert")
    let formats : Array[(@format.Format, String)] = [
      (NvFp4, "nv"),
      (MxFp4, "mx"),
      (Bf16, "bf"),
    ]
    for (format, name) in formats {
      let plan = packed_plan(format)
      let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
        plan,
        symbol=name,
      )
      println(@utf8.decode(source.source_bytes()))
      let offsets = plan.weight_offsets()
      println(
        "static constexpr size_t \{name}_stride=\{plan.expert_stride()}ULL;",
      )
      println(
        "static constexpr size_t \{name}_offsets[3]={\{offsets[0]}ULL,\{offsets[1]}ULL,\{offsets[2]}ULL};",
      )
    }
    println("LUNA_SOURCE_END")
  }
}

///|
test "compact expert lowering emits separate linear-work stages from precision IR" {
  let formats : Array[@format.Format] = [NvFp4, MxFp4, Bf16]
  for format in formats {
    let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      packed_plan(format),
      symbol="packed",
    )
    assert_eq(Array::makei(3, i => source.symbols()[i]), [
      "packed_gate_up", "packed_down", "packed_combine",
    ])
    assert_eq(Array::makei(3, i => source.launches()[i]), [
      (32, 128),
      (32, 128),
      (1, 256),
    ])
    let text = @utf8.decode(source.source_bytes())
    assert_true(text.contains("__shfl_down_sync"))
    assert_true(text.contains("expert_map[expert]"))
    assert_true(text.contains("float *output"))
    assert_true(text.contains("fminf(__bfloat162float(gate),10.0f)"))
    assert_false(text.contains("cudaMalloc"))
    assert_false(text.contains("glm"))
  }
}
