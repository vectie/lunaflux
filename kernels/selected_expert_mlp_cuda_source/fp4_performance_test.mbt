///|
test "full-width native FP4 expert performance export" {
  let rows = 64
  let tails = @env.get_env_var("LUNA_FP4_TAILS") == Some("1")
  let hidden = if tails { 192 } else { 4096 }
  let intermediate = if tails { 320 } else { 1024 }
  let selected = 8
  let full_bank = @env.get_env_var("LUNA_FP4_FULL_BANK") == Some("1")
  let bank_count = if full_bank { 288 } else { 8 }
  let weights = [
    (intermediate, hidden),
    (intermediate, hidden),
    (hidden, intermediate),
  ].map(s => {
    @precision_ir.plan_operand(
      role=Weight,
      representation=@format.Representation::new(
        format=NvFp4,
        rows=s.0,
        columns=s.1,
      ),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=268435456L,
      tile_elements=32,
    )
  })
  let quant = [(rows, hidden), (rows * selected, intermediate)].map(s => {
    @precision_ir.plan_operand(
      role=Activation,
      representation=@format.Representation::new(
        format=NvFp4,
        rows=s.0,
        columns=s.1,
      ),
      placement=QuantizeBeforeConsumer,
      supported_placements=[QuantizeBeforeConsumer],
      accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=268435456L,
    ).quantization()
  })
  let p = @precision_ir.ExpertMlpPrecision::new(
    gate=weights[0],
    up=weights[1],
    down=weights[2],
    expert_count=288,
    local_experts=Array::makei(bank_count, i => i),
    selected_experts=selected,
    maximum_rows=rows,
    dot_law=MatrixFloat32,
    scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
    activation_quantization=(quant[0], quant[1]),
    down_publication=ReduceFloat32BeforeRound,
    memory_ceiling=if full_bank { 4294967296L } else { 268435456L },
  )
  let variant = @env.get_env_var("LUNA_EXPORT_FP4_PERFORMANCE")
  let column_tile = match @env.get_env_var("LUNA_FP4_COLUMN_TILE") {
    Some("64") => 64
    Some("128") => 128
    _ => 32
  }
  let generated = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    p,
    symbol=if variant == Some("next") { "next" } else { "perf" },
    grouping=@matrix.ExpertGroupingPlan::new(
      maximum_rows=rows,
      selected_experts=selected,
      column_tile~,
      shared_input_staging=variant == Some("next") &&
        @env.get_env_var("LUNA_FP4_INDEPENDENT_INPUT") != Some("1"),
    ),
  )
  let alternate = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    p,
    symbol="alternate",
    grouping=@matrix.ExpertGroupingPlan::new(
      maximum_rows=rows,
      selected_experts=selected,
      shared_input_staging=true,
    ),
  )
  assert_true(@utf8.decode(alternate.source_bytes()).contains("tile[11520]"))
  try
    @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      p,
      symbol="unsupported_width",
      grouping=@matrix.ExpertGroupingPlan::new(
        maximum_rows=rows,
        selected_experts=selected,
        column_tile=96,
      ),
    )
  catch {
    @precision_ir.PrecisionPlanError::NoImplementation => ()
    other => fail(other.to_string())
  } noraise {
    _ => fail("unsupported output ownership silently ignored")
  }
  let ordinary = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    p,
    symbol="ordinary",
  )
  assert_true(
    !@utf8.decode(ordinary.source_bytes()).contains("const unsigned *bp1="),
  )
  assert_true(@utf8.decode(ordinary.source_bytes()).contains("fragment<2"))
  for i in 0..<ordinary.launches().length() {
    assert_eq(ordinary.launches()[i], alternate.launches()[i])
  }
  try
    @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      p,
      symbol="bad",
      grouping=@matrix.ExpertGroupingPlan::new(
        maximum_rows=rows - 1,
        selected_experts=selected,
        shared_input_staging=true,
      ),
    )
  catch {
    @precision_ir.PrecisionPlanError::NoImplementation => ()
    other => fail(other.to_string())
  } noraise {
    _ => fail("mismatched indirect capacity accepted")
  }
  if variant is Some(_) {
    println("LUNA_SOURCE_BEGIN fp4-performance")
    println("#define LUNA_FP4_BANK_COUNT \{bank_count}")
    println("static constexpr int column_tile=\{column_tile};")
    println(@utf8.decode(generated.source_bytes()))
    println(
      "static constexpr int R=\{rows},H=\{hidden},I=\{intermediate},S=\{selected};",
    )
    println(
      "static constexpr unsigned long long stride=\{p.expert_stride()},activation_bytes=\{generated.activation_bytes()};",
    )
    println(
      "static constexpr unsigned long long offsets[3]={\{p.weight_offsets()[0]},\{p.weight_offsets()[1]},\{p.weight_offsets()[2]}};",
    )
    for i, w in weights {
      let l = @precision_ir.packed_buffer_layout(w.representation())
      println(
        "static constexpr unsigned long long ws\{i}=\{l.scales_offset().unwrap()},wg\{i}=\{l.global_scale_offset().unwrap()};",
      )
    }
    for i, q in quant {
      let l = @precision_ir.packed_buffer_layout(q.operand().representation())
      println("static constexpr unsigned long long qb\{i}=\{l.buffer_bytes()};")
    }
    for i, launch in generated.launches() {
      println(
        "static constexpr int grid\{i}=\{launch.0},block\{i}=\{launch.1};",
      )
    }
    println("LUNA_SOURCE_END")
  }
}
