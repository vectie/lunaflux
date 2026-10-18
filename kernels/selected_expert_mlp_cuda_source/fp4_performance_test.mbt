///|
test "full-width native FP4 expert performance export" {
  let rows = 64
  let tails = @env.get_env_var("LUNA_FP4_TAILS") == Some("1")
  let wrapped = @env.get_env_var("LUNA_FP4_TAILS") == Some("wrapped")
  let hidden = if tails { 192 } else if wrapped { 704 } else { 4096 }
  let intermediate = if tails { 320 } else if wrapped { 576 } else { 1024 }
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
    // External matched-component fixture; ordinary exports retain no clamp.
    clamp?=if @env.get_env_var("LUNA_FP4_CLAMP") == Some("10") {
      Some(10.0)
    } else {
      None
    },
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
      fragment_slots=if @env.get_env_var("LUNA_FP4_FRAGMENT_SLOTS") == Some("2") {
        2
      } else {
        1
      },
      scale_partitions=if @env.get_env_var("LUNA_FP4_SCALE_PARTITIONS") ==
        Some("32") {
        32
      } else {
        1
      },
      transport=match @env.get_env_var("LUNA_FP4_TRANSPORT") {
        Some("tiles") =>
          PackedBulkTiles(columns=column_tile, consumer_teams=2, workers=48)
        Some("buffered") => IndexedBuffered
        Some("packed") => PackedRows
        Some("pipeline") => PackedPipeline
        Some("bulk") =>
          PackedBulk(columns=column_tile, consumer_teams=1, workers=48)
        Some("bulk2") =>
          PackedBulk(columns=column_tile, consumer_teams=2, workers=48)
        _ => IndexedRows
      },
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
  let ahead = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    p,
    symbol="read_ahead_check",
    grouping=@matrix.ExpertGroupingPlan::new(
      maximum_rows=rows,
      selected_experts=selected,
      fragment_slots=2,
    ),
  )
  assert_eq(ahead.activation_bytes(), ordinary.activation_bytes())
  // Check the actual preparation source on every normal test run, not only
  // the optional physical export. Checkpoint transfers initialize numerical
  // planes; alignment gaps between experts' operands are not readable input.
  if !tails && !wrapped {
    let tiled = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      p,
      symbol="padding_check",
      grouping=@matrix.ExpertGroupingPlan::new(
        maximum_rows=rows,
        selected_experts=selected,
        column_tile=128,
        transport=PackedBulkTiles(columns=128, consumer_teams=2, workers=48),
      ),
    )
    let text = @utf8.decode(tiled.source_bytes())
    assert_true(tiled.bank_preparation() is Some(_))
    assert_true(text.contains("target[destination]=readable?source[index]:0;"))
    for i, weight in weights {
      let layout = @precision_ir.packed_buffer_layout(weight.representation())
      let base = p.weight_offsets()[i]
      assert_true(
        text.contains(
          "readable=readable||(index>=\{base}ULL&&index<\{base+layout.buffer_bytes()}ULL);",
        ),
      )
    }
  }
  let ahead_source = @utf8.decode(ahead.source_bytes())
  assert_true(ahead_source.contains("ordered-operand-reads-v1;items=4;slots=2"))
  assert_false(
    @utf8.decode(ordinary.source_bytes()).contains("ordered-operand-reads-v1"),
  )
  let buffered = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    p,
    symbol="buffered_check",
    grouping=@matrix.ExpertGroupingPlan::new(
      maximum_rows=rows,
      selected_experts=selected,
      transport=IndexedBuffered,
    ),
  )
  assert_eq(buffered.symbols().length(), ordinary.symbols().length())
  assert_eq(buffered.activation_bytes(), ordinary.activation_bytes())
  assert_false(buffered.has_packed_rows())
  let buffered_source = @utf8.decode(buffered.source_bytes())
  assert_true(buffered_source.contains("cooperative operand ring=2"))
  assert_true(buffered_source.contains("unsigned char gate_up_operands[13824]"))
  assert_true(buffered_source.contains("unsigned char *ring=gate_up_operands"))
  assert_false(buffered_source.contains("mbarrier.try_wait"))
  for i in 0..<ordinary.launches().length() {
    assert_eq(buffered.launches()[i], ordinary.launches()[i])
  }
  for transport in [@matrix.PackedRows, PackedPipeline] {
    let packed = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
      p,
      symbol="transport_check",
      grouping=@matrix.ExpertGroupingPlan::new(
        maximum_rows=rows,
        selected_experts=selected,
        transport~,
      ),
    )
    assert_eq(packed.symbols().length(), 10)
    assert_true(packed.activation_bytes() > ordinary.activation_bytes())
    assert_true(
      @utf8.decode(packed.source_bytes()).contains("source_row=group*16+row"),
    )
    assert_true(
      @utf8.decode(packed.source_bytes()).contains(
        "bool valid=job_ids[row]>=0&&local>=0",
      ),
    )
    assert_eq(
      @utf8.decode(packed.source_bytes()).contains("barrier.arrive"),
      transport == PackedPipeline,
    )
  }
  assert_true(
    !@utf8.decode(ordinary.source_bytes()).contains("const unsigned *bp1="),
  )
  assert_true(!@utf8.decode(ordinary.source_bytes()).contains("fragment<"))
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
    if generated.bank_preparation() is Some(_) {
      println("#define LUNA_FP4_TILED_PLANES 1")
    }
    if generated.has_packed_rows() {
      println("#define LUNA_FP4_PACKED_ROWS 1")
      println(
        "static constexpr unsigned long long packed_base=\{(ordinary.activation_bytes()+15L)/16L*16L};",
      )
      println(
        "static constexpr unsigned long long metadata_offset=\{(p.workspace_bytes().0+3L)/4L*4L};",
      )
    }
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
    let physical = generated.quantized_workspace_bytes()
    println(
      "static constexpr unsigned long long physical_qb0=\{physical.0},physical_qb1=\{physical.1};",
    )
    for i, launch in generated.launches() {
      println(
        "static constexpr int grid\{i}=\{launch.0},block\{i}=\{launch.1};",
      )
    }
    println("LUNA_SOURCE_END")
  }
}
