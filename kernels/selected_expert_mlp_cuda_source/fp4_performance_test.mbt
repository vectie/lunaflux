///|
test "full-width native FP4 expert performance export" {
  let rows = 64
  let hidden = 4096
  let intermediate = 1024
  let selected = 8
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
    local_experts=[0, 1, 2, 3, 4, 5, 6, 7],
    selected_experts=selected,
    maximum_rows=rows,
    dot_law=MatrixFloat32,
    scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
    activation_quantization=(quant[0], quant[1]),
    down_publication=ReduceFloat32BeforeRound,
    memory_ceiling=268435456L,
  )
  let variant = @env.get_env_var("LUNA_EXPORT_FP4_PERFORMANCE")
  let generated = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    p,
    symbol=if variant == Some("next") { "next" } else { "perf" },
  )
  if variant is Some(_) {
    println("LUNA_SOURCE_BEGIN fp4-performance")
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
