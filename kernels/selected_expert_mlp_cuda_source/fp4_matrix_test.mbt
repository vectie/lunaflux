///|
test "native low-bit matrix fragment export" {
  let source = @luna_cuda_precision.e2m1_scaled_matrix_helper(
    symbol="native_fp4",
  )
  assert_true(source.contains("scale_vec::4X.m16n8k64"))
  let weights = [(128, 64), (128, 64), (64, 128)].map(s => {
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
      memory_ceiling=1000000L,
      tile_elements=32,
    )
  })
  let quant = [(17, 64), (34, 128)].map(s => {
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
      memory_ceiling=1000000L,
    ).quantization()
  })
  let p = @precision_ir.ExpertMlpPrecision::new(
    gate=weights[0],
    up=weights[1],
    down=weights[2],
    expert_count=2,
    local_experts=[0, 1],
    selected_experts=2,
    maximum_rows=17,
    dot_law=MatrixFloat32,
    scaled_dot_law=BlockScaledMatrixFloat32(block_columns=16),
    activation_quantization=(quant[0], quant[1]),
    down_publication=if @env.get_env_var("LUNA_FP4_PARTIAL") == Some("1") {
      ReduceFloat32BeforeRound
    } else {
      LocalBFloat16
    },
    memory_ceiling=1000000L,
  )
  let generated = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    p,
    symbol="low",
  )
  assert_eq(generated.symbols().length(), 8)
  if @env.get_env_var("LUNA_EXPORT_FP4_MATRIX") == Some("1") {
    println("LUNA_SOURCE_BEGIN fp4-matrix")
    if p.down_publication() == ReduceFloat32BeforeRound {
      println("#define LUNA_PARTIAL 1")
    }
    println(source)
    println(@utf8.decode(generated.source_bytes()))
    println(
      "static constexpr unsigned long long low_stride=\{p.expert_stride()},low_activation_bytes=\{generated.activation_bytes()};",
    )
    let offsets = p.weight_offsets()
    println(
      "static constexpr unsigned long long low_offsets[3]={\{offsets[0]},\{offsets[1]},\{offsets[2]}};",
    )
    for i, w in weights {
      let layout = @precision_ir.packed_buffer_layout(w.representation())
      println(
        "static constexpr unsigned long long low_scale\{i}=\{layout.scales_offset().unwrap()},low_global\{i}=\{layout.global_scale_offset().unwrap()};",
      )
    }
    for i, q in quant {
      let layout = @precision_ir.packed_buffer_layout(
        q.operand().representation(),
      )
      println(
        "static constexpr unsigned long long q\{i}_bytes=\{layout.buffer_bytes()},q\{i}_scale=\{layout.scales_offset().unwrap()},q\{i}_global=\{layout.global_scale_offset().unwrap()};",
      )
    }
    println("LUNA_SOURCE_END")
  }
}
