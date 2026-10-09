///|
fn dynamic_plan(
  shared? : Bool = false,
) -> @precision_ir.ExpertMlpPrecision raise {
  fn weight() -> @precision_ir.OperandPlan raise {
    @precision_ir.plan_operand(
      role=Weight,
      representation=if shared {
        @format.Representation::new(
          format=MxFp8E4M3,
          rows=128,
          columns=128,
          grouping=Block(rows=128, columns=128),
        )
      } else {
        @format.Representation::new(format=MxFp4, rows=128, columns=128)
      },
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=1000000L,
      tile_elements=32,
    )
  }
  fn quantization(rows : Int) -> @precision_ir.QuantizationPlan raise {
    @precision_ir.plan_operand(
      role=Activation,
      representation=@format.Representation::new(
        format=MxFp8E4M3,
        rows~,
        columns=128,
        grouping=Block(rows=1, columns=128),
      ),
      placement=QuantizeBeforeConsumer,
      supported_placements=[QuantizeBeforeConsumer],
      accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=1000000L,
    )
    .quantization()
    .with_amax_floor(0.0001)
  }
  @precision_ir.ExpertMlpPrecision::new(
    gate=weight(),
    up=weight(),
    down=weight(),
    expert_count=if shared { 1 } else { 4 },
    local_experts=if shared { [0] } else { [3, 1] },
    selected_experts=if shared { 1 } else { 2 },
    maximum_rows=2,
    clamp=10.0,
    activation_rounding=F32Product,
    routing_placement=BeforeDownInputRound,
    activation_quantization=(
      quantization(2),
      quantization(if shared { 2 } else { 4 }),
    ),
    memory_ceiling=1000000L,
  )
}

///|
test "dynamic expert lowering consumes quantization and numeric order" {
  let plan = dynamic_plan()
  let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    plan,
    symbol="dynamic",
  )
  assert_eq(Array::makei(5, i => source.symbols()[i]), [
    "dynamic_quantize_0", "dynamic_gate_up", "dynamic_quantize_1", "dynamic_down",
    "dynamic_combine",
  ])
  assert_eq(source.launches()[0], (2, 128))
  assert_eq(source.launches()[2], (4, 128))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("dynamic_a0(input,"))
  assert_true(text.contains("dynamic_a1(activation,"))
  assert_true(
    text.contains("product=__fmul_rn(product,selected_weights[job]);"),
  )
  assert_false(text.contains("const __nv_bfloat16 activated="))
  assert_true(text.contains("weighted[cell]=__float2bfloat16_rn(sum);"))
  assert_false(text.contains("DeepSeek"))
}
