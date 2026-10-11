///|
async test "original expert replay exports all six actual routed jobs" {
  fn weight(rows : Int, columns : Int) -> @precision_ir.OperandPlan raise {
    @precision_ir.plan_operand(
      role=Weight,
      representation=@format.Representation::new(format=MxFp4, rows~, columns~),
      placement=DecodeInOperandTile,
      supported_placements=[DecodeInOperandTile],
      accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=268435456L,
      tile_elements=32,
    )
  }
  fn activation(
    rows : Int,
    columns : Int,
  ) -> @precision_ir.QuantizationPlan raise {
    @precision_ir.plan_operand(
      role=Activation,
      representation=@format.Representation::new(
        format=MxFp8E4M3,
        rows~,
        columns~,
        grouping=Block(rows=1, columns=128),
      ),
      placement=QuantizeBeforeConsumer,
      supported_placements=[QuantizeBeforeConsumer],
      accuracy=@precision_ir.Accuracy::new(absolute=0.1, relative=0.1),
      memory_ceiling=268435456L,
    )
    .quantization()
    .with_amax_floor(0.0001)
  }
  let plan = @precision_ir.ExpertMlpPrecision::new(
    gate=weight(2048, 4096),
    up=weight(2048, 4096),
    down=weight(4096, 2048),
    expert_count=6,
    local_experts=[0, 1, 2, 3, 4, 5],
    selected_experts=6,
    maximum_rows=1,
    clamp=10.0,
    activation_rounding=F32Product,
    routing_placement=BeforeDownInputRound,
    activation_quantization=(activation(1, 4096), activation(6, 2048)),
    memory_ceiling=268435456L,
  )
  let source = @selected_expert_mlp_cuda_source.packed_expert_mlp_source(
    plan,
    symbol="matched_expert",
  )
  let offsets = plan.weight_offsets()
  let sizes = plan.quantized_workspace_bytes()
  let metadata = @utf8.encode(
    "\nstatic constexpr size_t expert_stride=\{plan.expert_stride()}ULL;\n" +
    "static constexpr size_t expert_offsets[3]={\{offsets[0]}ULL,\{offsets[1]}ULL,\{offsets[2]}ULL};\n" +
    "static constexpr size_t quantized_bytes[2]={\{sizes.0}ULL,\{sizes.1}ULL};\n",
  )
  assert_eq(source.launches().length(), 5)
  // Cover every activation block and every selected output cell, not just
  // one CTA per routed job.
  assert_eq(source.launches()[0], (32, 128))
  assert_eq(source.launches()[1], (3072, 128))
  assert_eq(source.launches()[2], (96, 128))
  assert_eq(source.launches()[3], (6144, 128))
  if @env.get_env_var("LUNAFLUX_MATCHED_EXPERT_SOURCE_OUTPUT") is Some(path) {
    @fs.write_file(
      path,
      source.source_bytes() + metadata,
      create_mode=CreateNew,
    )
  }
}
