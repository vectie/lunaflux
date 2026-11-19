///|
async test "flagship text uses untied tables centered norm and selected row logits" {
  let spec = @model.Spec::parse(
    @fs.read_file("model/minicpm_v47/config.fixture.json").binary()[:],
  )
  let operators = @operators.DecoderOperators::new(spec, maximum_rows=32)
  let compilation = @attention.compile(
    operators,
    device_target=@catalog.DeviceTarget::new(
      compute_major=12,
      compute_minor=1,
      supports_bf16=true,
      supports_cublas_lt=false,
    ),
    target_id=@strategy.AttentionTargetId::new(
      "sm121-unqualified-software-profile",
    ),
    capabilities=@strategy.AttentionCompilerCapabilities::new(
      subgroup_width=32,
      max_threads_per_block=256,
      max_shared_memory_bytes=49152L,
      max_pipeline_stages=1,
      parallel_workgroup_target=1,
      supports_async_copy=false,
      matrix_shape=Some(
        @strategy.AttentionMatrixShape::new(rows=16, columns=16, reduction=16),
      ),
    ),
  )
  let stack = @decoder.Program::new(operators, compilation)
  let program = @minicpm_v47_text.Program::new(
    operators,
    stack,
    maximum_workspace_bytes=1048576L,
  )
  let text = program.text_source()
  let plan = text.plan()
  assert_eq(plan.geometry(), (248144, 2048, 1, 32, 1))
  assert_eq(plan.table_bytes(), 1016397824L)
  assert_eq(program.workspace_bytes(), 996672L)
  assert_eq(program.storage_budget().owned_bytes, 152271612L)
  assert_eq(program.storage_budget().output_workspace_bytes, 996672L)
  assert_eq(plan.normalized_bytes(), 4096L)
  assert_eq(plan.logits_bytes(), 992576L)
  assert_eq(plan.normalization_weight_law(), OnePlusGamma)
  assert_eq(plan.normalization_law(), F32NormalizeThenWeight)
  assert_eq(plan.logits_law(), Bf16Round)
  assert_eq(program.launch_count(Prefill), 924)
  assert_eq(program.launch_count(Decode), 924)
  assert_eq(text.egress_order(), [0, 1, 2])
  let gamma : @precision.NumericExpr = Binary(
    AddRN,
    Float32Bits(0x3f800000U),
    Convert(Float32, Input(1)),
  )
  assert_true(
    plan.expression(Weight).root() ==
    Convert(BFloat16, Binary(MultiplyRN, Input(0), gamma)),
  )
  let source = @utf8.decode(program.source_bytes()[:])
  for stage in 0..<4 {
    assert_true(source.contains("void " + text.symbol(stage) + "("))
  }
  assert_true(source.contains("const int input_row = selected[row]"))
  assert_true(
    source.contains("float* logits") || source.contains("float *logits"),
  )
  let frame = @minicpm_v47_text.Frame::new(program)
  frame.close()
  frame.close()
  let closed = try frame.logits() catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(closed)
  let storage = @minicpm_v47_text.Storage::new(program)
  storage.close()
  storage.close()
  let unavailable = try storage.metadata() catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(unavailable)
}
