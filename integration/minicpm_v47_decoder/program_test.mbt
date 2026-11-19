///|
async fn decoder_fixture() -> @minicpm_v47_decoder.Program {
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
  @minicpm_v47_decoder.Program::new(operators, compilation)
}

///|
async test "40-layer decoder partitions state and preserves both residual updates" {
  let program = decoder_fixture()
  assert_eq(program.operations(), [
    NormalizeInput,
    Branch,
    AddBranchResidual,
    NormalizeMoe,
    Moe,
    AddMoeResidual,
  ])
  assert_eq(program.layers().length(), 40)
  assert_eq(program.launch_count(Prefill), 920)
  assert_eq(program.launch_count(Decode), 920)
  let mut recurrent_end = 0L
  let mut convolution_end = 0L
  let mut key_end = 0L
  let mut full_count = 0
  let mut recurrent_count = 0
  for index, layer in program.layers() {
    assert_eq(layer.index, index)
    if index % 4 == 3 {
      assert_eq(layer.kind, @hybrid.LayerKind::FullAttention)
      assert_eq(layer.convolution_offset, None)
      assert_eq(layer.state_offset, key_end)
      key_end += 4194304L
      full_count += 1
    } else {
      assert_eq(layer.kind, @hybrid.LayerKind::GatedDeltaLinearAttention)
      assert_eq(layer.state_offset, recurrent_end)
      assert_eq(layer.convolution_offset, Some(convolution_end))
      recurrent_end += 2097152L
      convolution_end += 49152L
      recurrent_count += 1
    }
  }
  assert_eq((full_count, recurrent_count), (10, 30))
  let budget = program.budget()
  assert_eq(budget.activation_bytes, 131072L)
  assert_eq(budget.branch_workspace_bytes, 1845248L)
  // Keep the startup expert map alive while attention/recurrence reuse scratch.
  assert_eq(budget.moe_workspace_bytes, 1756224L)
  assert_eq(budget.convolution_state_bytes, convolution_end)
  assert_eq(budget.recurrent_state_bytes, recurrent_end)
  assert_eq(budget.key_cache_bytes, key_end)
  assert_eq(budget.value_cache_bytes, key_end)
  assert_eq(
    (recurrent_end, convolution_end, key_end),
    (62914560L, 1474560L, 41943040L),
  )
  let source = @utf8.decode(program.source_bytes()[:])
  assert_true(source.contains("void lunaflux_minicpm_v47_decoder_norm("))
  assert_true(source.contains("void lunaflux_minicpm_v47_decoder_residual("))
  let frame = @minicpm_v47_decoder.Frame::new(program)
  frame.close()
  frame.close()
  let closed = try frame.launches(Prefill) catch {
    _ => true
  } noraise {
    _ => false
  }
  assert_true(closed)
}
