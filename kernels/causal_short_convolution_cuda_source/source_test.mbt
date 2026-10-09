///|
test "causal short convolution preserves layout state and exact round points" {
  let rendered = depthwise_causal_silu_bf16_source(
    channel_count=8192,
    kernel_size=4,
    maximum_rows=32,
    maximum_sequences=8,
    function_symbol="short_conv_v1",
  )
  assert_eq(rendered.operation(), DepthwiseCausalSiluBf16)
  assert_eq(
    rendered.numerical_contract(),
    OrderedFloat32AccumulationSiluBf16RoundV1,
  )
  assert_eq(rendered.channel_count(), 8192)
  assert_eq(rendered.kernel_size(), 4)
  assert_eq(rendered.maximum_rows(), 32)
  assert_eq(rendered.maximum_sequences(), 8)
  let source = @utf8.decode(rendered.source_bytes())
  assert_true(source.contains("counts[1] != sequence_count - counts[0]"))
  assert_true(source.contains("sequence_offsets[sequence_count] != live_rows"))
  assert_true(source.contains("end <= begin || end > live_rows"))
  assert_true(source.contains("local_row - 3 + offset"))
  assert_true(source.contains("channel * 4ULL + offset"))
  assert_true(source.contains("sum = __fadd_rn(sum, __fmul_rn"))
  assert_true(source.contains("__fdiv_rn(sum, __fadd_rn(1.0f, expf(-sum)))"))
  assert_true(source.contains("source < 0 ? initial_state[i + length]"))
  assert_false(source.contains("bias"))
}

///|
test "causal short convolution rejects hostile geometry and symbol" {
  for geometry in [(0, 4, 32, 8), (8, 1, 32, 8), (8, 4, 0, 8), (8, 4, 4, 5)] {
    let (channels, kernel, rows, sequences) = geometry
    try
      depthwise_causal_silu_bf16_source(
        channel_count=channels,
        kernel_size=kernel,
        maximum_rows=rows,
        maximum_sequences=sequences,
        function_symbol="ok",
      )
    catch {
      InvalidChannelCount
      | InvalidKernelSize
      | InvalidRowCount
      | InvalidSequenceCount => ()
      _ => fail("unexpected geometry rejection")
    } noraise {
      _ => fail("hostile geometry admitted")
    }
  }
  try
    depthwise_causal_silu_bf16_source(
      channel_count=2147483647,
      kernel_size=4,
      maximum_rows=2147483647,
      maximum_sequences=2147483647,
      function_symbol="ok",
    )
  catch {
    DimensionOverflow => ()
    _ => fail("unexpected overflow rejection")
  } noraise {
    _ => fail("overflowing geometry admitted")
  }
  try
    depthwise_causal_silu_bf16_source(
      channel_count=8,
      kernel_size=4,
      maximum_rows=16,
      maximum_sequences=1,
      function_symbol="9bad",
    )
  catch {
    InvalidFunctionSymbol => ()
    _ => fail("unexpected symbol rejection")
  } noraise {
    _ => fail("hostile symbol admitted")
  }
}
