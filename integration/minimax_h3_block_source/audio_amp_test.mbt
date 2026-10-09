///|
test "complete AMP source exposes shared activation and all convolution pairs" {
  let bundle = @minimax_h3_block_source.build_audio_amp(
    batch=2,
    channels=8,
    time=31,
    function_prefix="amp",
  )
  assert_eq(bundle.symbols.length(), 41)
  assert_eq(bundle.symbols[3], "amp_b0_p0_s0_norm")
  assert_eq(bundle.symbols[38], "amp_b2_p2_s1_conv")
  assert_eq(bundle.symbols[39], "amp_residual")
  assert_eq(bundle.symbols[40], "amp_average")
  let source = @utf8.decode(bundle.source)
  for symbol in bundle.symbols {
    assert_true(source.contains("void " + symbol + "("))
  }
  assert_true(source.contains("kernel_index * 5LL"))
  assert_true(source.contains("__fdiv_rn(sum012, 3.0f)"))
  let upsample = @minimax_h3_block_source.build_audio_upsample(
    batch=2,
    input_channels=16,
    output_channels=8,
    input_time=82800,
    kernel=4,
    stride=2,
    padding=1,
    function_prefix="up6",
  )
  let upsample_source = @utf8.decode(upsample.source)
  for symbol in upsample.symbols {
    assert_true(upsample_source.contains("void " + symbol + "("))
  }
  assert_true(upsample_source.contains("weight_g[input_channel]"))
  let tail = @minimax_h3_block_source.build_audio_tail(
    batch=2,
    channels=8,
    time=165600,
    function_prefix="tail",
  )
  let tail_source = @utf8.decode(tail.source)
  assert_eq(tail.symbols.length(), 5)
  for symbol in tail.symbols {
    assert_true(tail_source.contains("void " + symbol + "("))
  }
  assert_false(tail_source.contains("tanhf("))
  assert_false(tail_source.contains("bias[output_channel]"))
  assert_true(tail_source.contains("float sum = 0.0f"))
}
