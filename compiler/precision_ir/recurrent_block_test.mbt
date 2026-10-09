///|
test "recurrent block precision accounts every intermediate frame" {
  let delta = @precision_ir.RecurrentDeltaPrecision::new(
    head_count=64,
    key_dimension=128,
    value_dimension=128,
    maximum_rows=32,
    maximum_sequences=16,
    cache_slots=32,
    normalization_epsilon=0.000001,
    log_decay_lower_bound=-5.0,
    maximum_state_bytes=134217728L,
  )
  let conv = @precision_ir.CausalConvolutionPrecision::new(
    channel_count=8192,
    kernel_size=4,
    maximum_rows=32,
    maximum_sequences=16,
    cache_slots=32,
    maximum_state_bytes=1572864L,
  )
  let block = @precision_ir.RecurrentBlockPrecision::new(
    delta,
    conv,
    hidden_width=4096,
    control_rank=128,
    output_normalization_epsilon=0.000001,
  )
  assert_eq(block.frame_bytes(), 4743168L)
  assert_eq(block.projection_geometry(), (4096, 128))
}
