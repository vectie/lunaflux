///|
test "convolution lowering commits raw request history without synchronization" {
  let source = @causal_convolution_execution_cuda_source.causal_convolution_source(
    @precision.CausalConvolutionPrecision::new(
      channel_count=129,
      kernel_size=4,
      maximum_rows=8,
      maximum_sequences=2,
      cache_slots=4,
      maximum_state_bytes=3096L,
    ),
    symbol="request_conv",
  )
  assert_eq(source.launch(), (2, 2, 128))
  let text = @utf8.decode(source.source_bytes())
  assert_true(text.contains("slot=slots[sequence]"))
  assert_true(text.contains("past[2]=current"))
  assert_false(text.contains("__syncthreads"))
}

///|
test "export request convolution fixtures" {
  if @env.get_env_var("LUNA_EXPORT_CONVOLUTION") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN convolution")
  for
    (symbol, channels, kernel) in [
      ("conv_small", 8, 2),
      ("conv_tail", 129, 4),
      ("conv_glm", 8192, 4),
    ] {
    println(
      @utf8.decode(
        @causal_convolution_execution_cuda_source.causal_convolution_source(
          @precision.CausalConvolutionPrecision::new(
            channel_count=channels,
            kernel_size=kernel,
            maximum_rows=8,
            maximum_sequences=2,
            cache_slots=4,
            maximum_state_bytes=196608L,
          ),
          symbol~,
        ).source_bytes(),
      ),
    )
  }
  for
    (symbol, heads, dimension) in [
      ("delta_conv_small", 2, 4),
      ("delta_conv_tail", 1, 129),
      ("delta_conv_glm", 64, 128),
    ] {
    println(
      @utf8.decode(
        @delta.recurrent_delta_source(
          @precision.RecurrentDeltaPrecision::new(
            head_count=heads,
            key_dimension=dimension,
            value_dimension=dimension,
            maximum_rows=8,
            maximum_sequences=2,
            cache_slots=4,
            normalization_epsilon=0.000001,
            log_decay_lower_bound=-5.0,
            maximum_state_bytes=16777216L,
          ),
          symbol~,
        ).source_bytes(),
      ),
    )
  }
  println("LUNA_SOURCE_END")
}
