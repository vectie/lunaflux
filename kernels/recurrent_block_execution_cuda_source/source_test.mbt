///|
fn fixture(
  hidden : Int,
  heads : Int,
  dimension : Int,
  rank : Int,
) -> @recurrent_block_execution_cuda_source.RecurrentBlockCudaSource raise {
  @recurrent_block_execution_cuda_source.recurrent_block_source(
    @precision.RecurrentBlockPrecision::new(
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
      @precision.CausalConvolutionPrecision::new(
        channel_count=heads * dimension,
        kernel_size=4,
        maximum_rows=8,
        maximum_sequences=2,
        cache_slots=4,
        maximum_state_bytes=196608L,
      ),
      hidden_width=hidden,
      control_rank=rank,
      output_normalization_epsilon=0.000001,
    ),
    symbol_prefix="block",
  )
}

///|
test "full recurrent source uses parallel projections and explicit intermediate rounds" {
  let source = fixture(6, 2, 4, 3)
  let text = @utf8.decode(source.source_bytes())
  assert_false(text.contains("blockIdx.x != 0 || threadIdx.x != 0"))
  assert_true(text.contains("__float2bfloat16_rn(accumulator)"))
  assert_true(text.contains("void block_6("))
  assert_true(text.contains("void block_14("))
  assert_eq(source.plan().frame_bytes(), 1312L)
}

///|
test "export complete recurrent fixture" {
  if @env.get_env_var("LUNA_EXPORT_RECURRENT_BLOCK") != Some("1") {
    return
  }
  println("LUNA_SOURCE_BEGIN block")
  println(@utf8.decode(fixture(6, 2, 4, 3).source_bytes()))
  println("LUNA_SOURCE_END")
}
