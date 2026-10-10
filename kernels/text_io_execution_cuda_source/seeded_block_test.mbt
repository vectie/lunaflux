///|
test "seed selection follows the live output vector rather than padded main rows" {
  let vector = @text_io_execution_cuda_source.seeded_token_block_source(
    @precision.SeededTokenBlockPrecision::new(
      vocabulary_size=17,
      block_rows=5,
      maximum_main_rows=8,
      maximum_seed_tokens=8,
      noise_token_id=16,
    ),
    symbol_prefix="vector_seed",
  )
  let source = @utf8.decode(vector)
  assert_true(source.contains("outputs=main_counts[4]"))
  assert_true(
    source.contains("outputs<=n && outputs<=8 ? seed[outputs-1] : -1"),
  )
  assert_false(source.contains("seed[n-1]"))
  let scalar = @utf8.decode(
    @text_io_execution_cuda_source.seeded_token_block_source(
      @precision.SeededTokenBlockPrecision::new(
        vocabulary_size=17,
        block_rows=5,
        maximum_main_rows=8,
        noise_token_id=16,
      ),
      symbol_prefix="scalar_seed",
    ),
  )
  assert_true(scalar.contains("token=seed[0]"))
  assert_false(scalar.contains("outputs=main_counts[4]"))
}
