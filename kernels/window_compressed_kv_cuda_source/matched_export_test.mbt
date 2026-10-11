///|
async test "export original-input compressed attention kernel" {
  let window = @precision.WindowSharedKvPrecision::new(
    query_heads=64,
    head_width=512,
    window_tokens=128,
    maximum_rows=144,
    maximum_positions=256,
    maximum_state_bytes=100000000L,
    probability=OnlineBf16(block_tokens=64),
  )
  let source = @window_compressed_kv_cuda_source.window_compressed_kv_source(
    @precision.WindowCompressedKvPrecision::new(
      window,
      compression_ratio=128,
      read_set=AllCausal,
      coordinates=CacheRelative,
    ),
    symbol_prefix="matched_compressed",
  )
  assert_eq(source.plan().compression_ratio(), 128)
  assert_eq(
    source.plan().window().probability_law(),
    OnlineBf16(block_tokens=64),
  )
  if @env.get_env_var("LUNA_MATCHED_COMPRESSED_ATTENTION_SOURCE")
    is Some(output) {
    @fs.write_file(output, source.source_bytes(), create_mode=CreateNew)
  }
}
