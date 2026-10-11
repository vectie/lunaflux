///|
test "online law retains masked window slots and BF16 P before value accumulation" {
  let window = @precision.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=8,
    window_tokens=128,
    maximum_rows=144,
    maximum_positions=512,
    maximum_state_bytes=10000L,
    probability=OnlineBf16(block_tokens=64),
  )
  let simple = @window.window_shared_kv_source(
    window,
    symbol_prefix="online_window",
  )
  let simple_text = @utf8.decode(simple.source_bytes())
  assert_true(simple_text.contains("__float2bfloat16_rn(probability)"))
  assert_true(simple_text.contains("float maximum = -INFINITY"))
  assert_true(!simple_text.contains("float maximum = sink[head]"))
  for read_set in [@precision.CompressedRowReadSet::Learned(top_k=2), AllCausal] {
    let joined = @window_compressed_kv_cuda_source.window_compressed_kv_source(
      @precision.WindowCompressedKvPrecision::new(
        window,
        compression_ratio=4,
        read_set~,
        coordinates=CacheRelative,
      ),
      symbol_prefix="online_joined",
    )
    let text = @utf8.decode(joined.source_bytes())
    assert_true(text.contains("ordinal - window_span"))
    assert_true(!text.contains("ordinal - length"))
    assert_true(text.contains("__fmaf_rn(scores[ordinal]"))
    assert_true(text.contains("denominator = __fadd_rn(total, expf"))
    assert_true(!text.contains("}#include"))
  }
}

///|
test "offline online probability kernel exporter" {
  if @env.get_env_var("LUNA_EXPORT_ONLINE_PROBABILITY") != Some("1") {
    return
  }
  let window = @precision.WindowSharedKvPrecision::new(
    query_heads=2,
    head_width=8,
    window_tokens=128,
    maximum_rows=144,
    maximum_positions=512,
    maximum_state_bytes=10000L,
    probability=OnlineBf16(block_tokens=64),
  )
  println("LUNA_ONLINE_SOURCE_BEGIN")
  println(
    @utf8.decode(
      @window.window_shared_kv_source(window, symbol_prefix="online_window").source_bytes(),
    ),
  )
  for
    (prefix, read_set) in [
      ("online_learned", @precision.CompressedRowReadSet::Learned(top_k=2)),
      ("online_all", AllCausal),
    ] {
    println(
      @utf8.decode(
        @window_compressed_kv_cuda_source.window_compressed_kv_source(
          @precision.WindowCompressedKvPrecision::new(
            window,
            compression_ratio=4,
            read_set~,
            coordinates=CacheRelative,
          ),
          symbol_prefix=prefix,
        ).source_bytes(),
      ),
    )
  }
  println("LUNA_ONLINE_SOURCE_END")
}
