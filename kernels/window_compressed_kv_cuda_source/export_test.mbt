///|
test "export joint attention numerical probe kernels" {
  if @env.get_env_var("LUNA_EXPORT_JOINT_ATTENTION") != Some("1") {
    return
  }
  println("LUNA_JOINT_ATTENTION_SOURCE_BEGIN")
  for
    (prefix, rows, window, positions, ratio, read_set, coordinates) in [
      (
        "joint4",
        8,
        4,
        32,
        4,
        @precision.CompressedRowReadSet::Learned(top_k=3),
        @precision.CompressedRowCoordinates::OffsetRelative,
      ),
      ("joint128", 8, 4, 256, 128, AllCausal, OffsetRelative),
      ("joint0", 3, 3, 3, 4, Learned(top_k=3), OffsetRelative),
      ("joint4_cache", 8, 4, 32, 4, Learned(top_k=3), CacheRelative),
      ("joint128_cache", 8, 4, 256, 128, AllCausal, CacheRelative),
      ("joint0_cache", 3, 3, 3, 4, Learned(top_k=3), CacheRelative),
    ] {
    let plan = @precision.WindowCompressedKvPrecision::new(
      @precision.WindowSharedKvPrecision::new(
        query_heads=2,
        head_width=128,
        window_tokens=window,
        maximum_rows=rows,
        maximum_positions=positions,
        maximum_state_bytes=10000L,
      ),
      compression_ratio=ratio,
      read_set~,
      coordinates~,
    )
    println(
      @utf8.decode(
        @window_compressed_kv_cuda_source.window_compressed_kv_source(
          plan,
          symbol_prefix=prefix,
        ).source_bytes(),
      ),
    )
  }
  println("LUNA_JOINT_ATTENTION_SOURCE_END")
}
