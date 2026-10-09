///|
test "audio AMP cache keeps all eighteen normalized matrices disjoint" {
  for channels in [512, 256, 128, 64, 32, 16, 8] {
    let layout = @joint_diffusion_device.AudioAmpLayout::new(
      batch=2,
      channels~,
      time=207,
      max_bytes=1000000000L,
    )
    let (offsets, bytes) = audio_amp_cache_offsets(layout)
    assert_eq(offsets.length(), 18)
    assert_eq(offsets[0], 0L)
    for i in 0..<18 {
      let end = offsets[i] + layout.convolutions[i / 6].weight_bytes
      assert_eq(offsets[i] % 256L, 0L)
      assert_true(end <= bytes)
      if i < 17 {
        assert_true(end <= offsets[i + 1])
      } else {
        assert_eq(end, bytes)
      }
    }
    // Each branch has three pairs with two independently normalized weights.
    assert_eq(
      bytes,
      channels.to_int64() * channels.to_int64() * 4L * (3L + 7L + 11L) * 6L,
    )
  }
}

///|
test "audio empty weight cache cleanup is idempotent and not ready" {
  let cache = AudioWeightCache::new()
  assert_false(cache.ready)
  cache.close()
  cache.close()
  assert_true(cache.allocation is None)
  assert_true(cache.queue is None)
  assert_true(cache.source is None)
}
