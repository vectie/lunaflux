///|
test "limits reject every unsupported scalar envelope" {
  let cases : Array[(Int, Int, Int64, @device_layout.DeviceKvLimitRule)] = [
    (0, 1, 1L, TokensPerPage),
    (-1, 1, 1L, TokensPerPage),
    (1, 0, 1L, TotalPageCount),
    (1, -1, 1L, TotalPageCount),
    (1, 1, 0L, ArenaBytes),
    (1, 1, -1L, ArenaBytes),
  ]
  for case in cases {
    let (tokens, pages, maximum, expected) = case
    try
      @device_layout.DeviceKvLimits::new(
        tokens_per_page=tokens,
        total_page_count=pages,
        max_arena_bytes=maximum,
      )
    catch {
      InvalidLimit(actual) if actual == expected => ()
      _ => fail("unexpected device-KV limit error")
    } noraise {
      _ => fail("expected device-KV limit rejection")
    }
  }
}

///|
test "arena ceiling rejects the complete layout transactionally" {
  let geometry = layout_geometry(2, 2, 4)
  try
    @device_layout.plan(layout_identity(), geometry, layout_limits(4, 3, 3071L))
  catch {
    ArenaLimitExceeded => ()
    _ => fail("unexpected device-KV arena error")
  } noraise {
    _ => fail("expected exact arena ceiling rejection")
  }
}

///|
test "checked stride and arena arithmetic reject overflow" {
  let maximum = 9223372036854775807L
  try
    @device_layout.plan(
      layout_identity(),
      layout_geometry(1, 2147483647, 1),
      layout_limits(2147483647, 1, maximum),
    )
  catch {
    SizeOverflow(LayerStride) => ()
    _ => fail("unexpected layer-stride overflow stage")
  } noraise {
    _ => fail("expected layer-stride overflow")
  }

  try
    @device_layout.plan(
      layout_identity(),
      layout_geometry(2147483647, 1, 1),
      layout_limits(1, 2147483647, maximum),
    )
  catch {
    SizeOverflow(ArenaBytes) => ()
    _ => fail("unexpected arena overflow stage")
  } noraise {
    _ => fail("expected arena overflow")
  }
}
